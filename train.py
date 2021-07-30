import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from lib.utils.file import bool_flag
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder

import submitit
from pathlib import Path


def parse_args():

	parser = argparse.ArgumentParser(description='Template')

	# === PATHS === #
	parser.add_argument('-data', type=str, default="data",
						help='path to dataset directory')
	parser.add_argument('-out', type=str, default="out",
						help='path to out directory')

	# === GENERAL === #
	parser.add_argument('-model', type=str, default="my_model",
						help='Model name')
	parser.add_argument('-reset', action='store_true',
						help='Reset saved model logs and weights')
	parser.add_argument('-tb', action='store_true',
						help='Start TensorBoard')
	parser.add_argument('-gpus', type=str, default="0",
						help='GPUs list, only works if not on slurm')

	# === Dataset === #
	parser.add_argument('-dataset', type=str, default = 'random',
						help='Dataset to choose')
	parser.add_argument('-batch_per_gpu', type=int, default = 96,
						help='batch size per gpu')
	parser.add_argument('-shuffle', type=bool_flag, default = True,
						help='Shuffle dataset')
	parser.add_argument('-workers', type=int, default = 2,
						help='number of workers')

	# === Architecture === #
	parser.add_argument('-arch', type=str, default = 'mlp',
						help='Architecture to choose')

	# === Trainer === #
	parser.add_argument('-trainer', type=str, default = 'trainer',
						help='Trainer to choose')
	parser.add_argument('-epochs', type=int, default = 1000,
						help='number of epochs')
	parser.add_argument('-save_every', type=int, default = 10,
						help='Save frequency')
	parser.add_argument('-fp16', action='store_true',
						help='Use mixed precision only with RTX, V100 and A100 GPUs')


	# === Optimization === #
	parser.add_argument('-optimizer', type=str, default = 'adam',
						help='Optimizer function to choose')
	parser.add_argument('-lr_start', type=float, default = 5e-4,
						help='Initial Learning Rate')
	parser.add_argument('-lr_end', type=float, default = 1e-6,
						help='Final Learning Rate')
	parser.add_argument('-lr_warmup', type=int, default = 10,
						help='warmup epochs for learning rate')

	# === SLURM === #
	parser.add_argument('-slurm', action='store_true',
						help='Submit with slurm')
	parser.add_argument('-slurm_ngpus', type=int, default = 8,
						help='num of gpus per node')
	parser.add_argument('-slurm_nnodes', type=int, default = 2,
						help='number of nodes')
	parser.add_argument('-slurm_nodeslist', default = None,
						help='slurm nodeslist. i.e. "GPU17,GPU18"')
	parser.add_argument('-slurm_partition', type=str, default = "general",
						help='slurm partition')
	parser.add_argument('-slurm_timeout', type=int, default = 2800,
						help='slurm timeout minimum, reduce if running on the "Quick" partition')

	args = parser.parse_args()
	return args


class SLURM_Trainer(object):
	def __init__(self, args):
		self.args = args

	def __call__(self):

		init_dist_node(self.args)
		train(None, self.args)


def main():

	args = parse_args()
	
	if args.slurm:

		args.output_dir = get_shared_folder() / "%j"
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
		executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

		executor.update_parameters(
			mem_gb=12*args.slurm_ngpus,
			gpus_per_node=args.slurm_ngpus,
			tasks_per_node=args.slurm_ngpus,
			cpus_per_task=2,
			nodes=args.slurm_nnodes,
			timeout_min=2800,
			slurm_partition=args.slurm_partition
		)

		if args.slurm_nodeslist:
			executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

		executor.update_parameters(name=args.model)
		trainer = SLURM_Trainer(args)
		job = executor.submit(trainer)
		print(f"Submitted job_id: {job.job_id}")


	else:
		init_dist_node(args)
		mp.spawn(train, args = (args,), nprocs = args.ngpus_per_node)
	

def train(gpu, args):

	# === SET ENV === #
	init_dist_gpu(gpu, args)
	
	# === DATA === #
	get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset), fromlist=["get_dataset"]), "get_dataset")
	dataset = get_dataset(args)

	sampler = DistributedSampler(dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
	loader = DataLoader(dataset=dataset, 
							sampler = sampler,
							batch_size=args.batch_per_gpu, 
							num_workers= args.workers,
							pin_memory = True,
							drop_last = True
						)
	print(f"Data loaded")

	# === MODEL === #
	get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
	model = get_model(args).cuda(args.gpu)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
	model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

	# === LOSS === #
	from lib.core.loss import get_loss
	loss = get_loss(args).cuda(args.gpu)

	# === OPTIMIZER === #
	from lib.core.optimizer import get_optimizer
	optimizer = get_optimizer(model, args)

	# === TRAINING === #
	Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer), fromlist=["Trainer"]), "Trainer")
	Trainer(args, loader, model, loss, optimizer).fit()


if __name__ == "__main__":
	main()