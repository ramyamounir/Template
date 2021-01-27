import torch
import argparse
import os
from yacs.config import CfgNode as CN

SAVE_PATH = 'out'
DATA_PATH = 'data'

cfg = CN()
cfg.MODEL_NAME = 'my_model'
cfg.OUT_DIR = 'out'
cfg.GPU_COUNT = 0
cfg.DEV = 'cuda'
cfg.START_TB = False
cfg.DATASET = 'cifar'

cfg.TRAIN = CN()
cfg.TRAIN.BATCH_PER_GPU = 16
cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.SHUFFLE = True
cfg.TRAIN.NUM_EPOCHS = 1000
cfg.TRAIN.SAVE_EVERY = 10

cfg.LOSS = CN()
cfg.LOSS.REDUCE = 'mean'
cfg.LOSS.FN = 'CE'

cfg.ACCURACY = CN()
cfg.ACCURACY.FN = 'custom'

cfg.OPT = CN()
cfg.OPT.FN = 'adam'
cfg.OPT.LR = 1e-5
cfg.OPT.MOM = 0
cfg.OPT.LR_DECAY = 0
cfg.OPT.BETAS = (0.9, 0.999)

cfg.SCHED = CN()
cfg.SCHED.FN = 'reduceonplateau'
cfg.SCHED.STEP = 10
cfg.SCHED.GAMMA = 0.1
cfg.SCHED.PATIENCE = 10


def parse_args():

	# === CLI arguments parser === #
	parser = argparse.ArgumentParser()
	parser.add_argument('-cfg', type=str, help='config file path')
	parser.add_argument('-model', type=str, default = 'my_model', help ='model name')
	parser.add_argument('-gpu', type=str, default="0", help = 'gpus to use')
	parser.add_argument('-tb', default=False, action ="store_true", help = 'start tensorboard server') 
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
	GPU_count = torch.cuda.device_count()

	# === merge configs from file === #
	if args.cfg is not None:
		cfg.merge_from_file(args.cfg)
	
	# === merge configs from CLI arguments === #
	cfg.merge_from_list(["MODEL_NAME", args.model, "GPU_COUNT", GPU_count, "START_TB", args.tb])

	print("{} GPU(s) visible. Batch size = {}".format(cfg.GPU_COUNT, cfg.TRAIN.BATCH_PER_GPU * cfg.GPU_COUNT))

	return cfg.clone()