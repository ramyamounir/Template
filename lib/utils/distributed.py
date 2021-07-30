import torch
import os, random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from collections import defaultdict, deque
import time, datetime, signal
import subprocess
from pathlib import Path
import submitit


"""
Misc functions.
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	import builtins as __builtin__
	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print

def fix_random_seeds(seed=31):
	"""
	Fix random seeds.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)

def get_shared_folder() -> Path:
	user = os.getenv("USER")
	if Path("/data/sarkar-vision/slurm_jobs/").is_dir():
		p = Path(f"/data/sarkar-vision/slurm_jobs/{user}")
		p.mkdir(exist_ok=True)
		return p
	raise RuntimeError("No shared folder available")


def init_dist_node(args):

	if 'SLURM_JOB_ID' in os.environ:

		args.ngpus_per_node = torch.cuda.device_count()

		# requeue job on SLURM preemption
		signal.signal(signal.SIGUSR1, handle_sigusr1)
		signal.signal(signal.SIGTERM, handle_sigterm)

		# find a common host name on all nodes
		cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
		stdout = subprocess.check_output(cmd.split())
		host_name = stdout.decode().splitlines()[0]
		args.dist_url = f'tcp://{host_name}:{random.randint(49152,65535)}'

		# distributed parameters
		args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
		args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
		
	else:

		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
		args.ngpus_per_node = torch.cuda.device_count()

		args.rank = 0
		args.dist_url = f'tcp://localhost:{random.randint(49152,65535)}'
		args.world_size = args.ngpus_per_node

def init_dist_gpu(gpu, args):

	if args.slurm:
		job_env = submitit.JobEnvironment()
		args.output_dir = Path(str(args.output_dir).replace("%j", str(job_env.job_id)))
		args.gpu = job_env.local_rank
		args.rank = job_env.global_rank
	else:
		args.gpu = gpu
		args.rank += gpu

	dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
	fix_random_seeds()
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	dist.barrier()

	args.main = (args.rank == 0)
	setup_for_distributed(args.main)
	
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
	"""Track a series of values and provide access to smoothed values over a
	window or the global series average.
	"""

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = "{median:.6f} ({global_avg:.6f})"
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""
		Warning: does not synchronize the deque!
		"""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value)


class MetricLogger(object):
	def __init__(self, delimiter="\t"):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError("'{}' object has no attribute '{}'".format(
			type(self).__name__, attr))

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {}".format(name, str(meter))
			)
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.6f}')
		data_time = SmoothedValue(fmt='{avg:.6f}')
		space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
		if torch.cuda.is_available():
			log_msg = self.delimiter.join([
				header,
				'[{0' + space_fmt + '}/{1}]',
				'eta: {eta}',
				'{meters}',
				'time: {time}',
				'data: {data}',
				'max mem: {memory:.0f}'
			])
		else:
			log_msg = self.delimiter.join([
				header,
				'[{0' + space_fmt + '}/{1}]',
				'eta: {eta}',
				'{meters}',
				'time: {time}',
				'data: {data}'
			])
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj
			iter_time.update(time.time() - end)
			if i % print_freq == 0 or i == len(iterable) - 1:
				eta_seconds = iter_time.global_avg * (len(iterable) - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time),
						memory=torch.cuda.max_memory_allocated() / MB))
				else:
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time)))
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('{} Total time: {} ({:.6f} s / it)'.format(
			header, total_time_str, total_time / len(iterable)))