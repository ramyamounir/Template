import torch
import argparse
import os
from yacs.config import CfgNode as CN

SAVE_PATH = 'out/'
MNIST_PATH = 'data/mnist/'

cfg = CN()
cfg.MODEL_NAME = 'my_model'
cfg.OUT_DIR = 'out'
cfg.GPU_COUNT = 0
cfg.DEV = 'cuda'

cfg.TRAIN = CN()
cfg.TRAIN.BATCH_PER_GPU = 16
cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.SHUFFLE = True


def parse_args():

	# === CLI arguments parser === #
	parser = argparse.ArgumentParser()
	parser.add_argument('-cfg', type=str, help='config file path')
	parser.add_argument('-model', type=str, default = 'my_model', help ='model name') # model name
	parser.add_argument('-gpu', type=str, default="0", help = 'gpus to use') # gpu to be used
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
	GPU_count = torch.cuda.device_count()

	# === merge configs from file === #
	if args.cfg is not None:
		cfg.merge_from_file(args.cfg)
	
	# === merge configs from CLI arguments === #
	cfg.merge_from_list(["MODEL_NAME", args.model, "GPU_COUNT", GPU_count])

	print("{} GPU(s) visible. Batch size = {}".format(cfg.GPU_COUNT, cfg.TRAIN.BATCH_PER_GPU * cfg.GPU_COUNT))

	return cfg.clone()