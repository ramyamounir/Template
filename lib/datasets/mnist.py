import torch
import torchvision
import torchvision.io as IO
import torchvision.transforms as T

import joblib
from torch.utils.data import Dataset, DataLoader
from lib.core.config import MNIST_PATH


class MNIST(Dataset):

	def __init__(self):

		data = joblib.load('{}mnist.pkl'.format(MNIST_PATH))
		self.len = len(data)

	def __getitem__(self, index):

		return self.data[index]

	def __len__(self):
		return self.len


def get_loader(cfg):

	loader = DataLoader(dataset=DatasetGenerator(), 
							shuffle = cfg.TRAIN.SHUFFLE, 
							batch_size=cfg.TRAIN.BATCH_PER_GPU*cfg.GPU_COUNT, 
							num_workers= cfg.TRAIN.NUM_WORKERS
						)

	return loader