import torch
import torchvision
import torchvision.io as IO
import torchvision.transforms as T

import joblib
from torch.utils.data import Dataset, DataLoader
from lib.core.config import DATA_PATH


class DatasetGenerator(Dataset):

	def __init__(self, cfg, split):

		data = joblib.load('{}/{}/mnist.pkl'.format(DATA_PATH, cfg.DATASET))

		if split == 'train':
			data = data['train']
		elif split == 'valid':
			data = data['valid']

		self.len = len(data)

	def __getitem__(self, index):

		return self.data[index]

	def __len__(self):
		return self.len


def get_loader(cfg):

	train_loader = DataLoader(dataset=DatasetGenerator(cfg, split = 'train' ), 
							shuffle = cfg.TRAIN.SHUFFLE, 
							batch_size=cfg.TRAIN.BATCH_PER_GPU*cfg.GPU_COUNT, 
							num_workers= cfg.TRAIN.NUM_WORKERS
						)

	valid_loader = DataLoader(dataset=DatasetGenerator(cfg, split = 'valid'), 
							shuffle = cfg.TRAIN.SHUFFLE, 
							batch_size=cfg.TRAIN.BATCH_PER_GPU*cfg.GPU_COUNT, 
							num_workers= cfg.TRAIN.NUM_WORKERS
						)

	return (train_loader, valid_loader)