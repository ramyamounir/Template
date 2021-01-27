import torch
import torchvision
import torchvision.io as IO
import torchvision.transforms as T

import joblib, os
from torch.utils.data import Dataset, DataLoader
from lib.utils.file import checkdir
from lib.core.config import DATA_PATH


class DatasetGenerator(Dataset):

	def __init__(self, cfg, split):

		pkl_path = '{}/{}/mnist.pkl'.format(DATA_PATH, cfg.DATASET)

		if os.path.exists(pkl_path):
			self.data = joblib.load(pkl_path)
		else:
			print("Preparing dataset...")
			self.data = self.prepare_dataset(pkl_path)


		if split == 'train':
			self.data = self.data['train']
		elif split == 'valid':
			self.data = self.data['valid']

		self.len = len(self.data)

	def __getitem__(self, index):

		return self.data[index]

	def __len__(self):
		return self.len

	def prepare_dataset(self, pkl_path):
		d={"train":[], "test":[]}
		checkdir(os.path.dirname(pkl_path))
		joblib.dump(d, pkl_path)
		return d


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