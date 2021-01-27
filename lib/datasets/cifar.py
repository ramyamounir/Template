import torchvision.io as IO
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import joblib, os, wget
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

from lib.utils.file import checkdir, extract, unpickle
from lib.core.config import DATA_PATH

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def download_dataset(path):
	checkdir(path)

	# Download here
	download_path = wget.download(CIFAR_URL, path)
	print("\n")

	return download_path

def extract_dataset(file, from_path, to_path):

	checkdir(to_path)
	extract(file, to_path)


def prepare_dataset(extracted_path, prepared_path):

	data = {"train":[], "valid":[], "test":[]}

	checkdir(prepared_path+"/train/")
	checkdir(prepared_path+"/valid/")
	checkdir(prepared_path+"/test/")

	img_counter = 1
	folder = 'train'
	for batch in tqdm(sorted(glob(extracted_path+"/*/data_batch_*"))):
		
		d = unpickle(batch, encoding='bytes')
		imgs = d['data'.encode("utf-8")]
		labels = d['labels'.encode("utf-8")]

		imgs = np.array(imgs)
		imgs_r = imgs[:,:1024].reshape((-1,32,32))
		imgs_g = imgs[:,1024:-1024].reshape((-1,32,32))
		imgs_b = imgs[:,-1024:].reshape((-1,32,32))

		for img in range(imgs.shape[0]):

			stacked = np.stack([imgs_b[img], imgs_g[img], imgs_r[img]]).transpose((1,2,0))
			to_save = "{}/{}/{}.jpg".format(prepared_path, folder, str(img_counter).zfill(6))
			cv2.imwrite(to_save, stacked)
			data[folder].append([to_save, labels[img]])
			
			if img_counter == 40000:
				folder = 'valid'
			img_counter += 1



	img_counter = 1
	folder = 'test'
	for batch in tqdm(sorted(glob(extracted_path+"/*/test_batch"))):
		
		d = unpickle(batch, encoding='bytes')
		imgs = d['data'.encode("utf-8")]
		labels = d['labels'.encode("utf-8")]

		imgs = np.array(imgs)
		imgs_r = imgs[:,:1024].reshape((-1,32,32))
		imgs_g = imgs[:,1024:-1024].reshape((-1,32,32))
		imgs_b = imgs[:,-1024:].reshape((-1,32,32))

		for img in range(imgs.shape[0]):

			stacked = np.stack([imgs_b[img], imgs_g[img], imgs_r[img]]).transpose((1,2,0))
			to_save = "{}/{}/{}.jpg".format(prepared_path, folder, str(img_counter).zfill(6))
			cv2.imwrite(to_save, stacked)
			data[folder].append([to_save, labels[img]])

			img_counter += 1

	return data


class DatasetGenerator(Dataset):

	def __init__(self, cfg, data):

		self.data = data
		self.len = len(self.data)

	def __getitem__(self, index):

		img = IO.read_image(self.data[index][0])
		label = self.data[index][1]
		return img, label

	def __len__(self):
		return self.len

	


def get_loader(cfg):

	# === Get Dataset === #
	dataset_path = '{}/{}/files/'.format(DATA_PATH, cfg.DATASET)

	if not os.path.exists(dataset_path):
		print("Downloading dataset...")
		downloaded_file = download_dataset(dataset_path)

	# === Extract Dataset === #
	extracted_path = '{}/{}/extracted/'.format(DATA_PATH, cfg.DATASET)
	
	if not os.path.exists(extracted_path):
		print("Extracting dataset...")
		extract_dataset(downloaded_file, dataset_path, extracted_path)


	# === Prepare Dataset === #
	prepared_path = '{}/{}/prepared/'.format(DATA_PATH, cfg.DATASET, cfg.DATASET)

	if os.path.exists(prepared_path+"{}.pkl".format(cfg.DATASET)):
		data = joblib.load(prepared_path+"{}.pkl".format(cfg.DATASET))
	else:
		print("Preparing dataset...")
		data = prepare_dataset(extracted_path, prepared_path)
		joblib.dump(data, prepared_path+"{}.pkl".format(cfg.DATASET))


	# === Create Loaders === #
	train_loader = DataLoader(dataset=DatasetGenerator(cfg, data['train']), 
							shuffle = cfg.TRAIN.SHUFFLE, 
							batch_size=cfg.TRAIN.BATCH_PER_GPU*cfg.GPU_COUNT, 
							num_workers= cfg.TRAIN.NUM_WORKERS
						)

	valid_loader = DataLoader(dataset=DatasetGenerator(cfg, data['valid']), 
							shuffle = cfg.TRAIN.SHUFFLE, 
							batch_size=cfg.TRAIN.BATCH_PER_GPU*cfg.GPU_COUNT, 
							num_workers= cfg.TRAIN.NUM_WORKERS
						)


	return train_loader, valid_loader