import torch
import torchvision.io as IO
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):

    def __init__(self, args):
        self.args = args

    def __getitem__(self, index):
        return torch.randn(1024), torch.randn(1024)

    def __len__(self):
        return 100_000


def get_dataset(args):

    # === Get Dataset === #
    train_dataset = DatasetGenerator(args)

    return train_dataset
