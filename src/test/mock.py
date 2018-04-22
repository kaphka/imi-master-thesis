from torch.utils.data import Dataset
import numpy as np
import torch

class RandDataset(Dataset):

    def __init__(self, size=10, img_size=32, channels= 3, labels=3):
        self.size = size
        self.img_size = img_size
        self.channels = channels
        self.y = torch.zeros((self.size)).long()
        for idx in range(size):
            self.y[idx] = int(idx / (size / labels))

    def __len__(self):
        return  self.size

    def __getitem__(self, index):
        x = torch.randn(( self.channels, self.img_size, self.img_size))
        return x * (self.y[index] + 1), self.y[index]