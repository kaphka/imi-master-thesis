import torch
import torch.utils.data as td
import numpy as np


def combine(dataset, path):
    x_train = None
    y_train = None
    for tiles, y in dataset:
        if x_train is None:
            x_train = tiles
            y_train = y
        else:
            x_train = np.concatenate((x_train, tiles))
            y_train = np.concatenate((y_train, y))
    np.savez_compressed(path / 'train', x=x_train, y=y_train)


class Tiles(td.Dataset):

    def __init__(self, path, train=True):
        self.path = path
        if train:
            self.data = np.load(path / 'train.npz')

    def __len__(self):
        return min(len(self.data['x']), len(self.data['y']))

    def __getitem__(self, item):
        return (self.data['x'][item], self.data['y'][item])