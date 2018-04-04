import torch
import torch.utils.data as td
import numpy as np
from PIL import Image

def combine(dataset, path, split='train', balance=False):
    x_train = None
    y_train = None
    for tiles, y in dataset:
        if x_train is None:
            x_train = tiles
            y_train = y
        else:
            x_train = np.concatenate((x_train, tiles))
            y_train = np.concatenate((y_train, y))
    if balance:
        print(sorted(np.unique(y_train, return_counts=True)[1]))
        n_label_max = sorted(np.unique(y_train, return_counts=True)[1])[-2]
        idx = np.where(y_train == 0)[0]
        np.random.shuffle(idx)
        idx = idx[:n_label_max]
        idx = np.append(idx, np.where(y_train == 1)[0][:n_label_max])
        idx = np.append(idx, np.where(y_train == 2)[0][:n_label_max])
        idx = np.append(idx, np.where(y_train == 3)[0][:n_label_max])
        x_train = x_train[idx]
        y_train = y_train[idx]
    np.savez_compressed(path / split, x=x_train, y=y_train.astype(np.int32))


class Tiles(td.Dataset):

    def __init__(self, path, train=True, transforms=None ):
        self.path = path
        self.transforms = transforms
        # TODO: Use enum
        if train:
            self.data = np.load(path / 'training.npz')
        else:
            self.data = np.load(path / 'validation.npz')

        self.x = self.data['x']
        self.y = self.data['y'].astype(np.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = self.x[item]
        img = Image.fromarray(img, mode='L')
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, self.y[item])