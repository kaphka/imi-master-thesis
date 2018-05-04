import torch
import torch.utils.data as td
import numpy as np
from PIL import Image
from pathlib import Path
import datasets.divahisdb as diva
import random
from experiment.data import Environment
import logging

from data.metrics import mean_gradient

TILE_SET_NAME = 'tiles'
MIN_GRAD = 30

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

def make_img_set(tile_set, path):
    pass

class Tiles(td.Dataset):

    def __init__(self, path: Path, train=True, transforms=None ):
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

def create_tile_img_set(path, dataset, crop, img_transforms, filters, names ):
    for idx, data in enumerate(dataset):
        for item, transform, img_filter, name in zip(data, img_transforms, filters, names):
            img = transform(item)
            if img_filter(img):
                img.save(path / name / 'img{}.jpg'.format(str(idx)))
            else:
                pass

class RandCrop():
    def __init__(self, size, rnd=random.Random()):
        self.size = size
        self.rnd = rnd

    def __call__(self, img):
        left  = self.rnd.randint(0, img.width - self.size)
        upper = self.rnd.randint(0, img.height - self.size)
        return img.crop(box=(left, upper, left+self.size, upper + self.size))

class SaveTo():
    def __init__(self, path):
        self.path = path

    def __call__(self, img, name):
        logging.debug('saving to %s', str(self.path / name))
        img.save(self.path / name)



class ImgTiles(td.Dataset):

    def __init__(self, path: Path, split=diva.Splits.training.name, transforms=None ):
        self.path = path / split
        self.img_paths = list(self.path.glob('tile*.jpg'))
        self.gt_paths = list(self.path.glob('gt*.png'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        gt  = Image.open(self.gt_paths[item])
        return img, gt



def document_to_tile(dataset, path, size=255, use_grad_propabilty=False, crops_per_page=10):
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        crop = RandCrop(size)
        for idxcrop in range(crops_per_page):
            cropped = crop(img)
            grad  = mean_gradient(np.array(cropped.convert('L')))
            if grad > MIN_GRAD:
                cropped.save(path / 'img_{}_{}.jpg'.format(idx, idxcrop))
