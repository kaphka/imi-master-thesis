import torch
import torch.utils.data
import torchvision
from glob import glob
import json
import os
from pathlib import Path
import random
from PIL import Image
from zipfile import ZipFile
import torchvision.transforms as transforms
from enum import IntFlag

import numpy as np

SET_NAMES = ['CB55', 'CSG18', 'CSG863']
SPLITS = ['public-test', 'training', 'validation']
NAME = 'DIVA-HisDB'
DATA_NAMES = ['img', 'PAGE-gt', 'pixel-level-gt']

class Annotations(IntFlag):
    BODY_TEXT = 0x000008
    DECORATION = 0x000004
    COMMENT = 0x000002
    BACKGROUND = 0x000001

# ANNOTATIONS = {
#         0x000008: 'main text body',
#         0x000004: 'decoration',
#         0x000002: 'comment',
#         0x000001: 'background',
#     }
BOUNDARY = 0x800000

def to_class_vector(flag, all=True):
    mask = 0xFFFFF
    return np.array(list(format(flag & mask, '04b')), dtype=np.ubyte)

def to_encoding(array, one_hot=False):
    encoding = np.zeros((array.shape[0], array.shape[1], len(Annotations)),dtype=np.ubyte)
    classes = list(Annotations)
    it = np.nditer(array, flags=['multi_index'])
    while not it.finished:
        annotation = Annotations(int(it.value))
        for c in classes:
            if bool(c & annotation):
                # print(c,annotation,bool(c & annotation),classes.index(c))
                encoding[it.multi_index[0],it.multi_index[1],int(classes.index(c))] = 1
        it.iternext()
    return encoding


def to_hex(array):
    """
    Effective hex conversion
    source https://stackoverflow.com/questions/26226354/numpy-image-arrays-how-to-efficiently-switch-from-rgb-to-hex

    :return:
    """
    array = np.asarray(array, dtype='uint32')
    return ((array[:, :, 0] << 16) + (array[:, :, 1] << 8) + array[:, :, 2])


def gtpath(p):
    if isinstance(p, str):
        p = Path(p)
    return p.parents[2] / 'pixel-level-gt' / 'training' / (p.stem + '.png')


class HisDBDataset(torch.utils.data.Dataset):

    def unzip(self, zippath, outpath):
        # TODO: unzip
        # assert zippath.exists()
        # assert outpath.exists()
        # zipfile = ZipFile(str(zippath))
        pass


    def __init__(self, path, transform=None, download=True, train=True, gt=False):
        self.gt = gt
        for set_name in SET_NAMES:
            folder = path / set_name
            folder.mkdir(exist_ok=True)
            for data_name in DATA_NAMES:
                data_folder = folder / data_name
                data_folder.mkdir(exist_ok=True)
                for split_name in SPLITS:
                    split_folder = data_folder / split_name
                    split_folder.mkdir(exist_ok=True)
            images = path / 'img-{}.zip'.format(set_name)
            imagesgt = path / 'pixel-level-gt-{}.zip'.format(set_name)
            self.unzip(images, folder)
            self.unzip(imagesgt, folder)
        if train:
            split = 'training'
        else: 
            split = 'validation'
            
        self.paths = glob(str(path / '*' / 'img' / split / '*.jpg'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        assert index < len(self)
        imgpath = self.paths[index]
        gt_file = str(gtpath(self.paths[index]))
        img =  Image.open(imgpath)
        arr = np.array(img)

        if self.gt:
            # only load one channel
            gt = Image.open(gt_file)
            gt_arr = to_hex(np.array(gt))
        else: 
            gt_arr = None

        return arr, gt_arr
