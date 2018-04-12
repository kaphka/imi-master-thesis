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
from enum import IntFlag, IntEnum, Enum
import logging

import numpy as np

SET_NAMES = ['CB55', 'CSG18', 'CSG863']
DATA_NAMES = ['img', 'PAGE-gt', 'pixel-level-gt']
SPLITS = ['training', 'validation', 'public-test']
NAME = 'DIVA-HisDB'


class Splits(Enum):
    training = 'training'
    validation = 'validation'
    test = 'public-test'

class Annotations(IntFlag):
    BODY_TEXT = 0x000008
    DECORATION = 0x000004
    COMMENT = 0x000002
    BACKGROUND = 0x000001



# class Labels(Annotations):
#     BODY_DECORATION = Annotations.BODY_TEXT|Annotations.DECORATION
#     COMMENT_DECORATION = Annotations.COMMENT|Annotations.DECORATION

BOUNDARY = 0x800000

LABEL_DICT = {
    0: Annotations.BACKGROUND,
    1: Annotations.DECORATION,
    2: Annotations.COMMENT,
    3: Annotations.BODY_TEXT
}
LABEL_COLORS = {
    0: [255, 255, 255],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
}


def label_to_code(label):
    return LABEL_DICT[label]

def numeric_gt(gt, label_boundary=False):
    gt_labels = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.ubyte)
    gt_labels[gt == (Annotations.BACKGROUND)] = 0
    gt_labels[gt == (Annotations.DECORATION)] = 1
    gt_labels[gt == (Annotations.COMMENT)] = 2
    gt_labels[gt == (Annotations.BODY_TEXT)] = 3
    if label_boundary:
        gt_labels[gt == (Annotations.DECORATION | BOUNDARY)] = 1
        gt_labels[gt == (Annotations.COMMENT | BOUNDARY)] = 2
        gt_labels[gt == (Annotations.BODY_TEXT | BOUNDARY)] = 3
    return  gt_labels


def binary_gt(gt):
    gt_bin_labels = np.zeros(gt.shape + (4,), dtype=np.float)
    gt_bin_labels[gt == (Annotations.BACKGROUND)] += [1, 0, 0, 0]
    gt_bin_labels[gt == (Annotations.DECORATION)] += [0, 1, 0, 0]
    gt_bin_labels[gt == (Annotations.COMMENT)]    += [0, 0, 1, 0]
    gt_bin_labels[gt == (Annotations.BODY_TEXT) ] += [0, 0, 0, 1]
    return  gt_bin_labels


def to_class_vector(flags, all=True):
    masked = flags & 0xFFFFF
    encoding = np.zeros((len(masked), len(Annotations)), dtype=np.ubyte)
    for idx, anno in enumerate(Annotations):
        encoding[..., idx] = (anno & masked).astype(bool).astype(np.ubyte)
    return encoding

def to_encoding(array, one_hot=False):
    # encoding = np.zeros((array.shape[0], array.shape[1], len(Annotations)),dtype=np.ubyte)
    shape = array.shape
    encoding = array.reshape((-1, 1))[:,0]
    encoding = to_class_vector(encoding)
    encoding = encoding.reshape((shape[0], shape[1], -1))
    return encoding


def to_hex(array):
    """
    Effective hex conversion
    source https://stackoverflow.com/questions/26226354/numpy-image-arrays-how-to-efficiently-switch-from-rgb-to-hex

    :return:
    """
    array = np.asarray(array, dtype='uint32')
    return ((array[:, :, 0] << 16) + (array[:, :, 1] << 8) + array[:, :, 2])

class DIVAPath(IntEnum):
    split = 0
    data_format = 1
    set = 2

def change_diva_path(p, set=None, split=None, ext=None, data_format=None):
    """
    Change diva path parameters:
        path/Set/data_format/split
    :param p:
    :param set:
    :param split:
    :param ext:
    :return:
    """
    if isinstance(p, str):
        p = Path(p)
    if not split:
        split = p.parents[DIVAPath.split].name
    if not data_format:
        data_format = p.parents[DIVAPath.data_format].name
    if not set:
        set = p.parents[DIVAPath.set].name
    if ext is not None:
        name = p.stem + ext
    else:
        name = p.name
    return p.parents[3] / set / data_format/ split / name


def color_gt(gt, show_boundary=False):
    render = np.ones((gt.shape[0], gt.shape[1], 3), dtype=np.ubyte) * 255
    # render[(gt & BOUNDARY).astype(bool)]
    for id, color in LABEL_COLORS.items():
        render[gt == id] = color
    # render[gt == Annotations.BACKGROUND] = [255, 255, 255]
    # render[(gt == Annotations.BODY_TEXT).astype(bool)] = [0, 0, 255]
    # render[(gt == Annotations.COMMENT).astype(bool)] = [0, 255, 0]
    # render[(gt == Annotations.DECORATION).astype(bool)] = [255, 0, 0]
    if show_boundary:
        render[(gt & BOUNDARY).astype(bool)] = [100, 100, 100]
    return render

def load_pixel_label(path):
    gt = Image.open(str(path))
    gt_arr = to_hex(np.array(gt))
    return gt_arr

class HisDBDataset(torch.utils.data.Dataset):

    def unzip(self, zippath, outpath):
        # TODO: unzip
        # assert zippath.exists()
        # assert outpath.exists()
        # zipfile = ZipFile(str(zippath))
        pass

    def __init__(self, path, transform=None, download=True, split=None, gt=False):
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

        self.split = split
        if split is None:
            self.split = Splits.training.name

        self.paths = sorted(glob(str(path / '*' / 'img' / self.split / '*.jpg')))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        assert index < len(self)
        imgpath = self.paths[index]
        gt_file = change_diva_path(self.paths[index], data_format= 'pixel-level-gt', ext='.png')
        img = Image.open(imgpath)
        # arr = np.array(img)

        if self.gt:
            # only load one channel
            gt_arr = numeric_gt(load_pixel_label(gt_file))
        else: 
            gt_arr = None

        return img, gt_arr


class Processed(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, set='*', split=None, data=None, gt=False, load=['img']):
        self.set = set
        self.path = path
        self.data = data
        self.split = split
        self.ext = '.jpg'
        self.load = load

        if not self.split:
            self.split = SPLITS[0]
        if not data:
            self.data = DATA_NAMES[0]
        glob_str = str(path / self.set / self.data / self.split / ('*' + self.ext))
        self.paths = sorted(glob(glob_str))
        logging.debug(glob_str)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Path(self.paths[index])
        p = change_diva_path(img,ext='')
        paths = {
        'slic': change_diva_path(p, ext='.slic.npy'),
        'tiles': change_diva_path(p, ext='.tiles.npy'),
        'meta': change_diva_path(p, ext='.meta.npy'),
        'y': change_diva_path(p, ext='.patchgt.npy')
        }
        items = []
        for l in self.load:
            if l == 'img':
                items.append(Image.open(str(img)))
            else:
                items.append(np.load(paths[l]))

        return items

