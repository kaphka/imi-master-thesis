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

import numpy as np

class HisDBDataset(torch.utils.data.Dataset):
    
    gtrgbmap = {
        0x000008: 'main text body',
        0x000004: 'decoration',
        0x000002: 'comment',
        0x000001: 'background',
        0x800000: 'boundary pixel'
    }
    setnames = ['CB55', 'CS18', 'CS863']
    def unzip(self, zippath, outpath):
        assert zippath.exists()
        assert outpath.exists()
        zipfile = ZipFile(str(zippath))
        # TODO: unzip
    
    def gtpath(self, p):
        if isinstance(p, str):
            p = Path(p)
        return p.parents[2] / 'pixel-level-gt' / 'training' / (p.stem + '.png')
    
    def __init__(self, path, transform=None, download=True, train=True, gt=False):
        self.gt = gt
        for setname in self.setnames:
            folder = path / setname
            folder.mkdir(exist_ok=True)
            images = path / 'img-{}.zip'.format(setname)
            imagesgt = path / 'pixel-level-gt-{}.zip'.format(setname)
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
        gtpath = str(self.gtpath(self.paths[index]))
        img =  Image.open(imgpath)
        if self.gt:
            # only load one channel
            gt = Image.open(gtpath)
            return img, gt 
        else: 
            return img