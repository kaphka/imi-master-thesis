import skimage.io
import skimage.segmentation as seg
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize, rotate
from skimage import io
import matplotlib.pyplot as plt
import argparse

import random as rnd
import math
from collections import namedtuple
import numpy as np


class Scale(object):
    def __init__(self, scale=2 ** -2):
        self.scale = scale

    def __call__(self, image):
        return image.resize((int(image.width * self.scale), int(image.height * self.scale)))


class SLIC(object):
    def __init__(self, k=3000):
        self.k = k

    def __call__(self, image):
        floats = img_as_float(image)
        segments = seg.slic(floats, n_segments=self.k)
        return segments


class TileGroundTruth(object):
    def __init__(self, dtype=np.int):
        self.dtype = dtype

    def __call__(self, tile_meta, gt, scale):
        y = gt[np.round(tile_meta[:, 0] / scale).astype(self.dtype),
               np.round(tile_meta[:, 1] / scale).astype(self.dtype)]
        return y


class SegmentTiling(object):
    def __init__(self, patch_width=28):
        self.patch_width = patch_width

    def __call__(self, image, segments, max_patches=0):
        SLICPixel = namedtuple('SLICPixel', ['x', 'y', 'width', 'height'])
        Point = namedtuple('Point', ['x', 'y'])

        nsegments = int(np.max(segments)) + 1
        # print(nsegments)
        if max_patches > 0:
            nsegments = max_patches
        if len(image.shape) > 2:
            patch_shape = (nsegments, self.patch_width, self.patch_width, 3)
        else:
            patch_shape = (nsegments, self.patch_width, self.patch_width)

        patches = np.zeros(patch_shape, dtype=image.dtype)
        patch_meta = np.zeros((nsegments, 3), dtype=np.int32)
        patch_count = 0

        for snum in range(0, nsegments):
            spixel = np.where(segments == snum)
            mmin, nmin = list(map(min, spixel))
            mmax, nmax = list(map(max, spixel))

            spixel = SLICPixel(mmin, nmin, mmax - mmin, nmax - nmin)
            center = Point(spixel.x + spixel.width / 2 + 0.5,
                           spixel.y + spixel.height / 2 + 0.5)
            patch = SLICPixel(center.x - self.patch_width / 2,
                              center.y - self.patch_width / 2,
                              self.patch_width, self.patch_width)
            p = self.centered_patch(patch.x, patch.y, patch.width)
            img_patch = image[p]

            if img_patch.shape[0] == self.patch_width and img_patch.shape[1] == self.patch_width:
                patches[patch_count, :, :] = img_patch
                patch_meta[patch_count, :] = [center.x, center.y, snum]
                patch_count += 1

        # prune
        patches = patches[:patch_count]
        patch_meta = patch_meta[:patch_count]
        return patches, patch_meta

    def centered_patch(self, m, n, width, offset=[0, 0]):
        return (slice(int(m + offset[0]),
                      int(m + offset[0] + width)),
                slice(int(n + offset[1]),
                      int(n + offset[1] + width)))