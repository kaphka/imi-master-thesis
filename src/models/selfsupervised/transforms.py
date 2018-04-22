from PIL import Image
from pathlib import Path
import datasets.divahisdb as diva
import random
import numpy as np


class RandTransform():
    def __init__(self, rnd):
        self.rnd = rnd

    def __call__(self, img):
        return img


class RandInvert(RandTransform):

    def __call__(self, img):
        pass
