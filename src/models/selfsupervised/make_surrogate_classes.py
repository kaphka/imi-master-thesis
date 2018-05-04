from experiment.data import Environment
import datasets.divahisdb as diva
import datasets.tiles as tile
import logging
import random
from PIL import Image
import numpy as np
import math
import models.selfsupervised.discriminative as disc
from data.metrics import mean_gradient

SOURCE_NAME = 'codices_all'
size = 32
original = 'original'
n_surrogate_classes = 1000 #
n_samples = 100



def task(env):
    dataset = diva.HisDBDataset(env.dataset(SOURCE_NAME))
    rnd = random.Random(diva.NAME)
    save_folder = env.dataset(disc.NAME)
    save_folder.mkdir(exist_ok=True, parents=True)
    (save_folder / original).mkdir(exist_ok=True, parents=True)
    images = [dataset[idx][0] for idx in range(4)]
    crop = tile.RandCrop(int(size * math.sqrt(2)), random.Random('tiles'))

    n_class = 0
    # for idx in range(100):
    while n_class < n_surrogate_classes:
        img = images[rnd.randint(0, len(images) - 1)]
        cropped = crop(img)
        arr = np.array(cropped)
        bw = np.array(cropped.convert('L'))
        mean_grad = mean_gradient(bw)
        logging.debug('accept patch %f', mean_grad)
        if mean_grad  / 50 > rnd.random():
            cropped.save(save_folder / original /'{}.jpg'.format(n_class))
            n_class += 1


    for patch in (save_folder / original).glob('*.jpg'):
        print(patch)


def save_transforms(path, img):
    pass

if __name__ == '__main__':
    env = Environment()
    logging.basicConfig(level=logging.DEBUG)
    task(env)