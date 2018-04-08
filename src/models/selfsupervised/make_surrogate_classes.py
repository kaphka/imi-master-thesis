from experiment.data import Environment
import datasets.divahisdb as diva
import datasets.tiles as tile
import logging
import random
from PIL import Image
import numpy as np
import math
import models.selfsupervised.discriminative as disc

SOURCE_NAME = 'codices_all'
size = 32
original = 'original'

def mean_gradient(bw):
    dx = bw[1:, :] - bw[:-1, :]
    dy = bw[:, 1:] - bw[:, :-1]
    grad = np.add(np.power(dx[:, :-1], 2), np.power(dy[:-1, :], 2))
    return np.mean(grad)

def task(env):
    dataset = diva.HisDBDataset(env.dataset(SOURCE_NAME))
    rnd = random.Random(diva.NAME)
    save_folder = env.dataset(disc.NAME)
    save_folder.mkdir(exist_ok=True, parents=True)
    (save_folder / original).mkdir(exist_ok=True, parents=True)
    images = [dataset[idx][0] for idx in range(4)]
    crop = tile.RandCrop(int(size * math.sqrt(2)), random.Random('tiles'))

    n_class = 0
    for idx in range(100):
        img = images[rnd.randint(0, len(images) - 1)]
        cropped = crop(img)
        arr = np.array(cropped)
        bw = np.array(cropped.convert('L'))
        mean_grad = mean_gradient(bw)
        logging.debug('accept patch %f', mean_grad)
        if mean_grad  / 50 > rnd.random():
            cropped.save(save_folder / original /'{}.jpg'.format(n_class))
            class_folder = save_folder / class_folder
            save_transforms(class_folder, cropped)
            n_class += 1

def save_transforms(path, img):
    pass

if __name__ == '__main__':
    env = Environment()
    logging.basicConfig(level=logging.DEBUG)
    task(env)