from experiment.data import Environment
import datasets.divahisdb as diva
import datasets.tiles as tile
import logging
import random
from PIL import Image
import numpy as np

def task(env):
    split = diva.Splits.training.name
    dataset = diva.HisDBDataset(env.dataset(diva.NAME), gt=True, split=split)
    width = 320
    tiles_path = env.dataset(tile.TILE_SET_NAME) / split
    tiles_path.mkdir(parents=True, exist_ok=True)
    print(tiles_path)

    crop = tile.RandCrop(320, random.Random('tiles'))
    crop_gt = tile.RandCrop(320, random.Random('tiles'))
    store = tile.SaveTo(tiles_path)

    mean = np.zeros((3))
    img, gt = dataset[5]
    img_arr = np.array(img)
    mean = mean + img_arr.mean(axis=(0,1))
    store(crop(img), 'tile0.jpg')
    store(crop_gt(Image.fromarray(gt)), 'gt0.png')

    gt = Image.open(tiles_path/'gt0.png')

def test(env):
    tile_set = tile.ImgTiles(env.dataset(tile.TILE_SET_NAME))
    print(tile_set[0])


if __name__ == '__main__':
    env = Environment()
    logging.basicConfig(level=logging.DEBUG)
    task(env)
    test(env)