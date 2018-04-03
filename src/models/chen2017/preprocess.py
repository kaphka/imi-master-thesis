import datasets.divahisdb as diva
import experiment.data as exd
from models.chen2017.transforms import *
from pathlib import Path
from PIL import Image
from os.path import splitext

from tqdm import tqdm
import logging

env = exd.Environment()
processed_path = env.dataset('DIVA_Chen2017_processed')
hisdb_path = env.dataset('DIVA-HisDB')
dataset = diva.HisDBDataset(hisdb_path)

import skimage.io
import skimage.segmentation as seg
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize, rotate

# logging.basicConfig(level=logging.DEBUG)
n_patches = 0
scaler = Scale()
slic = SLIC()
get_tiles = SegmentTiling()
get_gt = TileGroundTruth()

def change_ext(name, ext):
    return splitext(name)[0] + '.' + ext

for p in tqdm(dataset.paths[:2]):
    path = Path(p)
    gt_file = diva.change_diva_path(path, data_format='pixel-level-gt', ext='.png')
    print(gt_file)
    processed_file = processed_path / path.relative_to(hisdb_path)
    processed_file.parent.mkdir(exist_ok=True, parents=True)

    seg_file   = diva.change_diva_path(processed_file, ext='.slic')
    tiles_file = diva.change_diva_path(processed_file, ext='.tiles')
    meta_file  = diva.change_diva_path(processed_file, ext='.meta')
    patchgt_file =  diva.change_diva_path(processed_file, ext='.patchgt')

    img = Image.open(path)
    # scale
    logging.debug('scaled %s', processed_file)
    scaled = scaler(img)

    # segment
    spixel = slic(scaled)
    logging.debug('slic  %s', seg_file)

    # patches
    tiles, tile_specs = get_tiles(img_as_ubyte(scaled), spixel, max_patches=n_patches)
    logging.debug('tiles %s', tiles_file)
    logging.debug('meta  %s', meta_file)


    # gt
    gt = diva.load_pixel_label(gt_file)
    gt_labels = diva.numeric_gt(gt, label_boundary=True)
    scaled_gt = np.array(Image.fromarray(gt_labels).resize((scaled.width, scaled.height)))
    # y = gt[np.round(tile_specs[:, 0] / scaler.scale).astype(int), np.round(tile_specs[:, 1] / scaler.scale).astype(int)]
    y = scaled_gt[tile_specs[:, 0], tile_specs[:, 1]]

    scaled.save(processed_file)
    np.save(str(seg_file), spixel)
    np.save(str(tiles_file), tiles)
    np.save(str(meta_file), tile_specs)
    np.save(str(patchgt_file), y)

