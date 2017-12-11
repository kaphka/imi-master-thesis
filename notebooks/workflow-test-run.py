import logging
from pathlib import Path
from archiveorg import AOrg
import os
from tqdm import tqdm
import glymur
from PIL import Image

from zipfile import ZipFile

logging.basicConfig(level=logging.DEBUG)
db_folder = Path("/media/jakob/bigdata/datasets/archive_org/")
os.chdir(str(db_folder))

logging.info("Unzip jp2 books")
identifiers = [d.name for d in db_folder.iterdir() if d.is_dir()]
for item in tqdm(identifiers):
    itemPath = Path(AOrg.file(item, kind='jp2zip'))
    if not itemPath.is_file():
        continue

    zipfile = ZipFile(str(itemPath))
    images = [name for name in zipfile.namelist() if name.endswith("jp2")]
    # print(images)
    for image in images:
        # print(AOrg.path(item))
        unziptarget = AOrg.path(item) / image
        if unziptarget.is_file():
            unziped = zipfile.extract(image, path=str(AOrg.path(item)))

logging.info("Unzip jp2 books")
def conversion(identifiers, kindfrom, kindto):
    for item in identifiers:
        for jp2file in AOrg.files(item, kind=kindfrom):
            # print(jp2file)
            jpgfile = AOrg.path(item, kind=kindto) / jp2file.with_suffix('.' + kindto).name
            yield jp2file, jpgfile

conversions = list(conversion(identifiers,'jp2', 'jpg'))


def convert(source, target, subsamples=[1]):
    page = glymur.Jp2k(source)
    for sample in subsamples:
        step = 2 ** sample
        out = target.parent / str(sample) / target.name
        out.parent.mkdir(exist_ok=True)
        img = Image.fromarray(page[::step, ::step])
        # skimage.io.imsave(out, page[::step, ::step], plugin='pil', quality=100)
        # print(out)
        img.save(out, quality=100, progressive=True)

logging.info("convert jp2 to jpeg")
for c in tqdm(conversions):
    jp2file, jpgfile = c
    # print(c)
    Path(jpgfile).parent.mkdir(exist_ok=True)
    convert(jp2file, jpgfile, subsamples=[3])
