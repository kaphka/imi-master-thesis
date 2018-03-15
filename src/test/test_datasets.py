from datasets.washington import Washington
from datasets.divahisdb import HisDBDataset

import json
import os
from pathlib import Path

config = json.load(open(os.path.expanduser("~/.thesis.conf")))
db_folder = Path(config['datasets'])


def getdatasetpath(name):
    folder = db_folder / name
    assert folder.exists()
    return folder

def test_diva():
    path = getdatasetpath('hisdb')
    dataset = HisDBDataset(path)
    run_dataset_test(dataset)

def test_washington():
    path = getdatasetpath('washington')
    dataset = Washington(path)
    run_dataset_test(dataset)


def run_dataset_test(dataset):
    img = dataset[5]
    assert not img is None
