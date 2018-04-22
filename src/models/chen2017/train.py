
import json
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from models.chen2017.chennet import ChenNet as Model
import datasets.tiles as tiles
from experiment.data import Environment, TrainLog
import logging
logging.basicConfig(level=logging.INFO)

env = Environment()
dataset_name  = 'Chen2017_np_tiles_balanced'
dataset_path = env.dataset(dataset_name)

mean =168.61987390394304
std =56.83193208713197

transform = transforms.Compose(
    [
        lambda img: img.convert('L'),
        lambda img: img.resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

train_set = torchvision.datasets.ImageFolder(str(env.dataset('tile_img')),transform=transform)
# train_set = array.Tiles(dataset_path, transforms=transform)
test_set = tiles.Tiles(dataset_path, train=False, transforms=transform)
train_loader = DataLoader(train_set, batch_size=32,
                                           shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=512,
                                           shuffle=True, num_workers=2)

# model = Model(n_classes=4, in_channels=1, layers=2)
import models.selfsupervised.discriminative as dis
model = dis.BaseNet(in_channels=1, n_classes=4)
model.classify = True
exp = TrainLog(env, 'img_folder', model.info ,log_time=True)
logging.info(' saving  to %s', exp.save_directory)
logging.info(' logging to %s', exp.log_directory)

# Build trainer
max_num_iterations = 10000
trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
    .save_every((1, 'epochs')) \
    .save_to_directory(str(exp.save_directory))\
    .validate_every((2, 'epochs'))\
    .set_max_num_epochs(100) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iterations'),
                                    log_images_every='never'),
                  log_directory=str(exp.log_directory))
    # .save_every((2000, 'iterations'), to_directory=str(exp.save_directory), checkpoint_filename='latest') \

# Bind loaders
trainer.bind_loader('train', train_loader)
trainer.bind_loader('validate', test_loader)
trainer.cuda()

# Go!
trainer.fit()

# TODO: train different max pool sizes

# TODO: train different kernel numbers

# TODO: train different conv layers

# TODO: train with different number of training images

# TODO: report run time