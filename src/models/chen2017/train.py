
import json
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from models.chen2017.chennet import ChenNet as Model
from experiment.data import Environment

env = Environment()
dataset_name  = 'MNIST'
dataset_path = env.dataset(dataset_name)


# TODO: train different max pool sizes

# TODO: train different kernel numbers

# TODO: train different conv layers

# TODO: train with different number of training images

# TODO: report run time