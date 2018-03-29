import torch.nn as nn
from inferno.io.box.cifar import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten

import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import numpy as np
import json, os
from glob import glob
from PIL import Image
from tqdm import tnrange, tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.chennet import ChenNet as Model
from torchvision.datasets import ImageFolder

from models.chennet import ChenNet as Model

# Build torch model
# model = nn.Sequential(
#     ConvELU2D(in_channels=1, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     Flatten(),
#     nn.Linear(in_features=(256 * 4 * 4), out_features=10,bias=False),
#     nn.Softmax()
# )

config = json.load(open(os.path.expanduser("~/.thesis.conf")))
datasets_path = Path(config['datasets'])
dataset_name = 'MNIST'
db_folder = Path(config['datasets']) / dataset_name
models_folder = Path(config['models'])
log_folder = models_folder / 'logs'
modules = Path(config['project']) / 'src'
ext = 'test'

# os.makedirs(save_directory, exist_ok=True)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
mnist_set = torchvision.datasets.MNIST(str(datasets_path / dataset_name), train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_set, batch_size=64,
                                           shuffle=True, num_workers=2)

model = Model(n_classes=10, in_channels=1, layers=4)
save_directory = models_folder / model.log_name / (dataset_name + '_{}'.format(ext))
log_name = log_folder / model.log_name
# Load loaders
# train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
#                                                     download=DOWNLOAD_CIFAR)

# Build trainer
trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
.save_to_directory(str(save_directory)) \
    .set_log_directory(str(save_directory)) \
    .save_every((2, 'epochs')) \
    .set_max_num_epochs(4) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every='never'),
                  log_directory=str(log_name))
# Bind loaders
trainer \
    .bind_loader('train', train_loader)

trainer.cuda()

# Go!
trainer.fit()