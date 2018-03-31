import json
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from models.chen2017.chennet import ChenNet as Model

from experiment.data import Environment, TrainLog

# config = json.load(open(os.path.expanduser("~/.thesis.conf")))
# datasets_path = Path(config['datasets'])
# db_folder = Path(config['datasets']) / dataset_name
# models_folder = Path(config['models'])
# log_folder = models_folder / 'logs'
# modules = Path(config['project']) / 'src'
# ext = 'test'

# os.makedirs(save_directory, exist_ok=True)

dataset_name = 'MNIST'
env = Environment()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
mnist_set = torchvision.datasets.MNIST(str(env.dataset(dataset_name)), train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_set, batch_size=64,
                                           shuffle=True, num_workers=2)

model = Model(n_classes=10, in_channels=1, layers=4)
exp = TrainLog(env, dataset_name,model)

# Load loaders
# train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
#                                                     download=DOWNLOAD_CIFAR)

# Build trainer
trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam') \
    .save_to_directory(str(exp.save_directory)) \
    .set_log_directory(str(exp.save_directory)) \
    .save_every((2, 'epochs')) \
    .set_max_num_epochs(4) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every='never'),
                  log_directory=str(exp.log_directory))
# Bind loaders
trainer.bind_loader('train', train_loader)
trainer.cuda()

# Go!
trainer.fit()