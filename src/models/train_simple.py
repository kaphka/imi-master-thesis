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
import logging
logging.basicConfig(level=logging.INFO)

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
dataset_path = env.dataset(dataset_name)
batch_size = 128
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
mnist_set = torchvision.datasets.MNIST(str(dataset_path), train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
mnist_test_set = torchvision.datasets.MNIST(str(dataset_path), train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=512,
                                           shuffle=True, num_workers=2)

model = Model(n_classes=10, in_channels=1, layers=4)
exp = TrainLog(env, dataset_name,model,log_time=True)

logging.info(' saving  to %s', exp.save_directory)
logging.info(' logging to %s', exp.log_directory)
# Load loaders
# train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
#                                                     download=DOWNLOAD_CIFAR)

# Build trainer
iterations = 5000
trainer = Trainer(model) \
    .build_criterion('CrossEntropyLoss') \
    .build_metric('CategoricalError') \
    .build_optimizer('Adam', lr=0.001) \
    .save_every((2000, 'iteration'), to_directory=str(exp.save_directory), checkpoint_filename='latest',
                best_checkpoint_filename='best') \
    .validate_every((1, 'epochs'))\
    .set_max_num_iterations(iterations) \
    .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every='never'),
                  log_directory=str(exp.log_directory))
# Bind loaders
trainer.bind_loader('train', train_loader)
trainer.bind_loader('validate', test_loader)
trainer.cuda()

# Go!
logging.info('start training')
trainer.fit()
trainer.set_max_num_iterations(trainer.iteration_count + iterations)
trainer.build_optimizer('Adam',lr=0.0001)
logging.info('slower lr')
trainer.fit()