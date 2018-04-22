import json
import os
from pathlib import Path

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from models.selfsupervised.discriminative import BaseNet as Model
import models.selfsupervised.discriminative as dis

from experiment.data import Environment, TrainLog
import logging



mnist_to_tensor32 = transforms.Compose(
        [
            lambda img: img.resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def set_log_and_save(trainer, exp):
    trainer.save_every((1000, 'iterations'))
    trainer.save_to_directory(str(exp.save_directory))
    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                           log_images_every='never'),
                         log_directory=str(exp.log_directory))


def task(env):
    dataset_name = 'MNIST'
    dataset_path = env.dataset(dataset_name)
    dataset = MNIST(str(dataset_path), train=True,  download=True, transform=mnist_to_tensor32)
    testset = MNIST(str(dataset_path), train=False, download=True, transform=mnist_to_tensor32)
    test_loader  = DataLoader(testset, batch_size=512, shuffle=True, num_workers=2)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    triplets = dis.TripelDataset(dataset, dataset.train_labels, K=1)
    train_loader_triple = DataLoader(triplets, batch_size=16,collate_fn=dis.stack_triplets)

    model = Model(in_channels=1, n_classes=10, shortcut=True)
    criterion = dis.TripletLoss()
    exp = TrainLog(env, dataset_name,model.info,log_time=True)

    logging.info(' saving  to %s', exp.save_directory)
    logging.info(' logging to %s', exp.log_directory)
    # Load loaders
    # train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
    #                                                     download=DOWNLOAD_CIFAR)

    iterations = 0
    trainer = Trainer(model)
    set_log_and_save(trainer, exp)
    trainer.build_criterion(criterion) \
            .build_optimizer('SGD', lr=0.001, weight_decay=0.0005) \
            .set_max_num_iterations(iterations)\
            .bind_loader('train', train_loader_triple)
    trainer.cuda()
    logging.info('start training')
    trainer.fit()

    print(model.forward(Variable(next(iter(train_loader_triple))[0]).cuda()))

    model.use_shortcut = False
    model.classify = True
    print(model.forward(Variable(next(iter(train_loader_triple))[0]).cuda()))

    trainer = Trainer(model)
    set_log_and_save(trainer, exp)
    trainer.build_criterion('CrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam', lr=0.001) \
        .set_max_num_iterations(5000) \
        .bind_loader('train', train_loader) \
        .bind_loader('test', test_loader)
    trainer.cuda()
    logging.info('start training')
    trainer.fit()

    # trainer.set_max_num_iterations(trainer.iteration_count + iterations)
    # trainer.build_optimizer('Adam',lr=0.0001)
    # logging.info('slower lr')
    # trainer.fit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = Environment()
    task(env)
