import torch

if torch.cuda.is_available():
    import torch.cuda as t
else:
    import torch as t

from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class ChenNet(nn.Module):
    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def block(self):
        return nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(inplace=True)
        )

    def __init__(self, n_classes=2, kernels=4, layers=1, max_pool_size=0, in_channels=1, tensor_width=28):
        super(ChenNet, self).__init__()
        self._log_name = 'ChenNet'
        self.n_conv_layers = layers
        self.n_kernels = kernels
        self.max_pool_size = max_pool_size
        self.conv = nn.Sequential()
        out_channels = kernels

        self.dropout = nn.Dropout()
        for n in range(layers):
            # self.conv.add_module('dropout', self.dropout)
            self.conv.add_module('conv' + str(n), nn.Conv2d(in_channels, out_channels, 3))
            self.conv.add_module('ReLU', nn.ReLU(inplace=True))
            in_channels = out_channels
            out_channels += 2

        # self.maxpool = nn.MaxPool2d(max_pool_size)
        self.out_dim = list(self.conv.children())[-1].out_channels * ((tensor_width - layers * 2) ** 2)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.out_dim, 100),
            self.dropout,
            nn.Linear(100, n_classes),
            nn.LogSoftmax(dim=1)
            )

        self.conv.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x

    @property
    def log_name(self):
        return self._log_name + self.conf_str

    @property
    def name(self):
        return self._log_name

    @property
    def conf_str(self):
        return '{}_{}_{}'.format(
            self.n_conv_layers,
            self.n_kernels,
            self.max_pool_size)
