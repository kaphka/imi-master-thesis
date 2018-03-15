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

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2, kernels=4, layers=1, max_pool_size=0):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential()
        in_channels = 1
        out_channels = kernels
        for n in range(layers):
            self.conv.add_module('conv' + str(n), nn.Conv2d(in_channels, out_channels, 3))
            in_channels = out_channels
            out_channels += 2
        self.maxpool = nn.MaxPool2d(max_pool_size)
        self.dropout = nn.Dropout()
        self.out_dim = (out_channels - 2) * ((28 - layers * 2) ** 2)

        self.fc1 = nn.Linear(self.out_dim, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1, self.out_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x