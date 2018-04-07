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


class XuNet(nn.Module):
    def __init__(self, n_classes=4, in_channels=3):
        super(XuNet, self).__init__()

    def forward(self, x):
        return x





class PageNet(nn.Module):
    def __init__(self, n_classes=4, in_channels=3):
        super(XuNet, self).__init__()

    def forward(self, x):
        return x