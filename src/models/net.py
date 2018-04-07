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


def cross_entropy2d(input, target, weight=None, size_average=True):
    '''
    Source: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss.py
    :param input:
    :param target:
    :param weight:
    :param size_average:
    :return:
    '''
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class PageNet(nn.Module):
    def __init__(self, n_classes=4, in_channels=3):
        super(PageNet, self).__init__()
        self.stage1 = nn.Sequential(
            self.block(in_channels, 64),
            self.block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.norm1 = nn.Conv2d(64,128,kernel_size=3, padding=1)

        self.stage2 = nn.Sequential(
            self.block( 64, 128),
            self.block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.norm2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.final = nn.Sequential(
            self.block(128,n_classes)
        )

    def block(self, in_channels, v):
        return torch.nn.Sequential(
            nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
            nn.BatchNorm2d(v),
            nn.ReLU(inplace=True),
            nn.Conv2d(v, v, kernel_size=3, padding=1),
            nn.BatchNorm2d(v),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # n
        x_2 = self.stage1(x)
        x_4 = self.stage2(x_2)
        score_2 = self.norm1(x_2)
        score_4 = self.norm2(x_4)
        # n / 4
        up_2 = F.upsample(x_4, x_2.size()[2:],mode='bilinear')
        up_2 += score_2
        up   = F.upsample(up_2, x.size()[2:],mode='bilinear')
        up = self.final(up)
        return up