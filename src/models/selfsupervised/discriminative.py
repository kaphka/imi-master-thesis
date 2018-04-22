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
import  torch.utils.data as td
from torch.utils.data.sampler import Sampler

import torchvision.transforms as transforms
import models.net as base
import random
import itertools as it

import numpy as np

import experiment.info as info

NAME = 'codices_surrogate'

# class TripletBatchIterator():
#
#     def __init__(self, label_pos, K=5):
#         self.label_pos = label_pos
#         self.K = K
def stack_triplets(batch_data):
    # batch =
    batch = torch.cat([x for x, y in batch_data])
    target = torch.cat([y for x, y in batch_data])
    return batch, target

class TripelDataset(td.Dataset):

    def __init__(self, dataset,labels=None, K=1):
        self.dataset = dataset
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        elif labels is not None:
            self.labels = labels
        else:
            raise AssertionError()

        self.K = K
        label_names = np.unique(self.labels)
        self.label_pos = [np.where(self.labels.numpy() == name)[0] for name in label_names]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_1, y_1 = self.dataset[index]
        x_2_pos = random.choice(self.label_pos[y_1])
        x_2, y_2 = self.dataset[x_2_pos]

        items = [x_1, x_2]
        for idx in range(self.K):
            other_labels = list(range_without(0, len(self.label_pos), y_1))
            x_neg_pos = random.choice(self.label_pos[random.choice(other_labels)])
            x_3, y_3 = self.dataset[x_neg_pos]
            items.append(x_3)
        return torch.stack(items), torch.zeros((len(items)))

def range_without(a, b, n):
    return it.chain(range(a, n), range(n+1, b))




class TripletLoss(torch.nn.Module):

    def __init__(self, M=0.5):
        super(TripletLoss, self).__init__()
        self.M = M

    def forward(self, x, target=None):
        return triplet_loss(x,self.M)

def triplet_loss(x, M=0.5):
    '''
    See Wang et al. 2015, Doersch et al. 2017
    '''
    x_1 = x[0::3]
    x_2 = x[1::3]
    x_3 = x[2::3]
    x_pos =  F.cosine_similarity(x_1, x_2, dim=1)
    x_neg =  F.cosine_similarity(x_1, x_3, dim=1)
    # raise AssertionError
    return F.relu(x_pos - x_neg + M).mean()

class BaseNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=4, embeding_size=2, shortcut=False):
        super(BaseNet, self).__init__()
        self.n_classes = n_classes
        self.classify = False
        self.embeding_size = embeding_size
        self.use_shortcut = shortcut
        self.info = info.LogInfo('VGGBase')
        # 64
        self.stage1 = nn.Sequential(
            self.block(in_channels, 64),
            self.block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 128
        self.stage2 = nn.Sequential(
            self.block( 64, 128),
            self.block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 256
        self.stage3 = nn.Sequential(
            self.block(128, 256),
            self.block(256, 256),
            self.block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 512
        self.stage4 = nn.Sequential(
            self.block(256, 512),
            self.block(512, 512),
            self.block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 512
        self.stage5 = nn.Sequential(
            self.block(512, 512),
            self.block(512, 512),
            self.block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.embeding = nn.Sequential(
            self.block(512, self.embeding_size)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=1),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.n_classes, kernel_size=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(256,256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,self.embeding_size, kernel_size=1)
        )

        self.norm1 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.norm2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.norm4 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.norm5 = nn.Conv2d(512, 128, kernel_size=3, padding=1)


    def block(self, in_channels, target_channels, kernel_size=3, padding=1):
        return torch.nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # n
        x_2  = self.stage1(x)
        x_4  = self.stage2(x_2)
        x_8  = self.stage3(x_4)
        if self.use_shortcut:
            return self.shortcut(x_8)[:,:,0,0]
        x_16 = self.stage4(x_8)
        x_32 = self.stage5(x_16)
        x = x_32

        x = self.embeding(x)

        if self.classify:
            x = self.classifier(x_32)
            x = x[:,:,0,0]
        #
        # score_2 = self.norm1(x_2)
        # score_4 = self.norm2(x_4)
        # # n / 4
        # up_2 = F.upsample(x_4, x_2.size()[2:],mode='bilinear')
        # up_2 += score_2
        # up   = F.upsample(up_2, x.size()[2:],mode='bilinear')
        # up = self.final(up)
        return x