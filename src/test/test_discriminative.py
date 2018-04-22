import models.selfsupervised.discriminative as dis
import numpy as np
import torch as t
import torch.autograd as grad
import test.mock
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from experiment.data import Environment


def test_triplet_loader_mnist():
    env = Environment()
    transform = transforms.Compose(
        [
            lambda img: img.resize((32,32)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    dataset = MNIST(str(env.dataset('MNIST')),train=True, download=False, transform=transform)
    tripelset = dis.TripelDataset(dataset, dataset.train_labels)
    trainloader = DataLoader(tripelset, collate_fn=dis.stack_triplets, batch_size=5)
    x, y = next(iter(trainloader))
    # print(x)
    # x_1, x_2, x_3 = x
    assert x.shape == (5 * 3, 1, 32, 32)
    print(x.shape)
    # assert False

def test_triplet_sample():
    dataset = test.mock.RandDataset()
    print(dataset.y.numpy())
    triplets = dis.TripelDataset(dataset, dataset.y.long())
    for triplet_batch in range(len(triplets)):
        triplet = triplets[triplet_batch]


def test_triplet_loader():
    dataset = test.mock.RandDataset()
    triplets = dis.TripelDataset(dataset, dataset.y)
    trainloader = DataLoader(triplets, collate_fn=dis.stack_triplets, batch_size=5)

    for i, data in enumerate(trainloader, 0):
        x, y = data
        assert x.shape == (5 * 3, dataset.channels, dataset.img_size, dataset.img_size)


def test_baseNet():
    dataset = test.mock.RandDataset(size=10, img_size=32, channels= 3, labels=3)
    triplets = dis.TripelDataset(dataset, dataset.y)
    trainloader = DataLoader(triplets,batch_size=5, collate_fn=dis.stack_triplets)
    criterion = dis.TripletLoss()
    x, y = next(iter(trainloader))
    net = dis.BaseNet(embeding_size=10)
    # x_1, x_2, x_3 = x
    input = grad.Variable(t.Tensor(x))
    out = net(input)
    assert out.shape == (5 * 3, 10, 1, 1)
    loss = criterion(out)
    loss.backward()
    print(loss, loss.grad)
    # assert False


def test_triplet_loss():
    net = dis.BaseNet()
    data = np.random.randn(3, 3, 32, 32)
    input = grad.Variable(t.Tensor(data))
    out = net(input)
    assert out.shape == (3, 2, 1, 1)
    x_1 = out[0].view(2)
    x_2 = out[1].view(2)
    x_3 = out[2].view(2)
    assert x_1.shape == (2,)
    loss = dis.triplet_loss(x_1, x_2, x_3)
    print(loss)

