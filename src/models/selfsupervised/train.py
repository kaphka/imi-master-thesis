import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from models.selfsupervised.discriminative import triplet_loss


criterion = triplet_loss
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        x, x_plus, x_minus = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')