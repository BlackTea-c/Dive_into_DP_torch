import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys


import torchvision
import torchvision.transforms as transforms

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
num_workers=0
batch_size=256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
        )

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5

for epoch in range(num_epochs+1):
    for X,y in train_iter:
        y_hat=net(X.view(X.shape[0],-1))
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))