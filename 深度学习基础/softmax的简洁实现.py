

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

num_inputs = 784
num_outputs = 10

class LinerNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinerNet, self).__init__()
        self.linear=nn.Linear(num_inputs,num_outputs)

    def forward(self,x): #注意此处x.shape=(1,28,28) 需要展平成（1，28*28）
        y=self.linear(x.view(x.shape[0],-1))
        return y


net=LinerNet(num_inputs,num_outputs)

#初始化参数
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

#定义损失函数，选择优化器
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    for X, y in train_iter:
        output = net.forward(X)
        l = loss(output, y).sum()
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))