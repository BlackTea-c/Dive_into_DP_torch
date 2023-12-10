import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


#生成一下数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32) #数据点
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b #对应y值
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32) #加上干扰项nata

import torch.utils.data as Data #torch中专门处理数据的模块

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


'''首先，导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写。
顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。
nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。
下面先来看看如何用nn.Module实现一个线性回归模型。'''
import torch.nn as nn


class LinearNet(nn.Module):#继承nn.Module
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        '''Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，
        然后“被转换”的类NNet对象调用自己的init函数，
        其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。'''
        self.linear = nn.Linear(n_feature, 1) #input:n_feature  output=1
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x) #y=w*n_feature + bias
        return y

net = LinearNet(num_inputs)
#print(net) # 使用print可以打印出网络的结构

'''事实上我们还可以用nn.Sequential来更加方便地搭建网络，
Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。'''

'''
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))
print(net)
print(net[0])
'''

'''
可以通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。

for param in net.parameters():
    print(param)
'''

from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

loss=nn.MSELoss()
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net.forward(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))