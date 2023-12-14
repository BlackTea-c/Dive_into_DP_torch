


'''使用正弦函数和一些可加性噪声来生成序列数据， 时间步为1,2,3...1000'''

import torch
from torch import nn
import random

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.5, (T,)) #正弦+噪声


#接下来转换为 features-labels 对
#print(x.shape)
tau = 4 #Xt-tau...Xt-1
features = torch.zeros((T - tau, tau))
#print(features[0,:])
for i in range(tau):
    features[:, i] = x[i: T - tau + i]

labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #[0,1,2....num_examples-1]
    random.shuffle(indices)  # 样本的读取顺序是随机的 ,打乱indices的顺序
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j) #0表示以行为标准


import torch.utils.data as Data
dataset = Data.TensorDataset(features, labels)
train_data, test_data = Data.random_split(dataset, [600,len(dataset)-600])
train_iter = Data.DataLoader(train_data, batch_size, shuffle=True)


train_iter_error=data_iter(batch_size,features,labels) #不知道为什么用我自己写得函数，epoch2就没东西了？

'''知道原因了，for X,y in data_iter(batch_size,features,labels):才行，你自己对照一下简单的线性回归.py就知道了'''
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')#reduction='none'表示不进行降维或聚合


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()

            total_loss += l.sum().item()
            total_samples += len(y)

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            print(f'epoch {epoch + 1}, '
                  f'loss: {avg_loss}')
        else:
            print(f'No samples found in epoch {epoch + 1}')


net = get_net()
train(net, train_iter_error, loss, 5, 0.01)
