
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



'''在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签。'''

# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #[0,1,2....num_examples-1]
    random.shuffle(indices)  # 样本的读取顺序是随机的 ,打乱indices的顺序
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j) #0表示以行为标准


'''batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break '''



#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)


#之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True。
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):  # torch.mm实现矩阵乘法 y=w*X+b
    return torch.mm(X, w) + b

def squared_loss(y_hat,y): #y_hat=linreg(X,w,b)
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2             #1/2{(y_true-yi)**2}\

'''以下的sgd函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。
这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。'''
def sgd(params, lr, batch_size): #params=[w,b]参数列表,lr学习率？
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

lr = 0.03
num_epochs = 3
net = linreg # 输出格式
loss = squared_loss # 损失函数
batch_size=10
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w,w)
print(true_b,b)