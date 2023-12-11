
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import numpy as np

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))


#以下函数可以将数值标签转成相应的文本标签。
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


batch_size = 256
if sys.platform.startswith('win'): #判断系统类型 我的是windows 貌似就只能num_workers=0?
    num_workers = 0  # 0表示不用额外的进程来加速读取数据

else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_inputs = 28*28
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


'''下面我们就可以定义前面小节里介绍的softmax运算了。
在下面的函数中，矩阵X的行数是样本数，列数是输出个数。为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，
再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。
因此，该矩阵每行都是合法的概率分布。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。
'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


'''有了softmax运算，我们可以定义上节描述的softmax回归模型了。这里通过view函数将每张原始图像改成长度为num_inputs的向量。
'''
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b) #torch.mm 矩阵乘法

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1))) #gather 根据输入的inedx来筛 有个问题，不应该是y*log(y_predict)吗？

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
def sgd(params, lr, batch_size): #params=[w,b]参数列表,lr学习率？
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params,lr):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum() #交叉熵l=sum(yi1*logy^)
            # 梯度清零
            l.backward()
            sgd(params, lr, batch_size)

            for param in params:
                    param.grad.data.zero_()   #改了一下顺序，先计算backward 然后sgd更新参数 最后清零
            train_l_sum += l.item() #获取值

            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b],lr)