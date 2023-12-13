import torch
import torch.nn as nn


def corr2d(X,K):#X为输入矩阵，K为卷积核； X.shape=m*n  K.shape=h*w
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j]=(X[i:i+h,j:j+w]*K).sum()
    return Y
    #输出数组shape=m-h+1,n-w+1

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
#print(corr2d(X, K))



'''二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。
卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。
下面基于corr2d函数来实现一个自定义的二维卷积层。在构造函数__init__里我们声明weight和bias这两个模型参数。
前向计算函数forward则是直接调用corr2d函数再加上偏差。'''


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


'''下面我们来看一个卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置。
首先我们构造一张6×8
6×8的图像（即高和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。'''

X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1, -1]])
print('==============')
print(corr2d(X,K))
#由此，我们可以看出，卷积层可通过重复使用卷积核有效地表征局部空间。
Y=corr2d(X,K)
#现在我们来学习一下:
# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print(conv2d.weight,K) #学习到的与K接近


'''实际上，卷积运算与互相关运算类似。为了得到卷积运算的输出，我们只需将核数组左右翻转并上下翻转，再与输入数组做互相关运算。可见，
卷积运算和互相关运算虽然类似，但如果它们使用相同的核数组，对于同一个输入，输出往往并不相同。
那么，你也许会好奇卷积层为何能使用互相关运算替代卷积运算。其实，在深度学习中核数组都是学出来的：
卷积层无论使用互相关运算或卷积运算都不影响模型预测时的输出。为了解释这一点，假设卷积层使用互相关运算学出图5.1中的核数组。
设其他条件不变，使用卷积运算学出的核数组即图5.1中的核数组按上下、左右翻转。也就是说，图5.1中的输入与学出的已翻转的核数组再做卷积运算时，
依然得到图5.1中的输出。为了与大多数深度学习文献一致，如无特别说明，本书中提到的卷积运算均指互相关运算。'''