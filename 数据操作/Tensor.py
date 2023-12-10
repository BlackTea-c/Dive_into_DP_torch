import torch

'''x=torch.empty(5,3)
print(x)

y=torch.rand(5,3)
print(y)

#创建一个5x3的long型全0的Tensor:
x=torch.zeros(5,3,dtype=torch.long)
print(x)'''


x=torch.tensor([1,2,3,4,5,6])
print(x)
y=x
y[0]=5
print(x,y) #共享内存，x,y同时被改动！
z=y.view(-1,3)
print(z)

'''所以如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？Pytorch还提供了一个reshape()可以改变形状，
但是此函数并不能保证返回的是其拷贝，所以不推荐使用。
推荐先用clone创造一个副本然后再使用view
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)
'''


'''另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：
x = torch.randn(1)
print(x)
print(x.item())
Copy to clipboardErrorCopied
输出：
tensor([2.3466])
2.3466382026672363'''
