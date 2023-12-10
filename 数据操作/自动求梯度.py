import torch

x=torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)

'''每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function,
 就是说该Tensor是不是通过某些运算得到的，若是，
 则grad_fn返回一个与这些运算相关的对象，否则是None。'''


y=x+2
print(y)


z=y*y*3
out=z.mean()
print(out)
out.backward() #计算出导数out/z  out/y out/x 所有涉及到的

print(x.grad) #只能求出最底层的x？


'''# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)'''