import random

import torch

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
x=y_hat.gather(0, y.view(-1, 1))
print(x)