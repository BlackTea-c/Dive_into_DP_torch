import random

import torch

indices=list(range(10))
random.shuffle(indices)
print(indices)
j=torch.LongTensor(indices[0:5])
print(j)