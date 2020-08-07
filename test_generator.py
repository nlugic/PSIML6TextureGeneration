import torch
from generator import SGANGenerator

x = torch.ones(1, 100, 32, 32)
gen = SGANGenerator(100, 5)
y = gen(x)
print(x.shape, '->', y.shape)