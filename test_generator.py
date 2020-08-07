import torch
import torch.nn as nn
from generator import SGANGenerator

x = torch.ones(1, 100, 256, 256)
gen = SGANGenerator(100, 5)
y = gen(x)
print(y)
print(y.shape)