import torch
from discriminator import SGANDiscrimantor

x = torch.ones(1, 3, 1024, 1024)
dis = SGANDiscrimantor(5)
y = dis(x)
print(x.shape, '->', y.shape)