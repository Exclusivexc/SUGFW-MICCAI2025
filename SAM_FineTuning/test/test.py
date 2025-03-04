import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
layer = nn.Linear(768, 256)
p = torch.zeros((256, 768), requires_grad=True)
layer.weight = Parameter(p)
b = torch.rand((4096, 768))
c = layer(b)
print(c.shape)