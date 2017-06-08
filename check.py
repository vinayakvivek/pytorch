import torch
import torch.nn as nn
from torch.autograd import Variable

m = nn.Conv2d(16, 32, (3, 3)).float()
loss = nn.WeightedNLLLoss2d()
# input is of size nBatch x nClasses x height x width
input = Variable(torch.randn(3, 16, 10, 10))
weight_map = Variable(torch.ones(3, 1, 8, 8))
# each element in target has to have 0 <= value < nclasses
target = Variable(torch.ones(3, 8, 8).long())
output = loss(m(input), target, weight_map)
output.backward()

print(output)