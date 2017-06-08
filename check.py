import torch
import torch.nn as nn
from torch.autograd import Variable

m = nn.Conv2d(16, 2, (3, 3)).float().cuda()

# input is of size nBatch x nClasses x height x width
input = Variable(torch.randn(3, 16, 10, 10)).cuda()
weight_map = Variable(torch.zeros(3 * 8 * 8)).cuda()
# each element in target has to have 0 <= value < nclasses
target = Variable(torch.ones(3 * 8 * 8).long()).cuda()
weights = torch.FloatTensor([1, 1]).cuda()

F = nn.WeightedNLLLoss()
output = m(input)

loss = F(output.view(output.numel() // 2, 2), target, weight_map)
loss.backward()

print(loss)


######### for WeightedNLLLoss2d

# # input is of size nBatch x nClasses x height x width
# input = Variable(torch.randn(3, 16, 10, 10))
# weight_map = Variable(torch.ones(3, 1, 8, 8))
# # each element in target has to have 0 <= value < nclasses
# target = Variable(torch.ones(3, 8, 8).long())
# weights = torch.FloatTensor([1, 1])

# F = nn.WeightedNLLLoss2d()
# output = m(input)

# loss = F(output, target, weight_map)
# loss.backward()

# print(loss)
# print(nn.NLLLoss2d().forward(output, target))