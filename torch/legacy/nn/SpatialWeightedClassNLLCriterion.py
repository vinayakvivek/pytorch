import torch
from .utils import recursiveType
import torch._thnn


class SpatialWeightedClassNLLCriterion(object):

        def __init__(self, weights=None, sizeAverage=True):
                assert weights is None or weights.dim() == 1
                super(SpatialWeightedClassNLLCriterion, self).__init__()
                self.gradInput = torch.Tensor()
                self.output = 0
                self._backend = torch._thnn.type2backend[type(self.gradInput)]

                self.sizeAverage = sizeAverage
                self.weights = weights

                self.output_tensor = torch.zeros(1)
                self.total_weight_tensor = torch.ones(1)

        def updateOutput(self, input, target, weight_map):
                self._backend.SpatialWeightedClassNLLCriterion_updateOutput(
                                self._backend.library_state,
                                input,
                                target,
                                weight_map,
                                self.output_tensor,
                                self.sizeAverage,
                                self.weights,
                                self.total_weight_tensor
                )
                self.output = self.output_tensor[0]
                return self.output

        def updateGradInput(self, input, target, weight_map):
                self.gradInput.resize_as_(input).zero_()
                self._backend.SpatialWeightedClassNLLCriterion_updateGradInput(
                                self._backend.library_state,
                                input,
                                target,
                                weight_map,
                                self.gradInput,
                                self.sizeAverage,
                                self.weights,
                                self.total_weight_tensor
                )
                return self.gradInput

        def forward(self, input, target, weight_map):
                return self.updateOutput(input, target, weight_map)

        def backward(self, input, target, weight_map):
                return self.updateGradInput(input, target, weight_map)

        def type(self, type, tensorCache=None):
                # find all tensors and convert them
                for key, param in self.__dict__.items():
                        setattr(self, key, recursiveType(param, type, tensorCache or {}))

                self._backend = torch._thnn.type2backend[type]
                return self

        def float(self):
                return self.type('torch.FloatTensor')

        def double(self):
                return self.type('torch.DoubleTensor')

        def cuda(self):
                return self.type('torch.cuda.FloatTensor')