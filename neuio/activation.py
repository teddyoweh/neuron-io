import torch
import torch.nn as nn
import numpy as np
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.relu_()
        else:
            x = torch.relu(x)
        return x


class Softmax:
    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, x):
        exp_vals = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=self.dim, keepdims=True)
        return self.output

    def backward(self, grad_output):
        softmax_output = self.output
        grad_input = grad_output * softmax_output * (1 - softmax_output)
        return grad_input
