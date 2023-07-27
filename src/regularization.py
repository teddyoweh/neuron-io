import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        
    def forward_prop(self, x, train_mode=True):
        if train_mode:
            self.mask = np.random.rand(*x.shape) < self.p
            return x * self.mask / (1 - self.p)
        else:
            return x
        
    def backward_prop(self, grad_output):
        return grad_output * self.mask / (1 - self.p)
