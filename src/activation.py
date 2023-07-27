import numpy as np
 
class ReLU():
    def __init__(self, inplace=False):
        pass
    def _max(self,x):
        try:
            return np.maximum(0,x)
        except:
            return 0
    def forward_prop(self, x):
        return self._max(x)
class Softmax:
    def __init__(self, dim=1):
        self.dim = dim

    def forward_prop(self, x):
        exp_vals = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=self.dim, keepdims=True)
        return self.output

    def backward_prop(self, grad_output):
        softmax_output = self.output
        grad_input = grad_output * softmax_output * (1 - softmax_output)
        return grad_input


class LayerNorm:
    def __init__(self, d_model, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.mean = None
        self.variance = None
        
    def forward_prop(self, x, train_mode=True):
        if train_mode:
            self.mean = np.mean(x, axis=-1, keepdims=True)
            self.variance = np.var(x, axis=-1, keepdims=True)
            x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)
            return self.gamma * x_normalized + self.beta
        else:
            return x
        
    def backward_prop(self, grad_output):
        m = grad_output.shape[-1]
        grad_normalized = grad_output * self.gamma
        grad_variance = np.sum(grad_normalized * (grad_output - self.mean), axis=-1, keepdims=True) * -0.5 * (self.variance + self.epsilon) ** (-1.5)
        grad_mean = np.sum(grad_normalized * -1.0 / np.sqrt(self.variance + self.epsilon), axis=-1, keepdims=True) + grad_variance * np.mean(-2.0 * (grad_output - self.mean), axis=-1, keepdims=True)
        grad_x = grad_normalized / np.sqrt(self.variance + self.epsilon) + grad_variance * 2.0 * (grad_output - self.mean) / m + grad_mean / m
        return grad_x
