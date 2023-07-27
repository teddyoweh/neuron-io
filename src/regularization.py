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
class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        
        self.W_ih = np.random.randn(4 * hidden_size, input_size)
        self.W_hh = np.random.randn(4 * hidden_size, hidden_size)
        self.b_ih = np.zeros((4 * hidden_size, 1))
        self.b_hh = np.zeros((4 * hidden_size, 1))
        self.h = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        gates = np.dot(self.W_ih, x) + np.dot(self.W_hh, self.h) + self.b_ih + self.b_hh

        
        gate_i = gates[:self.hidden_size]
        gate_f = gates[self.hidden_size:2 * self.hidden_size]
        gate_o = gates[2 * self.hidden_size:3 * self.hidden_size]
        gate_g = gates[3 * self.hidden_size:]

        
        self.c = gate_f * self.c + gate_i * self.tanh(gate_g)
        self.h = gate_o * self.tanh(self.c)

        return self.h

    def get_hidden_state(self):
        return self.h

    def set_hidden_state(self, hidden_state):
        self.h = hidden_state
