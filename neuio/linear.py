import numpy as np

class Linear:
    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights = np.random.randn(hidden_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
