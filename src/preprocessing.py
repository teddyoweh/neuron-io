import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_matrix = np.random.randn(vocab_size, d_model)
        self.grad_input = None

    def forward(self, x):
        if isinstance(x, int):
            return self.embedding_matrix[x]
        elif isinstance(x, np.ndarray):
            return self.embedding_matrix[x]
        else:
            raise TypeError("Input must be an integer or a NumPy array of integers.")

    def backward(self, grad_output, x):
        self.grad_input = np.zeros_like(self.embedding_matrix)
        if isinstance(x, int):
            self.grad_input[x] = grad_output
        elif isinstance(x, np.ndarray):
            np.add.at(self.grad_input, x, grad_output)
        else:
            raise TypeError("Input must be an integer or a NumPy array of integers.")
        return self.grad_input

    def parameters(self):
        return [self.embedding_matrix]

    def zero_grad(self):
        self.grad_input = None

    def __call__(self, x):
        return self.forward(x)

    def unsqueeze(self, x, dim):
        if isinstance(x, int):
            return np.expand_dims(self.embedding_matrix[x], axis=dim)
        elif isinstance(x, np.ndarray):
            return np.expand_dims(self.embedding_matrix[x], axis=dim)
        else:
            raise TypeError("Input must be an integer or a NumPy array of integers.")
