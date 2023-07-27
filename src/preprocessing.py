import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_matrix = np.random.randn(vocab_size, d_model)   
        
    def forward(self, x):

        return self.embedding_matrix[x]
    
    def backward(self, grad_output,x):
        grad_input = np.zeros_like(self.embedding_matrix)
        np.add.at(grad_input, x, grad_output)
        return grad_input
