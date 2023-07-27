import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.tokenizer import Tokenizer
from src.activation import ReLU, Softmax
from src.vectorizer import BuildVectors
from src.linear import Linear
from src.loss import Loss

def matmul(input1, input2):
    m1, n1 = input1.shape
    m2, n2 = input2.shape
    result = np.zeros((m1, n2))
    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                result[i, j] += input1[i, k] * input2[k, j]

    return result
class BuildClassifier:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.output = Linear(hidden_size, output_size)
        self.softmax = Softmax(dim=1)
        self.loss = Loss.mean_squared_error

    def forward_prop(self, x):
        hidden = self.relu(self.hidden.forward(x))
        output = self.softmax.forward(self.output.forward(hidden))
        return output

    def train(self, x, y, learning_rate=0.1, num_epochs=10):
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            output = self.forward_prop(x)
            loss = self.loss(output, y)
            grad_output = self.softmax.backward(output - y)
            grad_hidden = self.output.backward_prop(grad_output, learning_rate)
            grad_input = self.hidden.backward(grad_hidden, learning_rate)
            
            print(f"Loss: {loss}")
    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

def shape(array):
    if isinstance(array, list):
        return (len(array),) + shape(array[0])
    else:
        return ()

df = pd.read_csv('./data/twitter_training.csv')
df = df.dropna()
text_data = df['text']
sentiment_data = df['sentiment']
label_encoder = Tokenizer()
sentiment_labels = label_encoder.fit_tokenize(sentiment_data)
vectorizer = BuildVectors()
vectorizer.fit(text_data)
print(vectorizer.vocab_store['coming'])
train_vectors =vectorizer.transform(text_data[0:10])


input_size = shape(train_vectors)[1]
hidden_size = 64
output_size = len(label_encoder.classes)  

model = BuildClassifier(input_size, hidden_size, output_size)
model.train(train_vectors, sentiment_labels)


def run_prediction(model, text):
    text_vector = vectorizer.transform([text])
    _, predicted_index = torch.max(model.predict(text_vector), 1)
    predicted_label = label_encoder.inverse_transform(predicted_index.numpy())[0]
    return predicted_label

