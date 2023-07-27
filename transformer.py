import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import SelfAttention
from src.linear import Linear
from src.regularization import Dropout
from src.activation import LayerNorm
from src.preprocessing import Embedding

class ModuleList:
    def __init__(self):
        self.modules = []
        
    def append(self, module):
        self.modules.append(module)
        
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __len__(self):
        return len(self.modules)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward_prop(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        x = self.fc(x)
        return x
