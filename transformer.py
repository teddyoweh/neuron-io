import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        
    def forward(self, queries, keys, values, mask=None):
        Q = self.q_linear(queries)
        K = self.k_linear(keys)
        V = self.v_linear(values)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        energy = torch.matmul(Q, K.transpose(-2, -1))
        scaled_attention = energy / math.sqrt(self.head_dim)
        
        if mask is not None:
            scaled_attention = scaled_attention.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scaled_attention, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        output = self.combine_heads(output)
        
        return output
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, num_heads, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
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
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        x = self.fc(x)
        return x
