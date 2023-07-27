import torch
import torch.nn as nn
import torch.nn.functional as F
from src.linear import Linear
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.head_dim = d_model // num_heads
        
        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        
    def forward_prop(self, queries, keys, values, mask=None):
        Q = self.q_linear(queries)
        K = self.k_linear(keys)
        V = self.v_linear(values)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        energy = torch.matmul(Q, K.transpose(-2, -1))
        scaled_attention = energy / (self.head_dim ** 0.5)
        
        if mask is not None:
            scaled_attention = scaled_attention.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmxax(scaled_attention, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        output = self.combine_heads(output)
        output = self.out_linear(output)
        
        return output, attention_weights
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, num_heads, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)