import torch
import torch.nn as nn
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):

        keys = x @ self.W_key
        query = x @ self.W_query
        value = x @ self.W_value

        attn_scores = query @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ value

        return context_vec
    

# Making the W_query, W_key and W_value a linear layer (with bias set to false) is an effecient way of doing matrix multiplication.
# Parameter initilisation is better with the linear layer too. Better than rand function. 
# linear layer without bias ->matrix multiplacation between parameter of that layer and argument passed to that layer. 

import torch
import torch.nn as nn
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
    
    def forward(self, x):

        keys = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_scores = query @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ value

        return context_vec