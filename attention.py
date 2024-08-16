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
# linear layer without bias -> matrix multiplacation between parameter of that layer and argument passed to that layer. 

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
    
'''
Causal Attention Class:
- Adding Dropout.
- Ability to handle batches of data.
'''

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
        )
        '''
        REGISTER BUFFERS:
        1) registering a tensor as a buffer can make our lives a lot easier: 
           We don't have to remember to move tensors to a target device like a GPU manually.
        2) Another advantage of PyTorch buffers, over regular tensors, 
           is that they get included in a model's state_dict
        '''
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape # b ---> is the batch dimension. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill( self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1] ** 0.05,dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

class MutliHeadAttentionWrapper(nn.Module):
    '''
    Wrapper class to create multiple heads of Causal attention class
    '''
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in,d_out,context_length, dropout , qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# Using a wrapper makes the attention model work, but is not an effecient way to run it
# The wrapper uses a for loop, therefore the attention is not calculated parallely across heads. 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # d_out must be divisible by num_heads
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out= d_out
        self.num_heads= num_heads
        self.head_dim = d_out // num_heads

        self.W_query= nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key= nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value= nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout= nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal =1)
        )

    def forward(self,x):

        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # print(keys)
        # print(keys.shape)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # print(keys)
        # print(keys.shape)
        # print("------------------")
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1,2)
        queries =queries.transpose(1,2)
        values = values.transpose(1,2)
        # print(keys)
        # print(keys.shape)
        # print(values.shape)
        # print(values.transpose(2,3).shape)
        # print("------------------")

        attn_scores = queries @ keys.transpose(2, 3) #F
        #print(attn_scores)
        #print(attn_scores.shape)
        # print(self.mask)
        # print(self.mask.shape)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        # print(mask_bool)
        #Dealing with tensor shapes

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        #print(attn_weights)
        #print(attn_weights.shape)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        #print(context_vec)

        return context_vec


        


