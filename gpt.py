import torch
import torch.nn as nn 
from attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale= nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                                    GELU(),
                                    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                                    )
    
    def forward(self, x):
        return self.layers(x)

#Sample cfg:
# GPT_CONFIG_124M = {
# "vocab_size": 50257, # Vocabulary size
# "context_length": 1024, # Context length
# "emb_dim": 768, # Embedding dimension
# "n_heads": 12, # Number of attention heads
# "n_layers": 12, # Number of layers
# "drop_rate": 0.1, # Dropout rate
# "qkv_bias": False # Query-Key-Value bias
# }

# MultiHeadAttention init:
#def __init__(self, d_in, d_out, context_lenght, dropout, num_heads, qkv_bias=False)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_resid = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_resid(x)

        x = x + shortcut
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)

        x = x + shortcut

        return x

class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, input):
        batch_size, seq_len = input.shape
        tok_embs = self.tok_emb(input)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=input.device))

        X = tok_embs + pos_embds
        X = self.drop_emb(X)
        X = self.trf_blocks(X)
        X = self.final_norm(X)
        logits = self.out_head(X)
        return logits


