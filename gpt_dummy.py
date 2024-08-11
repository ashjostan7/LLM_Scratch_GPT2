import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    '''
    Dummy GPT Model to trace the entire GPT2 Model Architecture. 
    '''
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias = False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 1. create embeddings 
        tok_embds = self.tok_emb(in_idx)
        pos_embds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embds + pos_embds

        # 2. Pass data through layers:
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x= self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class DummyTransformerBlock(nn.Module):
    '''
    Dummy Transformer block class top help trace overview of GPT-2 architecture.
    '''
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    '''
    Dummy Layer Normalisation block to help trace overview of GPT-2 architecture.
    '''
    def __init__(self, emd_dim):
        super().__init__()
    
    def forward(self, x):
        return x
