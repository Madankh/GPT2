import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size = 50257
    n_head = 12
    n_layers = 12
    n_embd = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, 4 * config.n_embd),
        self.gelu = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd)
        self.final_layer = nn.Linear(config.n_embd, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
    def forward(self,x):
        B, T, C = x.size() # Batch size , sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q,k,v = self.c_attn(x)
        q = q.view(B , T, self.n_heads, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        k = k.view(B , T, self.n_heads, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        v = v.view(B , T, self.n_heads, C//self.n_head).transpose(1,2) # (B , nh, T, hs)

        # att 
        # att = (q @ k.tranpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,  float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v 
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.final_layer(y)
        return y
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.selfattention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self,x):
        x = x + self.selfattention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size : int = 58257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
             wte = nn.Embedding(config.vocab_size, config.n_embd),
             wpe = nn.Embedding(config.block_size, config.n_embd),
             h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
             ln_r = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = pos_emb + tok_emb
        for block in self.transformer.h:
            block(x)
        x = self.transformer.ln_r(x)
        logits = self.lm_head(x)
        loss = None
        if targets is  not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


        