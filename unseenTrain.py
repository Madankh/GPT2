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