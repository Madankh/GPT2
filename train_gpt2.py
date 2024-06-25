import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import math
from dataclasses import dataclass


# @dataclass
# class GPTConfig:
#     block_size:int = 256
#     vocab_size:int = 65
#     n_layer : int = 6
#     n_head : int = 6
#     n_embd : int = 384

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd , config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query , values projections for all head but in batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size , config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B , T, C  = x.shape # Batch size , sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        q = q.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        k = k.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        v = v.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.shape(-1)))
        att = att.masked_fill(self.bias[:,:,:T:T] == 0, float('-inf'))
        y = att @ v # (B , nh , T, T) x (B , nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B , T , C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self , x):
        x = x + self.attn(self.ln_1(x))
        x =  x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size :int = 1024
    vocab_size : int = 58257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size , config.n_embd),
            wpe = nn.Embedding(config.block_size , config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx):
        # idxus is if shape (B, T)
        B , T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of lenth {T}, block size"
        # forward the token and positition embedding
        pos = torch.arange(0,T,dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T , n_embed)
        print(idx.shape)
        tok_emb = self.transformer.wte(idx) # position embeddings of shape (B , T, n_embed)
        print(tok_emb.shape)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and time classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B , T , vocab_size)
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingfacve"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # assert model_type in {'gpt-2', 'gpt2_medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_;ater. n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12,n_head = 12, n_embd = 768), # 124M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model
        config_args['block_size'] = 1024
        
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer not param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
# ---------------------------------------------------
model = GPT.from_pretrained('gpt2')
print("did't crash yay!")