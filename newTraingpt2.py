import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from dataclasses import dataclass
import os

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, values projections for all head but in batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B,T,C = x.size()  # Batch size , sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2) 
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        # print(q.shape, "q")
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1))
        att = att.masked_fill(self.bias[:,:,:T,:T] ==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B , nh , T, T) x (B , nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B , T , C) # re-assemble all head outputs side by side 
        y = self.c_proj(y)
        return y


 
@dataclass
class GPTConfig:
    block_size : int = 1024 # Max sequence length
    vocab_size : int = 50257 # number of tokens
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size , config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # self.apply(self._)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B , T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size "
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shap-e (T)
        pos_emb = self.transformer.wpe(pos) # position emebding 
        tok_emb = self.transformer.wte(idx) 
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits,  loss

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

# ---------------------
# Attemp to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# perfix tokens
import tiktoken

class DataLoaderLite:
    def __init__(self, B ,T) -> None:
        self.B = B
        self.T = T

        # At init load tokens from disk and   store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # Input
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B  * T
        # if loading the next batch would be out of bounds , reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position  = 0
        return x, y

train_loader = DataLoaderLite(B=4, T=32)
# get logits
model = GPT(GPTConfig())
model.to(device)
# logits, loss = model(x, y)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    x,y =  train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss : {loss.item()}")

# print(logits.shape)
print(loss, "loss")
import sys;
sys.exit(0)

print("didn't crash yayyyyy")
