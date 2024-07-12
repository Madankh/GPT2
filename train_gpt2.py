import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from dataclasses import dataclass
import time

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd , config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NONOGPT_SCALE_INIT = 1.0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size , config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B , T, C  = x.size() # Batch size , sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        
        q,k,v = qkv.split(self.n_embd, dim=2)
        q = q.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        # print(q.shape , "q" )
        k = k.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)
        v = v.view(B , T, self.n_head, C//self.n_head).transpose(1,2) # (B , nh, T, hs)


        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B , nh , T, T) x (B , nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
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
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, "NONOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer) ** -5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
    
    def forward(self, idx , targets=None):
        # idxus is if shape (B, T)
        B , T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of lenth {T}, block size"
        # forward the token and positition embedding
        pos = torch.arange(0,T,dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T , n_embed)
        tok_emb = self.transformer.wte(idx) # position embeddings of shape (B , T, n_embed)
        # print(idx.shape)
        # print(tok_emb.shape)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and time classifier
        x = self.transformer.ln_f(x)
        # print(x.shape , "x from transformer")
        # print(x.shape , "x")
        logits = self.lm_head(x) # (B , T , vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits , loss
    
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
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        # start with all of the candiddate parameters that require grad
        param_dict = {pn : p for pn , p in self.named_parameters()}
        param_dict = {pn : p for pn , p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n , p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nondecay_params, 'weight_decay' : 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nondecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nondecay_params)}, with {num_nodecay_params:,} parameters")

        fused_avaliable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaliable and 'cuda' in device
        print(f"using fused adam : {use_fused}" )
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

 # --------------------------------------------------
import tiktoken
class DataLoaderLite:
    def __init__(self , B, T):
        self.B = B
        self.T = T
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"load {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        #state 
        self.current_position = 0
    def next_batch(self):
        B , T  = self.B , self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B , T)
        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
# ---------------------------------------------------
device = "cuda"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print("using device" , device)
# de
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B=16
T=1024 # seq_len
assert total_batch_size % (B * T) == 0, "Make same total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch_size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

torch.set_float32_matmul_precision('high')
# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
warmup_steps = 10
max_steps = 50
max_lr = 6e-4
min_lr = max_lr * 0.1

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1)/ warmup_steps
    if it>max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5* (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x , y = train_loader.next_batch()
        x, y = x.to(device) , y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits , loss = model(x , y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0# time difference in mi;eseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_pre_sec = tokens_processed / dt
    print(f"Step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm : {norm:.4f}| dt : dt {dt*1000:2f}ms , tok/sec:{tokens_pre_sec}")

import sys; sys.exit(0)   


