import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from dataclasses import dataclass
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hellaswag import render_example, iterate_examples
import torch._dynamo
torch._dynamo.config.suppress_errors = True



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
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
class DataLoaderLite:
    def __init__(self , B, T, process_rank, num_processes , split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        # assert split in {'train', 'val'}

        # get the share filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        # state , init at shard zero
        self.reset()

        # at init load tokens from disk and store them in memory
        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"load {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        # #state 
        # self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B , T  = self.B , self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B , T)
        self.current_position += B*T*self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.token = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y

# Run the training loop
from torch.distributed import init_process_group, destroy_process_group
# set up DDP 
# torchrun command set the env variables Rank , Local_Rank are world_s9ze
# ddp = int(os.environ.get('RANK', -1)) !=1 # is this a ddp run ? 
ddp = int(os.environ.get('RANK', -1)) != -1  # Corrected line
if ddp:
    # Ensure CUDA is available
    assert torch.cuda.is_available(), "For now we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # This process will do logging, checkpointing, etc.
else:
    # Vanilla non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288
B=16
T=1024 # seq_len
assert total_batch_size % (B * T * ddp_world_size) == 0, "Make same total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T  * ddp_world_size)
if master_process:
    print(f"Total desired batch_size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T,process_rank=ddp_rank, num_processes=ddp_world_size , split="train")
val_loader = DataLoaderLite(B=B , T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
torch.set_float32_matmul_precision('high')

# Create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
warmup_steps = 715
max_steps = 19073
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


# optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 20
        
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
    
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
            val_loss_accum /= dist.get_world_size()
        
        # if master_process:
        #     print(f"validation loss {val_loss_accum.item():.4f}")
        #     # print(f"validation loss: {val_loss_accum.item():.4f}")
        #     with open(log_file, "a") as f:
        #         f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        #     if step > 0 and (step % 1000 == 0 or last_step):
        #         # optionally write model checkpoints
        #         checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        #         checkpoint = {
        #             'model': raw_model.state_dict(),
        #             'config': raw_model.config,
        #             'step': step,
        #             'val_loss': val_loss_accum.item()
        #         }
        #         # you might also want to add optimizer.state_dict() and
        #         # rng seeds etc., if you wanted to more exactly resume training
        #         torch.save(checkpoint, checkpoint_path)

    # # once in a while evaluate hellaswag
    # if (step % 100 == 0 or last_step) and (not use_compile):
    #     num_correct_norm = 0
    #     num_total = 0
    #     for i, example in enumerate(iterate_examples("val")):
    #         # only process examples where i % ddp_world_size == ddp_rank
    #         if i % ddp_world_size != ddp_rank:
    #             continue
    #         # render the example into tokens and labels
    #         _, tokens, mask, label = render_example(example)
    #         tokens = tokens.to(device)
    #         mask = mask.to(device)
    #         # get the logits
    #         with torch.no_grad():
    #             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    #                 logits, loss = model(tokens)
    #             pred_norm = get_most_likely_row(tokens, mask, logits)
    #         num_total += 1
    #         num_correct_norm += int(pred_norm == label)
    #     # reduce the stats across all processes
    #     if ddp:
    #         num_total = torch.tensor(num_total, dtype=torch.long, device=device)
    #         num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
    #         dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
    #         dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
    #         num_total = num_total.item()
    #         num_correct_norm = num_correct_norm.item()
    #     acc_norm = num_correct_norm / num_total
    #     if master_process:
    #         print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
    #         with open(log_file, "a") as f:
    #             f.write(f"{step} hella {acc_norm:.4f}\n")


    # once in a while generate from the model (except step 0, which is noise)
    # disabled because torch.compile throws a scary error i can't solve rn
    # if you disable torch.compile this code works fine
    # if ((step > 0 and step % 100 == 0) or last_step) and (not use_compile):
    #     model.eval()
    #     num_return_sequences = 4
    #     max_length = 32
    #     tokens = enc.encode("Hello, I'm a language model,")
    #     tokens = torch.tensor(tokens, dtype=torch.long)
    #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     xgen = tokens.to(device)
    #     sample_rng = torch.Generator(device=device)
    #     sample_rng.manual_seed(42 + ddp_rank)
    
    #     # Ensure xgen is on the correct device and of the correct dtype
    #     assert xgen.device == device, "xgen is not on the correct device"
    #     assert xgen.dtype == torch.long, "xgen is not of dtype torch.long"
    
    #     while xgen.size(1) < max_length:
    #         # forward the model to get the logits
    #         with torch.no_grad():
    #             # Check if model is on the correct device
    #             assert next(model.parameters()).device == device, "Model is not on the correct device"
    #             # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    #             logits, loss = model(xgen) # (B, T, vocab_size)
    #             # take the logits at the last position
    #             logits = logits[:, -1, :] # (B, vocab_size)
    #             # get the probabilities
    #             probs = F.softmax(logits, dim=-1)
    #             # do top-k sampling of 50 (huggingface pipeline default)
    #             # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    #             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    #             # select a token from the top-k probabilities
    #             # note: multinomial does not demand the input to sum to 1
    #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
    #             # gather the corresponding indices
    #             xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    #             # append to the sequence
    #             xgen = torch.cat((xgen, xcol), dim=1)
    #     # print the generated text
    #     for i in range(num_return_sequences):
    #         tokens = xgen[i, :max_length].tolist()
    #         decoded = enc.decode(tokens)
    #         print(f"rank {ddp_rank} sample {i}: {decoded}")
    


# Tranning loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x , y = train_loader.next_batch()
        x, y = x.to(device) , y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits , loss = model(x , y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0# time difference in mi;eseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_pre_sec = tokens_processed / dt
    if master_process:
        print(f"Step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm : {norm:.4f}| dt : dt {dt*1000:2f}ms , tok/sec:{tokens_pre_sec}")
if ddp:
    destroy_process_group()

import sys; sys.exit(0)   
