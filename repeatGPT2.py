import os
import math
import glob
import struct
import inspect

import numpy as np
import torch
import torch.nn as nn
import time
import dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size:int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers : int = 12 # number of layers
    n_head : int = 12 # number of heads
    n_embd : int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tranformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_enbd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        

