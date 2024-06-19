import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size:int = 256
    vocab_size:int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd : int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config