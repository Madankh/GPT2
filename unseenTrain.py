import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size = 50,257
    n_heads = 12
    n_layers = 12
    n_emb = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.emb, 4 * config.emb),
        self.gelu = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4 * config.emb, config.emb)
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x