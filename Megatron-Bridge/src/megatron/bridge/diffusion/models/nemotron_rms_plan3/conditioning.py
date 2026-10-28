"""Semantic-time conditioner; t, never gamma, is the model condition."""
from __future__ import annotations
import math
import torch
from torch import nn

class TimeConditioner(nn.Module):
    def __init__(self, hidden_size: int, condition_dim: int = 1024, layers: int = 1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(256, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.projections = nn.ModuleList([nn.Linear(condition_dim, 6 * hidden_size) for _ in range(layers)])
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, t):
        t = torch.as_tensor(t)
        if t.ndim == 0:
            t = t.reshape(1)
        original_shape = t.shape
        t = t.float().reshape(-1, 1)
        half = 128
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        value = self.time_mlp(
            torch.cat([(t * freq).sin(), (t * freq).cos()], dim=-1).to(self.time_mlp[0].weight.dtype)
        ).reshape(*original_shape, -1)
        return [tuple(part.reshape(*original_shape, -1) for part in projection(value).chunk(6, dim=-1)) for projection in self.projections]
