# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gamma sinusoidal embedding and Ada modulation projection."""

import math

import torch
from torch import nn


class GammaConditioner(nn.Module):
    """Encode path progress and project per-layer Ada modulation parameters."""

    def __init__(self, hidden_size: int, condition_dim: int = 512, layers: int = 34):
        super().__init__()
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(256, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim)
        )
        self.projections = nn.ModuleList([nn.Linear(condition_dim, 6 * hidden_size) for _ in range(layers)])
        for proj in self.projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def embed(self, gamma: torch.Tensor):
        original_shape = gamma.shape
        gamma = gamma.float().reshape(-1, 1)
        half = 128
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=gamma.device) / (half - 1))
        sinusoid = torch.cat([(gamma * freq).sin(), (gamma * freq).cos()], dim=-1)
        cond = self.time_mlp(sinusoid.to(dtype=next(self.time_mlp.parameters()).dtype))
        return cond.reshape(*original_shape, self.condition_dim)

    def forward(self, gamma: torch.Tensor):
        cond = self.embed(gamma)
        return [proj(cond).chunk(6, dim=-1) for proj in self.projections]
