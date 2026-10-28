"""The sole Plan 3 observation-noise definition shared by train and inference."""
from __future__ import annotations
from enum import Enum
import torch

class NoiseMode(str, Enum):
    A_DIRECT_ONSET = "A_direct_onset"
    B_TAPERED_ONSET = "B_tapered_onset"

def observation_sigma(mode: NoiseMode | str, t, sigma_10):
    mode = NoiseMode(mode)
    t_tensor = torch.as_tensor(t) if not isinstance(t, torch.Tensor) else t
    sigma = torch.as_tensor(sigma_10, device=t_tensor.device, dtype=t_tensor.dtype)
    if mode is NoiseMode.A_DIRECT_ONSET:
        result = sigma
    else:
        result = t_tensor * sigma
    return result

def add_observation_noise(state, mode, t, sigma_10, *, generator=None):
    sigma = observation_sigma(mode, t, sigma_10)
    while sigma.ndim < state.ndim:
        sigma = sigma.unsqueeze(-1)
    epsilon = torch.randn(state.shape, device=state.device, dtype=torch.float32, generator=generator)
    return state.float() + sigma.float() * epsilon
