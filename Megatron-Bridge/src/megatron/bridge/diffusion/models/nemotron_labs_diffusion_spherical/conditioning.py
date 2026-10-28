# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Explicit scalar conditioning for native transformer blocks."""

import math

import torch
from torch import nn

from ...nemotron_labs_diffusion_spherical_protocol import SPHERICAL_CONDITION_DIM


def conditioning_input(time: torch.Tensor, gamma: torch.Tensor, coordinate: str = "t") -> torch.Tensor:
    """Select the conditioner coordinate without changing the noise schedule."""
    if coordinate == "t":
        return time
    if coordinate == "gamma":
        return gamma
    raise ValueError("spherical_condition_coordinate must be 't' or 'gamma'")


def mark_tp_replicated_parameters(module: nn.Module) -> nn.Module:
    """Tell MCore to average this replicated module's gradients over TP once."""
    for parameter in module.parameters():
        parameter.average_gradients_across_tp_domain = True
    return module


def ada_condition_input(condition: torch.Tensor, activation: str = "identity") -> torch.Tensor:
    """Select the block-conditioning activation without changing checkpoint keys."""
    if activation == "identity":
        return condition
    if activation == "silu":
        return torch.nn.functional.silu(condition)
    raise ValueError("spherical_ada_activation must be 'identity' or 'silu'")


def condition_parameter_dtype(config, model_dtype: torch.dtype) -> torch.dtype:
    mode = getattr(config, "spherical_condition_compute_dtype", "model")
    if mode not in {"model", "fp32"}:
        raise ValueError("spherical_condition_compute_dtype must be 'model' or 'fp32'")
    return torch.float32 if mode == "fp32" else model_dtype


def keep_condition_parameters_fp32(module: nn.Module) -> None:
    """Use Bridge's existing preservation hook before mixed-precision wrapping/DDP."""
    module.float()
    for child in module.modules():
        names = tuple(name for name, _ in child.named_parameters(recurse=False))
        if names:
            child._keep_in_float32_parameter_names = names


class TimeConditioner(nn.Module):
    """Embed the configured per-token t or gamma coordinate.

    The historical module/state-dict name is retained for tensor compatibility.
    Callers select the input coordinate with ``conditioning_input``.
    """

    def __init__(self, hidden_size: int, layers: int, condition_dim: int = SPHERICAL_CONDITION_DIM,
                 output_rms_max: float | None = None, time_embedding_scale: float = 1.0,
                 output_norm: str = "none"):
        super().__init__()
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(256, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim)
        )
        self.hidden_size = hidden_size
        self.layers = layers
        if output_rms_max is not None and output_rms_max <= 0:
            raise ValueError("output_rms_max must be positive or None")
        self.output_rms_max = output_rms_max
        if not math.isfinite(time_embedding_scale) or time_embedding_scale <= 0:
            raise ValueError("time_embedding_scale must be finite and positive")
        self.time_embedding_scale = float(time_embedding_scale)
        if output_norm not in {"none", "layernorm"}:
            raise ValueError("output_norm must be 'none' or 'layernorm'")
        self.output_norm = output_norm
        self.output_normalization = (
            nn.LayerNorm(condition_dim, eps=1e-6, elementwise_affine=False)
            if output_norm == "layernorm" else nn.Identity()
        )

    def raw_embed(self, time: torch.Tensor) -> torch.Tensor:
        """Encode semantic time with a deterministic sinusoidal basis; accepts [B,S]."""
        shape = time.shape
        value = time.float().reshape(-1, 1).clamp(0.0, 1.0) * self.time_embedding_scale
        frequency = torch.exp(-math.log(10000.0) * torch.arange(128, device=value.device) / 127)
        sinusoid = torch.cat([(value * frequency).sin(), (value * frequency).cos()], dim=-1)
        output = self.time_mlp(sinusoid.to(next(self.time_mlp.parameters()).dtype)).reshape(*shape, self.condition_dim)
        return output

    def embed(self, time: torch.Tensor) -> torch.Tensor:
        """Normalize each token's condition channels before the per-layer Ada path."""
        output = self.output_normalization(self.raw_embed(time))
        if self.output_rms_max is not None:
            rms = output.float().square().mean(dim=-1, keepdim=True).sqrt()
            scale = (self.output_rms_max / rms.clamp_min(torch.finfo(torch.float32).tiny)).clamp_max(1.0)
            output = output * scale.to(output.dtype)
        return output

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return the compact [B,S,C] condition passed through native block calls."""
        return self.embed(time)


__all__ = ["TimeConditioner", "conditioning_input", "mark_tp_replicated_parameters"]
