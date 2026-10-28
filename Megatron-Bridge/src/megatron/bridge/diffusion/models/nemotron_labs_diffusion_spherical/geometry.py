# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Spherical geometry and the frozen native embedding snapshot."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return row-wise unit vectors, rejecting non-finite or zero rows."""
    if not torch.isfinite(value).all():
        raise ValueError("spherical vectors must be finite")
    norm = value.float().norm(dim=-1, keepdim=True)
    if (norm <= eps).any():
        raise ValueError("spherical vectors must be non-degenerate")
    return value / norm.to(dtype=value.dtype)


def random_sphere(shape: tuple[int, ...], *, device=None, dtype=torch.bfloat16, generator=None) -> torch.Tensor:
    """Sample uniformly distributed unit vectors."""
    value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return (value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype)


def terminal_noise(
    random_direction: torch.Tensor,
    mask_direction: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Construct a normalized terminal state from random and mask directions."""
    state = random_direction + mask_direction
    norm = state.float().norm(dim=-1, keepdim=True)
    bad = norm.squeeze(-1) <= 1e-6
    if bad.any():
        replacement = random_sphere(
            (int(bad.sum()), state.shape[-1]), device=state.device, dtype=state.dtype, generator=generator
        )
        state = state.clone()
        state[bad] = replacement + mask_direction.expand_as(state)[bad]
        norm = state.float().norm(dim=-1, keepdim=True)
    return state / norm.clamp_min(1e-6).to(dtype=state.dtype)


def slerp(x0: torch.Tensor, x1: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Interpolate two vectors on the unit sphere with stable endpoint limits."""
    work_dtype = torch.float64 if x0.dtype == torch.float64 or x1.dtype == torch.float64 else torch.float32
    a = normalize(x0).to(work_dtype)
    b = normalize(x1).to(work_dtype)
    g = torch.as_tensor(gamma, device=x0.device, dtype=work_dtype)
    while g.ndim < a.ndim:
        g = g.unsqueeze(-1)
    dot = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = b - dot * a
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    linear = (1 - g) * a + g * b
    linear = linear / linear.norm(dim=-1, keepdim=True).clamp_min(eps)
    curved = torch.cos(g * torch.atan2(tangent_norm, dot)) * a + torch.sin(
        g * torch.atan2(tangent_norm, dot)
    ) * tangent / tangent_norm.clamp_min(eps)
    result = torch.where(tangent_norm <= eps, linear, curved)
    result = torch.where(g == 0, a, torch.where(g == 1, b, result))
    return normalize(result).to(dtype=x0.dtype)


class EGeometry(nn.Module):
    """Persistent full-vocabulary spherical codebook copied from native input embeddings."""

    def __init__(self, embedding_weight: torch.Tensor):
        super().__init__()
        if embedding_weight.ndim != 2 or not torch.isfinite(embedding_weight).all():
            raise ValueError("native input embedding must be a finite [vocab, hidden] matrix")
        snapshot = embedding_weight.detach().clone()
        self.register_buffer("E_geometry", snapshot, persistent=True)
        # The normalized codebook is derived per chunk to avoid a second full-vocabulary
        # buffer on every TP rank.

    def refresh_codebook(self) -> None:
        """Keep the checkpoint-load hook while using lazy normalized codewords."""
        return None

    def normalized_chunk(
        self, start: int, end: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Return a normalized codebook slice without materializing the full codebook."""
        chunk = F.normalize(self.E_geometry[start:end].float(), dim=-1)
        if device is not None:
            chunk = chunk.to(device=device)
        return chunk.to(dtype=dtype)

    def codeword(
        self, token_ids: torch.Tensor | int, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Return normalized geometry vectors for one or more vocabulary ids."""
        ids = torch.as_tensor(token_ids, device=self.E_geometry.device, dtype=torch.long)
        values = F.normalize(self.E_geometry[ids].float(), dim=-1)
        if device is not None:
            values = values.to(device=device)
        return values if dtype is None else values.to(dtype=dtype)

    def codewords(
        self, token_ids: torch.Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Vectorized alias for codeword lookup used by inference."""
        return self.codeword(token_ids, device=device, dtype=dtype)

    @classmethod
    def from_native_embedding(cls, embedding: nn.Embedding | nn.Module) -> "EGeometry":
        """Copy a native embedding exactly once into a frozen geometry module."""
        weight = getattr(embedding, "weight", None)
        if weight is None:
            raise TypeError("native embedding must expose a weight tensor")
        expected_vocab = getattr(embedding, "num_embeddings", None)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from megatron.core import parallel_state

            group = parallel_state.get_tensor_model_parallel_group()
            world = torch.distributed.get_world_size(group=group)
            if world > 1:
                shards = [torch.empty_like(weight) for _ in range(world)]
                torch.distributed.all_gather(shards, weight.detach().contiguous(), group=group)
                weight = torch.cat(shards, dim=0)
        if expected_vocab is not None and weight.shape[0] != int(expected_vocab):
            raise ValueError(
                "native embedding gather produced an unexpected vocabulary size: "
                f"expected={int(expected_vocab)}, got={weight.shape[0]}; "
                "the input embedding may already be replicated across TP"
            )
        return cls(weight)

    @property
    def vocab_size(self) -> int:
        return self.E_geometry.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.E_geometry.shape[1]

    def clean(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed complete-vocabulary token ids using the frozen snapshot."""
        return F.embedding(token_ids, self.E_geometry)

    def x0_hat(self, posterior: torch.Tensor, *, chunk_size: int = 4096) -> torch.Tensor:
        """Compute posterior expectation in geometry coordinates without full logits materialization."""
        if posterior.shape[-1] != self.vocab_size:
            raise ValueError("posterior must use the complete vocabulary")
        result = torch.zeros(*posterior.shape[:-1], self.hidden_size, device=posterior.device, dtype=torch.float32)
        for start in range(0, self.vocab_size, chunk_size):
            result += (
                posterior[..., start : start + chunk_size].float() @ self.normalized_chunk(start, start + chunk_size)
            )
        return normalize(result).to(dtype=self.E_geometry.dtype)


def clean_parity(native: torch.Tensor, geometry: torch.Tensor, *, atol: float = 0.0) -> dict[str, float | bool]:
    """Compare native embedding output and geometry decoder input in eval mode."""
    delta = (native.float() - geometry.float()).abs()
    maximum = float(delta.max().item()) if delta.numel() else 0.0
    return {"max_abs_error": maximum, "passed": maximum <= atol}


__all__ = ["EGeometry", "clean_parity", "normalize", "random_sphere", "slerp", "terminal_noise"]
