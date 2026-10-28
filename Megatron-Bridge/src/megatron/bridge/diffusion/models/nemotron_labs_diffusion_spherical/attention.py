# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native MCore attention consuming the explicit three-segment additive mask."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .stable_attention import attention_score_bound, stable_scaled_dot_product_attention
from .explicit_attention import explicit_attention

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
    NemotronLabsDiffusionAttention,
    _get_llama_4_attn_scale,
    apply_rotary_pos_emb,
    repeat_kv,
)


class SphericalNemotronAttention(NemotronLabsDiffusionAttention):
    """Native projection-compatible attention with per-batch arbitrary masks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spherical_position_ids: torch.Tensor | None = None
        self.spherical_attention_stats: dict = {}

    def set_spherical_position_ids(self, position_ids: torch.Tensor) -> None:
        """Stage semantic RoPE positions for the enclosing transformer-layer call."""
        if position_ids.ndim != 2:
            raise ValueError("spherical position ids must have shape [batch, sequence]")
        if self._spherical_position_ids is not None:
            raise RuntimeError("spherical position ids were not cleared after the prior attention call")
        self._spherical_position_ids = position_ids

    def clear_spherical_position_ids(self) -> None:
        """Drop per-call RoPE state after the core attention computation finishes."""
        self._spherical_position_ids = None

    def _resolve_spherical_positions(
        self,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return semantic query/key RoPE positions for training or static KV cache use."""
        position_ids = self._spherical_position_ids
        if position_ids is None:
            raise RuntimeError("spherical attention requires explicit semantic position ids")
        if position_ids.device != device:
            raise ValueError("spherical position ids must be on the attention device")
        if position_ids.shape != (batch_size, query_length):
            raise ValueError(
                "spherical position ids must match the current attention queries: "
                f"expected={(batch_size, query_length)}, actual={tuple(position_ids.shape)}"
            )
        if query_length == key_length:
            return position_ids, position_ids

        query_start = key_length - query_length
        expected_query_positions = torch.arange(query_start, key_length, device=device).unsqueeze(0)
        if not torch.equal(position_ids, expected_query_positions.expand_as(position_ids)):
            raise ValueError("cached spherical attention requires contiguous semantic query positions")
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        return position_ids, key_positions

    def forward(
        self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None, packed_seq_params=None
    ):
        """Apply RoPE and native scaled-dot-product attention with all three regions."""
        if packed_seq_params is not None:
            raise ValueError("spherical attention does not support packed sequences")
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)
        query_length = query.shape[2]
        key_length = key.shape[2]
        if query_length > key_length:
            raise ValueError("query length cannot exceed cached key length")
        query_start = key_length - query_length
        query_positions, key_positions = self._resolve_spherical_positions(
            batch_size=query.shape[0],
            query_length=query_length,
            key_length=key_length,
            device=query.device,
        )
        query_cos, query_sin = self.rope_embedding_module(query, query_positions)
        key_cos, key_sin = self.rope_embedding_module(key, key_positions)
        query = apply_rotary_pos_emb(query, query, query_cos, query_sin)[0]
        key = apply_rotary_pos_emb(key, key, key_cos, key_sin)[0]
        if self.beta is not None:
            query = query * _get_llama_4_attn_scale(
                query_positions, self.beta, self.max_position_embeddings
            ).unsqueeze(1).to(query.dtype)
        repetitions = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        key = repeat_kv(key, repetitions)
        value = repeat_kv(value, repetitions)
        mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise ValueError("spherical attention mask must have shape [B, 1, Q, K] or [B, 1, K, K]")
            mask = attention_mask.to(device=query.device)
            if mask.shape[-2] == key_length:
                mask = mask[:, :, query_start:key_length, :key_length]
            elif mask.shape[-2:] != (query_length, key_length):
                raise ValueError("spherical attention mask does not match query/key lengths")
        dropout = self.attention_dropout.p if self.training else 0.0
        config = getattr(self, "config", None)
        backend = getattr(config, "spherical_attention_backend", "adaptive_math")
        if backend == "native":
            context = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=mask is None and query_start == 0
            )
        elif backend in {"explicit_fp32", "explicit_fp64"}:
            stats = getattr(self, "spherical_attention_stats", None)
            if stats is not None:
                precision = "math_fp64" if backend == "explicit_fp64" else "math_fp32"
                stats[precision] = stats.get(precision, 0) + 1
                stats["last_backend"] = backend
                # Observation only: this number never selects the arithmetic path.
                bound = attention_score_bound(query, key, mask)
                stats["last_score_bound"] = bound
                stats["max_score_bound"] = max(stats.get("max_score_bound", 0.0), bound)
            context = explicit_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=mask is None and query_start == 0,
                compute_dtype=torch.float64 if backend == "explicit_fp64" else torch.float32,
                max_score_elements=getattr(config, "spherical_attention_max_score_elements", 64 * 1024 * 1024),
                query_chunk_size=getattr(config, "spherical_explicit_fp64_query_chunk_size", 1024) if backend == "explicit_fp64" else 0,
            )
        elif backend == "adaptive_math":
            context = stable_scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=mask is None and query_start == 0,
                native_score_bound=getattr(config, "spherical_attention_native_score_bound", 128.0),
                fp64_score_bound=getattr(config, "spherical_attention_fp64_score_bound", 1024.0),
                max_score_elements=getattr(config, "spherical_attention_max_score_elements", 64 * 1024 * 1024),
                stats=getattr(self, "spherical_attention_stats", None),
            )
        else:
            raise ValueError(f"Unknown spherical attention backend: {backend}")
        context = context.transpose(1, 2).transpose(0, 1)
        return context.contiguous().view(*context.shape[:-2], self.hidden_size_per_partition)


__all__ = ["SphericalNemotronAttention"]
