# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Three-segment native decoder layout and attention semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import EGeometry


@dataclass(frozen=True)
class ThreeSegmentLayout:
    """Coordinates for ``[context | clean | noisy]`` in a fixed-width batch."""

    context_width: int
    target_width: int
    block_size: int

    @property
    def noisy_start(self) -> int:
        return self.context_width + self.target_width

    @property
    def total_length(self) -> int:
        return self.context_width + 2 * self.target_width

    def validate(self) -> None:
        """Validate block and sequence invariants."""
        if self.context_width <= 0 or self.target_width <= 0 or self.block_size <= 0:
            raise ValueError("layout widths and block_size must be positive")
        if self.target_width % self.block_size:
            raise ValueError("target width must be block aligned")


def continuous_position_ids(
    context_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    context_width: int,
    target_width: int,
) -> torch.Tensor:
    """Build RoPE positions using effective diffusion lengths, including tail EOS.

    ``target_lengths`` is never the original unfilled answer length. Clean and
    noisy copies share continuous semantic positions; only outer pad is zero.
    """
    if context_lengths.shape != target_lengths.shape:
        raise ValueError("context_lengths and target_lengths must have the same shape")
    device = context_lengths.device
    context_offset = torch.arange(context_width, device=device).unsqueeze(0)
    response_offset = torch.arange(target_width, device=device).unsqueeze(0)
    context = context_offset.expand(context_lengths.numel(), -1)
    response = context_lengths.reshape(-1, 1) + response_offset
    context = torch.where(context_offset < context_lengths.reshape(-1, 1), context, 0)
    response = torch.where(response_offset < target_lengths.reshape(-1, 1), response, 0)
    return torch.cat((context, response, response), dim=1)


def three_segment_attention_mask(
    context_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    context_width: int,
    target_width: int,
    block_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build causal context/clean and block-bidirectional noisy attention.

    ``target_lengths`` means effective diffusion lengths (including tail EOS).
    Outer padding is invisible to valid queries; invalid queries attend key 0.
    """
    if context_lengths.shape != target_lengths.shape:
        raise ValueError("context_lengths and target_lengths must have the same shape")
    batch = context_lengths.numel()
    total = context_width + 2 * target_width
    if block_size <= 0 or target_width % block_size:
        raise ValueError("target_width must be block aligned")
    if ((context_lengths < 1) | (context_lengths > context_width) | (target_lengths < 1) | (target_lengths > target_width)).any():
        raise ValueError("sample lengths exceed the fixed three-segment widths")
    positions = torch.arange(total, device=context_lengths.device)
    regions = torch.where(positions < context_width, 0, torch.where(positions < context_width + target_width, 1, 2))
    offsets = torch.where(regions == 0, positions, (positions - context_width) % target_width)
    q_region, k_region = regions[:, None], regions[None, :]
    q_offset, k_offset = offsets[:, None], offsets[None, :]
    allowed = (
        ((q_region == 0) & (k_region == 0) & (k_offset <= q_offset))
        | ((q_region == 1) & ((k_region == 0) | ((k_region == 1) & (k_offset <= q_offset))))
        | ((q_region == 2) & (
            (k_region == 0)
            | ((k_region == 1) & (k_offset < (q_offset // block_size) * block_size))
            | ((k_region == 2) & (k_offset // block_size == q_offset // block_size))
        ))
    )
    lengths = torch.where(regions[None, :] == 0, context_lengths.reshape(-1, 1), target_lengths.reshape(-1, 1))
    valid = offsets[None, :] < lengths
    allowed = allowed[None, :, :] & valid[:, :, None] & valid[:, None, :]
    allowed |= (~valid[:, :, None]) & (positions[None, None, :] == 0)
    result = torch.full((batch, 1, total, total), torch.finfo(dtype).min, device=context_lengths.device, dtype=dtype)
    return result.masked_fill_(allowed[:, None], 0)


def build_decoder_input(
    geometry: EGeometry,
    adapter,
    context_ids: torch.Tensor,
    clean_ids: torch.Tensor,
    noisy_state: torch.Tensor,
    *,
    return_adapter_stats: bool = False,
):
    """Build native MCore ``decoder_input`` without changing the embedding module."""
    context = geometry.clean(context_ids)
    clean = geometry.clean(clean_ids)
    adapted = adapter(noisy_state, return_stats=return_adapter_stats)
    noisy, stats = adapted if return_adapter_stats else (adapted, None)
    result = torch.cat((context, clean, noisy), dim=1)
    return (result, stats) if return_adapter_stats else result


def time_layout(
    time_blocks: torch.Tensor, *, context_width: int, target_width: int, block_size: int
) -> torch.Tensor:
    """Expand block semantic time with zero context and clean regions."""
    if target_width % block_size or time_blocks.shape[-1] != target_width // block_size:
        raise ValueError("semantic-time block count must exactly cover target_width")
    clean_prefix = torch.zeros(
        (*time_blocks.shape[:-1], context_width + target_width), device=time_blocks.device, dtype=time_blocks.dtype
    )
    return torch.cat((clean_prefix, time_blocks.repeat_interleave(block_size, dim=-1)), dim=-1)


__all__ = [
    "ThreeSegmentLayout",
    "build_decoder_input",
    "continuous_position_ids",
    "time_layout",
    "three_segment_attention_mask",
]
