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

"""RMS-normalized embedding geometry for continuous Nemotron denoising.

The state lives in the coordinates ``e / R``.  This module intentionally does
not expose a state-normalization helper: token norm differences and Gaussian
terminal variance are part of the modelled geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def compute_global_rms(embedding_weight: torch.Tensor, vocab_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Compute one scalar RMS over the selected finite embedding rows."""
    weight = embedding_weight.detach().float()
    explicit_selection = vocab_ids is not None
    if vocab_ids is not None:
        vocab_ids = vocab_ids.to(device=weight.device, dtype=torch.long)
        weight = weight.index_select(0, vocab_ids)
    finite = torch.isfinite(weight).all(dim=-1)
    if explicit_selection and not bool(finite.all()):
        raise ValueError("V_rms explicitly selected a non-finite embedding row")
    if not bool(finite.any()):
        raise ValueError("V_rms contains no finite embedding rows")
    weight = weight[finite]
    rms = weight.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms) <= 0:
        raise ValueError("V_rms global RMS must be finite and positive")
    return rms


def normalized_embedding(embedding_weight: torch.Tensor, rms: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return embeddings in RMS coordinates without per-row normalization."""
    return embedding_weight[token_ids] / rms.to(device=embedding_weight.device, dtype=embedding_weight.dtype)


def terminal_noise(
    mask_state: torch.Tensor,
    shape: tuple[int, ...],
    b_noise: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``z = m + b_noise * epsilon`` in RMS coordinates."""
    if len(shape) < 1:
        raise ValueError("terminal shape must include the embedding dimension")
    device = device or mask_state.device
    dtype = dtype or mask_state.dtype
    epsilon = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    scale = torch.as_tensor(b_noise, device=device, dtype=torch.float32)
    result = mask_state.to(device=device, dtype=torch.float32) + scale * epsilon
    return result.to(dtype=dtype)


def linear_path(clean_state: torch.Tensor, terminal_state: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Construct ``x_gamma = (1-gamma) x0 + gamma z`` with broadcast-safe gamma."""
    g = gamma.to(device=clean_state.device, dtype=clean_state.dtype)
    while g.ndim < clean_state.ndim:
        g = g.unsqueeze(-1)
    return (1.0 - g) * clean_state + g * terminal_state.to(dtype=clean_state.dtype)


@dataclass(frozen=True)
class RMSGeometry:
    """Immutable geometry metadata and operations bound to an embedding table."""

    rms: torch.Tensor
    mask_state: torch.Tensor
    r_mask: float
    b_noise: float
    terminal_mode: str
    mask_token_id: int
    vocab_size: int

    @classmethod
    def from_embedding(
        cls,
        embedding_weight: torch.Tensor,
        mask_token_id: int,
        *,
        vocab_ids: torch.Tensor | None = None,
        terminal_mode: str = "scale-matched",
        b_noise: float | None = None,
    ) -> "RMSGeometry":
        if not 0 <= mask_token_id < embedding_weight.shape[0]:
            raise ValueError(f"mask_token_id {mask_token_id} is outside the embedding table")
        rms = compute_global_rms(embedding_weight, vocab_ids)
        mask_embedding = embedding_weight[mask_token_id].detach().float()
        if not torch.isfinite(mask_embedding).all():
            raise ValueError("mask embedding must be finite")
        mask_state = mask_embedding / rms
        r_mask = float(mask_state.square().mean().sqrt())
        if terminal_mode not in {"scale-matched", "unit-noise", "equal-mask-rms"}:
            raise ValueError("terminal_mode must be 'scale-matched', 'unit-noise', or 'equal-mask-rms'")
        # Scale matching is defined only for r_mask <= 1.  Do not clamp a
        # negative variance to zero: that would silently change the terminal
        # distribution while leaving the metadata apparently valid.
        if terminal_mode == "scale-matched" and r_mask > 1.0:
            raise ValueError(f"scale-matched terminal is undefined because r_mask={r_mask:.8g} > 1")
        if b_noise is None:
            if terminal_mode == "scale-matched":
                b_noise = math.sqrt(1.0 - r_mask * r_mask)
            elif terminal_mode == "equal-mask-rms":
                b_noise = r_mask
            else:
                b_noise = 1.0
        expected_b = (
            math.sqrt(1.0 - r_mask * r_mask)
            if terminal_mode == "scale-matched"
            else r_mask
            if terminal_mode == "equal-mask-rms"
            else 1.0
        )
        if not math.isclose(float(b_noise), expected_b, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(
                f"b_noise={float(b_noise):.8g} does not match {terminal_mode} terminal (expected {expected_b:.8g})"
            )
        if not math.isfinite(float(b_noise)) or float(b_noise) < 0:
            raise ValueError("b_noise must be finite and non-negative")
        return cls(
            rms.detach(),
            mask_state.detach(),
            r_mask,
            float(b_noise),
            terminal_mode,
            int(mask_token_id),
            embedding_weight.shape[0],
        )

    @property
    def geometry_type(self) -> str:
        return "rms_normalized_embedding"

    def clean(self, embedding_weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return normalized_embedding(embedding_weight, self.rms, token_ids)

    def sample_terminal(self, shape: tuple[int, ...], *, dtype=None, device=None, generator=None) -> torch.Tensor:
        return terminal_noise(self.mask_state, shape, self.b_noise, dtype=dtype, device=device, generator=generator)

    def to_model(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.rms.to(device=state.device, dtype=state.dtype)

    def path(self, clean_state: torch.Tensor, terminal_state: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return linear_path(clean_state, terminal_state, gamma)

    def metadata(self) -> dict:
        return {
            "geometry_type": self.geometry_type,
            "mask_token_id": self.mask_token_id,
            "vocab_size": self.vocab_size,
            "R": float(self.rms),
            "r_mask": self.r_mask,
            "b_noise": self.b_noise,
            "terminal_mode": self.terminal_mode,
        }
