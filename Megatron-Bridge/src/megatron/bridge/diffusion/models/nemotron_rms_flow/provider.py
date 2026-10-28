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

"""Small provider helpers for direct RMS embedding injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .step import RMSContinuousDenoiser


@dataclass(frozen=True)
class RMSFlowConfig:
    """Explicit provider contract shared by training and inference entry points."""

    block_size: int = 32
    condition_dim: int = 1024
    terminal_mode: str = "scale-matched"
    uniform_gamma_fraction: float = 0.3
    posterior_mode: str = "topk"
    posterior_k: int = 512
    nfe: int = 64
    t_min: float = 0.01
    input_embedding_frozen: bool = True
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    use_megatron_fsdp: bool = False
    use_distributed_optimizer: bool = True

    def __post_init__(self) -> None:
        """Validate the currently supported RMS pilot configuration."""
        if self.block_size != 32:
            raise ValueError("the first RMS flow version requires block_size=32")
        if self.condition_dim != 1024:
            raise ValueError("the first RMS flow version requires condition_dim=1024")
        if self.terminal_mode not in {"scale-matched", "unit-noise", "equal-mask-rms"}:
            raise ValueError("terminal_mode must be 'scale-matched', 'unit-noise', or 'equal-mask-rms'")
        if not 0.0 <= self.uniform_gamma_fraction <= 1.0:
            raise ValueError("uniform_gamma_fraction must be in [0, 1]")
        if self.posterior_mode not in {"topk", "full"}:
            raise ValueError("posterior_mode must be 'topk' or 'full'")
        if self.posterior_k <= 0:
            raise ValueError("posterior_k must be positive")
        if self.nfe < 2:
            raise ValueError("nfe must be at least 2 so the schedule includes an update and final decode")
        if not 0.0 <= self.t_min < 1.0:
            raise ValueError("t_min must be in [0, 1)")
        if not self.input_embedding_frozen:
            raise ValueError("the RMS flow path requires a frozen input embedding")
        if self.tensor_parallelism != 1 or self.pipeline_parallelism != 1:
            raise ValueError("the RMS pilot currently supports only TP=1 and PP=1")
        if self.use_megatron_fsdp:
            raise ValueError("the RMS pilot does not support Megatron FSDP")
        if not self.use_distributed_optimizer:
            raise ValueError("the RMS pilot requires the distributed optimizer")


class NemotronRMSFlowProvider:
    """Construct the independent RMS wrapper around a HF Nemotron backbone.

    The Megatron legacy dLLM provider is deliberately not modified. This
    provider is the narrow integration point for the continuous HF pilot and
    exposes the same config contract to a future TP/PP implementation.
    """

    geometry_type = "rms_normalized_embedding"

    def __init__(self, base_model, mask_token_id: int, config: RMSFlowConfig | None = None, statistics_manifest=None):
        self.config = config or RMSFlowConfig()
        self.model = RMSContinuousDenoiser(
            base_model,
            mask_token_id,
            terminal_mode=self.config.terminal_mode,
            condition_dim=self.config.condition_dim,
            statistics_manifest=statistics_manifest,
        )

    def parameters(self):
        return self.model.parameters()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self


def build_continuous_inputs(context_embeddings, clean_embeddings, state, rms):
    """Return ``[context, clean, R * state]`` without discretizing the state."""
    if state.shape[-1] != clean_embeddings.shape[-1]:
        raise ValueError("continuous state and embedding hidden sizes differ")
    scale = torch.as_tensor(rms, device=state.device, dtype=state.dtype)
    noisy = state * scale
    return torch.cat([context_embeddings, clean_embeddings, noisy], dim=1)


def freeze_input_embedding(model):
    """Freeze the input table while leaving the output head trainable."""
    embedding = model.get_input_embeddings()
    embedding.weight.requires_grad_(False)
    return embedding


__all__ = ["NemotronRMSFlowProvider", "RMSFlowConfig", "build_continuous_inputs", "freeze_input_embedding"]
