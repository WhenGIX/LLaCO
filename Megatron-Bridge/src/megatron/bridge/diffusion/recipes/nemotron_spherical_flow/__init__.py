# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Nemotron spherical-flow protocol validation.

This module does not provide a native Megatron training recipe.  The executable
six-GPU launcher uses these fields to validate its HF compatibility pilot while
the native DatasetProvider and forward step remain unimplemented.  The
configuration is deliberately explicit so the intended protocol cannot drift.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SphericalFlowConfig:
    """First-version spherical Nemotron training protocol."""

    tensor_model_parallel_size: int = 3
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    data_parallel_size: int = 2
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 6
    global_batch_size: int = 12
    seq_length: int = 1024
    target_max_tokens: int = 480
    block_size: int = 32
    condition_dim: int = 1024
    ada_lr_mult: float = 10.0
    adapter_lr_mult: float = 10.0
    ada_projection_weight_decay: float = 0.0
    adapter_bias_weight_decay: float = 0.0
    log_beta_weight_decay: float = 0.0
    inference_interval: int = 100
    resume_interval: int = 500
    model_only_interval: int = 2000
    full_vocabulary: bool = True
    drop_oversized_targets: bool = True

    def __post_init__(self) -> None:
        """Reject protocol values that would violate the implementation contract."""
        if (self.tensor_model_parallel_size, self.pipeline_model_parallel_size, self.context_parallel_size) != (3, 1, 1):
            raise ValueError("spherical Nemotron requires TP=3, PP=1, CP=1")
        if self.data_parallel_size != 2:
            raise ValueError("the six-GPU spherical recipe requires DP=2")
        if self.block_size != 32 or self.condition_dim != 1024:
            raise ValueError("spherical flow requires block_size=32 and condition_dim=1024")
        if self.seq_length <= 0 or self.target_max_tokens <= 0 or self.seq_length < 2 * self.target_max_tokens + 1:
            raise ValueError("seq_length must fit two complete target copies and context")
        if self.global_batch_size != self.data_parallel_size * self.micro_batch_size * self.gradient_accumulation_steps:
            raise ValueError("global_batch_size must equal DP * micro_batch_size * gradient accumulation")
        if not self.full_vocabulary:
            raise ValueError("spherical flow uses the complete vocabulary for V_in and V_id")
        if not self.drop_oversized_targets:
            raise ValueError("oversized targets must be dropped rather than truncated")
        for name in ("inference_interval", "resume_interval", "model_only_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (self.inference_interval, self.resume_interval, self.model_only_interval) != (100, 500, 2000):
            raise ValueError("spherical checkpoint and inference cadence is fixed at 100/500/2000 steps")


def nemotron_spherical_flow_config(**overrides: object) -> SphericalFlowConfig:
    """Return the validated six-GPU TP=3 protocol configuration."""
    unknown = sorted(set(overrides) - set(SphericalFlowConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown spherical flow config fields: {', '.join(unknown)}")
    return SphericalFlowConfig(**overrides)


__all__ = ["SphericalFlowConfig", "nemotron_spherical_flow_config"]
