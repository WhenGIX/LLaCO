# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Nemotron spherical diffusion implementation."""

from .adapter import SphericalAdapter
from .bridge import NemotronLabsDiffusionBridge
from .checkpoint import (
    SphericalCheckpointMetadata,
    load_model_only_checkpoint,
    load_spherical_metadata,
    parameter_fingerprint,
    save_model_only_checkpoint,
    validate_resume_metadata,
    write_spherical_metadata,
)
from .conditioning import TimeConditioner
from .geometry import EGeometry, clean_parity, normalize, random_sphere, slerp, terminal_noise
from .inference import NativeKVCache, deterministic_topk, merge_tp_topk, spherical_block_sample
from .layout import ThreeSegmentLayout, build_decoder_input, time_layout, three_segment_attention_mask
from .panel import capture_training_state, preserved_training_state, restore_training_state
from .provider import NemotronLabsDiffusionModelProvider, initialize_spherical_components
from .scheduler import EPSILON_UNIFORM, TimeSample, SphericalScheduler, gamma_from_t, q_fit, t_from_gamma
from .step import SphericalDGPTStep, spherical_ce_loss


__all__ = [
    "EGeometry",
    "TimeConditioner",
    "NativeKVCache",
    "TimeSample",
    "NemotronLabsDiffusionBridge",
    "NemotronLabsDiffusionModelProvider",
    "SphericalAdapter",
    "SphericalCheckpointMetadata",
    "SphericalDGPTStep",
    "SphericalScheduler",
    "ThreeSegmentLayout",
    "build_decoder_input",
    "capture_training_state",
    "clean_parity",
    "EPSILON_UNIFORM",
    "gamma_from_t",
    "q_fit",
    "initialize_spherical_components",
    "load_model_only_checkpoint",
    "load_spherical_metadata",
    "merge_tp_topk",
    "normalize",
    "random_sphere",
    "parameter_fingerprint",
    "preserved_training_state",
    "save_model_only_checkpoint",
    "restore_training_state",
    "slerp",
    "spherical_block_sample",
    "spherical_ce_loss",
    "t_from_gamma",
    "terminal_noise",
    "three_segment_attention_mask",
    "time_layout",
    "deterministic_topk",
    "validate_resume_metadata",
    "write_spherical_metadata",
]
