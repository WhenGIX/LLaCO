# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Opt-in noisy endpoint decoding; no additional model parameters or mode tokens."""

from collections.abc import Mapping
import math
from typing import Any

DECODE_DEFAULTS = {
    "spherical_decode_training": False,
    "spherical_decode_probability": 0.2,
    "spherical_decode_gamma_min": 0.85,
    "spherical_decode_gamma_max": 0.95,
}


def decode_fields(config: Any) -> dict[str, Any]:
    """Read legacy-compatible flat fields from a config, sidecar or mapping."""
    getter = config.get if isinstance(config, Mapping) else lambda k, d: getattr(config, k, d)
    return {key: getter(key, default) for key, default in DECODE_DEFAULTS.items()}


def validate_decode_config(config: Any) -> dict[str, Any]:
    """Validate the shared training contract before constructing any tensors."""
    fields = decode_fields(config)
    enabled = fields["spherical_decode_training"]
    probability = fields["spherical_decode_probability"]
    lower, upper = fields["spherical_decode_gamma_min"], fields["spherical_decode_gamma_max"]
    if not isinstance(enabled, bool):
        raise ValueError("spherical_decode_training must be a boolean")
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("spherical_decode_probability must be in (0,1]")
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper < 1):
        raise ValueError("decode gamma bounds require 0 < min < max < 1")
    if enabled:
        getter = config.get if isinstance(config, Mapping) else lambda k, d: getattr(config, k, d)
        if getter("spherical_condition_coordinate", "t") != "t":
            raise ValueError("noisy endpoint decoding requires semantic t conditioning")
        if getter("spherical_gamma_uniform_probability", 0.0) != 0.0:
            raise ValueError("decode training currently requires the pure t-uniform denoise sampler")
    return fields


def decode_protocol(config: Any) -> dict[str, Any]:
    """Describe active behavior; ignore inactive hyperparameters for old resumes."""
    fields = validate_decode_config(config)
    if not fields["spherical_decode_training"]:
        return {"enabled": False}
    return {
        "enabled": True,
        "version": "masked_spherical_decode_t0_v1",
        "probability": fields["spherical_decode_probability"],
        "gamma_min": fields["spherical_decode_gamma_min"],
        "gamma_max": fields["spherical_decode_gamma_max"],
        "selection": "per_target_block",
        "gamma_distribution": "uniform_per_block",
        "terminal_noise": "existing_mask_biased_per_token",
        "condition_t": 0.0,
        "denoise_t_grid": "k/1000, k=1..1000",
        "loss": "unchanged_same_position_CE_sample_normalization",
        "inference": "steps_positive_time_forwards_then_one_extra_t0_decode",
        "inference_endpoint": 0.0,
    }


def decode_transition(saved: Any, current: Any, *, allow: bool = False) -> dict[str, Any] | None:
    """Require an explicit opt-in when changing objective distribution on resume."""
    before, after = decode_protocol(saved), decode_protocol(current)
    if before == after:
        return None
    if not allow:
        raise ValueError(
            "decode protocol changed; explicitly set SPHERICAL_ALLOW_DECODE_TRANSITION=1 "
            "for an intentional objective transition"
        )
    return {"from": before, "to": after}


def resolve_decode_endpoint(config: Any, requested: bool | None = None) -> bool:
    """Auto-enable t=0+extra-forward for trained decode models, with explicit legacy-model opt-in."""
    enabled = bool(getattr(config, "spherical_decode_training", False))
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("decode_endpoint must be a boolean or None")
    if enabled and requested is False:
        raise ValueError("decode-trained models require the t=0 endpoint and extra decode forward")
    return enabled or bool(requested)


def copy_decode_fields(config: Any, metadata: Any) -> None:
    """Restore the weight artifact's decode contract in evaluation providers."""
    for key, value in decode_fields(metadata).items():
        if not hasattr(config, key):
            raise ValueError(f"evaluation provider is missing {key}")
        setattr(config, key, value)
    validate_decode_config(config)
