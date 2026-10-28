# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native distributed checkpoint helpers for the spherical protocol."""

from __future__ import annotations

import hashlib
import math
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.utils import unwrap_model

from megatron.bridge.training.checkpointing import _generate_model_state_dict

from ...nemotron_labs_diffusion_spherical_protocol import (
    LEGACY_SPHERICAL_PROTOCOL_VERSION,
    SPHERICAL_CONDITION_DIM,
    SPHERICAL_PROTOCOL_VERSION,
    SUPPORTED_SPHERICAL_CONDITION_DIMS,
)


from .decode_protocol import DECODE_DEFAULTS, decode_transition, validate_decode_config


logger = logging.getLogger(__name__)


def warn_fingerprint_mismatch(label: str, saved: str, current: str) -> None:
    """Keep fingerprints as provenance, not a checkpoint loading restriction."""
    if saved != current:
        logger.warning(
            "%s fingerprint differs (saved=%s, current=%s); continuing with native "
            "tensor/key validation. Changed data is not an exact sample-order resume.",
            label, saved, current,
        )


@dataclass(frozen=True)
class SphericalCheckpointMetadata:
    """Metadata used to reject incompatible spherical artifacts and resumes."""

    checkpoint_type: str
    completed_step: int
    tp_size: int
    dp_size: int
    parameter_fingerprint: str
    data_manifest_fingerprint: str
    protocol_version: str
    pp_size: int = 1
    cp_size: int = 1
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_layers: int = 34
    num_attention_heads: int = 32
    num_query_groups: int = 8
    vocab_size: int = 131072
    block_size: int = 32
    # Old native v3 sidecars omitted this field.  They predate the selectable
    # HF-parity width and therefore explicitly default to the native 512 path.
    spherical_condition_dim: int = SPHERICAL_CONDITION_DIM
    spherical_condition_coordinate: str = "t"
    spherical_ada_activation: str = "identity"
    spherical_time_embedding_scale: float = 1.0
    spherical_condition_compute_dtype: str = "model"
    spherical_condition_norm: str = "none"
    tail_block_protocol: str = "original_length_v0"
    tail_eos_token_id: int | None = None
    spherical_qk_norm: str = "none"
    spherical_qk_norm_eps: float = 1e-6
    spherical_decode_training: bool = False
    spherical_decode_probability: float = 0.2
    spherical_decode_gamma_min: float = 0.85
    spherical_decode_gamma_max: float = 0.95

    @property
    def topology(self) -> dict[str, int]:
        return {"tp": self.tp_size, "dp": self.dp_size, "pp": self.pp_size, "cp": self.cp_size}

    def __post_init__(self) -> None:
        validate_decode_config(self)
        if self.spherical_qk_norm != "none":
            raise ValueError("this implementation cannot load active QK normalization checkpoints")
        if not math.isfinite(self.spherical_qk_norm_eps) or self.spherical_qk_norm_eps <= 0:
            raise ValueError("spherical_qk_norm_eps must be finite and positive")
        if not math.isfinite(self.spherical_time_embedding_scale) or self.spherical_time_embedding_scale <= 0:
            raise ValueError("spherical_time_embedding_scale must be finite and positive")
        if self.spherical_condition_compute_dtype not in {"model", "fp32"}:
            raise ValueError("spherical_condition_compute_dtype must be 'model' or 'fp32'")
        if self.spherical_condition_norm not in {"none", "layernorm"}:
            raise ValueError("spherical_condition_norm must be 'none' or 'layernorm'")
        if self.spherical_ada_activation not in {"identity", "silu"}:
            raise ValueError("spherical_ada_activation must be 'identity' or 'silu'")
        if self.spherical_condition_coordinate not in {"t", "gamma"}:
            raise ValueError("spherical_condition_coordinate must be 't' or 'gamma'")
        if self.checkpoint_type not in {"resume", "model_only"}:
            raise ValueError("checkpoint_type must be resume or model_only")
        if min(self.tp_size, self.dp_size, self.pp_size, self.cp_size) <= 0:
            raise ValueError(f"spherical topology sizes must be positive: {self.topology}")
        if self.protocol_version not in {LEGACY_SPHERICAL_PROTOCOL_VERSION, SPHERICAL_PROTOCOL_VERSION}:
            raise ValueError(
                "unsupported spherical protocol: "
                f"{self.protocol_version}; expected one of "
                f"({LEGACY_SPHERICAL_PROTOCOL_VERSION}, {SPHERICAL_PROTOCOL_VERSION})"
            )
        if self.spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
            raise ValueError(
                "spherical_condition_dim must be one of the supported native/HF widths "
                f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; got {self.spherical_condition_dim}"
            )


def metadata_dict(metadata: SphericalCheckpointMetadata) -> dict:
    """Omit optional fields for legacy artifacts so their sidecar schema stays unchanged."""
    values = asdict(metadata)
    # QK-off adds no tensors/operations; keep legacy sidecars unchanged.
    if metadata.spherical_qk_norm == "none":
        values.pop("spherical_qk_norm")
        values.pop("spherical_qk_norm_eps")
    if not metadata.spherical_decode_training:
        for key in DECODE_DEFAULTS:
            values.pop(key)
    return values


def parameter_fingerprint(model: torch.nn.Module | list[torch.nn.Module]) -> str:
    """Hash the parameter contract without copying multi-billion parameter values."""
    chunks = model if isinstance(model, list) else [model]
    digest = hashlib.sha256()
    for chunk_index, chunk in enumerate(chunks):
        for name, parameter in sorted(unwrap_model(chunk).named_parameters()):
            descriptor = (
                chunk_index,
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            digest.update(repr(descriptor).encode("utf-8"))
    return digest.hexdigest()


def _model_chunks(model: torch.nn.Module | list[torch.nn.Module]) -> list[torch.nn.Module]:
    chunks = model if isinstance(model, list) else [model]
    return [unwrap_model(chunk) for chunk in chunks]


def save_model_only_checkpoint(
    model: torch.nn.Module | list[torch.nn.Module],
    path: str | Path,
    metadata: SphericalCheckpointMetadata,
) -> None:
    """Collectively save a scalable MCore ``torch_dist`` model-only artifact."""
    if metadata.checkpoint_type != "model_only":
        raise ValueError("model-only artifacts require checkpoint_type=model_only")
    target = Path(path)
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            target.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
    else:
        target.mkdir(parents=True, exist_ok=True)

    state = _generate_model_state_dict(_model_chunks(model), ckpt_format="torch_dist")
    state["spherical_metadata"] = metadata_dict(metadata)
    dist_checkpointing.save(
        state,
        str(target),
        content_metadata={
            "checkpoint_type": "model_only",
            "protocol_version": metadata.protocol_version,
            "topology": metadata.topology,
        },
    )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        write_spherical_metadata(target, metadata)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_model_only_checkpoint(
    model: torch.nn.Module | list[torch.nn.Module], path: str | Path
) -> SphericalCheckpointMetadata:
    """Collectively load a model-only artifact with strict sharded-key checking."""
    target = str(path)
    common = dist_checkpointing.load_common_state_dict(target)
    raw_metadata = common.get("spherical_metadata")
    if not isinstance(raw_metadata, dict):
        raise ValueError("model-only checkpoint is missing spherical metadata")
    metadata = SphericalCheckpointMetadata(**raw_metadata)
    if metadata.checkpoint_type != "model_only":
        raise ValueError("checkpoint is not a model-only artifact")
    for chunk in _model_chunks(model):
        coordinate = getattr(getattr(chunk, "config", None), "spherical_condition_coordinate", "t")
        if coordinate != metadata.spherical_condition_coordinate:
            raise ValueError("model-only spherical_condition_coordinate mismatch")
        activation = getattr(getattr(chunk, "config", None), "spherical_ada_activation", "identity")
        if activation != metadata.spherical_ada_activation:
            raise ValueError("model-only spherical_ada_activation mismatch")
        config = getattr(chunk, "config", None)
        if getattr(config, "spherical_time_embedding_scale", 1.0) != metadata.spherical_time_embedding_scale:
            raise ValueError("model-only spherical_time_embedding_scale mismatch")
        if getattr(config, "spherical_condition_compute_dtype", "model") != metadata.spherical_condition_compute_dtype:
            raise ValueError("model-only spherical_condition_compute_dtype mismatch")
        if getattr(config, "spherical_condition_norm", "none") != metadata.spherical_condition_norm:
            raise ValueError("model-only spherical_condition_norm mismatch")
        transition = decode_transition(
            metadata, config, allow=os.environ.get("SPHERICAL_ALLOW_DECODE_TRANSITION") == "1"
        )
        if transition is not None:
            logger.warning("Explicit model-only decode protocol transition: %s", transition)


    content_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common)
    if content_metadata != {
        "checkpoint_type": "model_only",
        "protocol_version": metadata.protocol_version,
        "topology": metadata.topology,
    }:
        raise ValueError(f"invalid model-only content metadata: {content_metadata}")
    state = _generate_model_state_dict(_model_chunks(model), ckpt_format="torch_dist")
    loaded = dist_checkpointing.load(state, target, strict=StrictHandling.RAISE_ALL)
    chunks = _model_chunks(model)

    def load_chunk_state(chunk: torch.nn.Module, state_dict: dict) -> None:
        # Transformer Engine 1.13 serializes non-persistent FP8 metadata as a
        # zero-length tensor. Its ``set_extra_state`` tries to unpickle that
        # tensor under Torch 2.6, which raises EOFError. Keep the freshly
        # initialized local value for those keys while remaining strict for all
        # real model weights.
        omitted = {
            key
            for key, value in state_dict.items()
            if key.endswith("_extra_state")
            and isinstance(value, torch.Tensor)
            and value.numel() == 0
        }
        for key in omitted:
            state_dict.pop(key)
        if omitted:
            result = chunk.load_state_dict(state_dict, strict=False)
            missing = set(result.missing_keys)
            if missing != omitted or result.unexpected_keys:
                raise RuntimeError(
                    "model-only checkpoint state mismatch after omitting empty "
                    f"Transformer Engine metadata: missing={sorted(missing)} "
                    f"omitted={sorted(omitted)} unexpected={result.unexpected_keys}"
                )
        else:
            chunk.load_state_dict(state_dict, strict=True)

    if len(chunks) == 1:
        load_chunk_state(chunks[0], loaded["model"])
    else:
        for index, chunk in enumerate(chunks):
            load_chunk_state(chunk, loaded[f"model{index}"])
    for chunk in chunks:
        geometry = getattr(chunk, "spherical_geometry", None)
        if geometry is not None:
            geometry.refresh_codebook()
    return metadata


def validate_resume_metadata(
    metadata: SphericalCheckpointMetadata,
    *,
    tp_size: int,
    dp_size: int,
    data_manifest_fingerprint: str,
    protocol_version: str = SPHERICAL_PROTOCOL_VERSION,
    spherical_condition_dim: int | None = None,
    spherical_condition_coordinate: str = "t",
    spherical_ada_activation: str = "identity",
    spherical_time_embedding_scale: float = 1.0,
    spherical_condition_compute_dtype: str = "model",
    spherical_condition_norm: str = "none",
    allow_dp_resize: bool = False,
) -> None:
    """Reject incompatible topology, but treat data identity as provenance only."""
    if metadata.checkpoint_type != "resume":
        raise ValueError("resume requires checkpoint_type=resume")
    if metadata.spherical_condition_coordinate != spherical_condition_coordinate:
        raise ValueError("resume spherical_condition_coordinate mismatch")
    if metadata.spherical_ada_activation != spherical_ada_activation:
        raise ValueError("resume spherical_ada_activation mismatch")
    if metadata.spherical_time_embedding_scale != spherical_time_embedding_scale:
        raise ValueError("resume spherical_time_embedding_scale mismatch")
    if metadata.spherical_condition_compute_dtype != spherical_condition_compute_dtype:
        raise ValueError("resume spherical_condition_compute_dtype mismatch")
    if metadata.spherical_condition_norm != spherical_condition_norm:
        raise ValueError("resume spherical_condition_norm mismatch")
    expected = (tp_size, dp_size, 1, 1)
    actual = (
        metadata.tp_size,
        metadata.dp_size,
        metadata.pp_size,
        metadata.cp_size,
    )
    topology_matches = actual == expected
    dp_only_resize = (actual[0], actual[2], actual[3]) == (expected[0], expected[2], expected[3])
    if not topology_matches and not (allow_dp_resize and dp_only_resize):
        raise ValueError(f"resume metadata mismatch: expected={expected}, actual={actual}")
    if not topology_matches:
        logger.warning("Explicit DP resize %d -> %d; optimizer shards are loaded by MCore; RNG assignments change.",
                       metadata.dp_size, dp_size)
    warn_fingerprint_mismatch("resume data", metadata.data_manifest_fingerprint, data_manifest_fingerprint)
    if metadata.protocol_version not in {LEGACY_SPHERICAL_PROTOCOL_VERSION, protocol_version}:
        raise ValueError(
            "resume protocol version is incompatible with the current sampler: "
            f"expected={protocol_version}, actual={metadata.protocol_version}"
        )
    if metadata.spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "resume spherical_condition_dim must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}"
        )
    if spherical_condition_dim is not None and spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "requested spherical_condition_dim must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}"
        )
    if spherical_condition_dim is not None and metadata.spherical_condition_dim != spherical_condition_dim:
        raise ValueError(
            "resume spherical_condition_dim mismatch: "
            f"expected={spherical_condition_dim}, actual={metadata.spherical_condition_dim}"
        )


def load_spherical_metadata(path: str | Path) -> SphericalCheckpointMetadata:
    """Load the protocol sidecar shared by resume and model-only artifacts."""
    metadata_path = Path(path) / "spherical_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"spherical checkpoint is missing {metadata_path}")
    raw_metadata = json.loads(metadata_path.read_text())
    if "protocol_version" not in raw_metadata:
        raise ValueError("spherical checkpoint metadata is missing protocol_version")
    if "spherical_condition_dim" not in raw_metadata:
        # Native v3 sidecars written before the selectable HF-parity width are
        # unambiguously 512 because that was the native provider default.
        raw_metadata = {**raw_metadata, "spherical_condition_dim": SPHERICAL_CONDITION_DIM}
    return SphericalCheckpointMetadata(**raw_metadata)


def write_spherical_metadata(path: str | Path, metadata: SphericalCheckpointMetadata) -> None:
    """Atomically write protocol metadata after a native checkpoint is complete."""
    target = Path(path) / "spherical_metadata.json"
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(json.dumps(metadata_dict(metadata), sort_keys=True, indent=2) + "\n")
    temporary.replace(target)


__all__ = [
    "SphericalCheckpointMetadata",
    "load_model_only_checkpoint",
    "load_spherical_metadata",
    "parameter_fingerprint",
    "save_model_only_checkpoint",
    "validate_resume_metadata",
    "warn_fingerprint_mismatch",
    "write_spherical_metadata",
]
