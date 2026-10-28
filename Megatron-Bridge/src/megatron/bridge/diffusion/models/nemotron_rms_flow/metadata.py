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

"""Checkpoint metadata and strict compatibility checks for RMS flow runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .scheduler import IdentityScheduler
from .statistics import file_sha256


def stable_json_sha256(value) -> str:
    """Hash JSON-compatible metadata using a canonical representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def tokenizer_files_sha256(model_path: str | Path) -> str:
    """Hash the tokenizer artifacts used by training and inference metadata."""
    digest = hashlib.sha256()
    root = Path(model_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = root / name
        if path.exists():
            digest.update(name.encode())
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    """Load a tensor/container checkpoint without enabling arbitrary pickle globals."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("RMS checkpoint must contain a dictionary")
    return state


def build_metadata(
    *,
    model: str,
    model_config_hash: str,
    embedding_hash: str,
    tokenizer_hash: str,
    chat_template_hash: str,
    geometry,
    statistics_manifest,
    data_manifest_hash: str,
    data_manifest: dict,
    block_size=32,
    condition_dim=1024,
    uniform_gamma_fraction=0.3,
    data_seed=1234,
    batch_size=1,
    grad_accum=1,
    max_length=1024,
    shuffle_buffer=8192,
    posterior_mode="topk",
    posterior_k=512,
    nfe=64,
    t_min=0.01,
    sample_every=100,
    sample_prompts=(),
    terminal_seeds=(),
    sample_max_blocks=4,
    optimizer_groups=None,
):
    """Build the compatibility metadata persisted beside a model state."""
    scheduler = (
        statistics_manifest.scheduler
        if statistics_manifest
        else IdentityScheduler.for_terminal(geometry.terminal_mode)
    )
    excluded = list(statistics_manifest.excluded_ids) if statistics_manifest else []
    rms_excluded = list(statistics_manifest.rms_excluded_ids) if statistics_manifest else []
    prediction_ids_hash = stable_json_sha256(
        [i for i in range(geometry.vocab_size) if i != geometry.mask_token_id and i not in excluded]
    )
    metadata = geometry.metadata()
    metadata.update(
        {
            "version": 3,
            "model": model,
            "model_config_sha256": model_config_hash,
            "input_embedding_sha256": embedding_hash,
            "tokenizer_sha256": tokenizer_hash,
            "chat_template_sha256": chat_template_hash,
            "V_rms": {
                "definition": "all finite embedding rows minus manifest exclusions",
                "excluded_ids": rms_excluded,
            },
            "V_id": {"definition": "vocabulary minus mask and statistics exclusions", "sha256": prediction_ids_hash},
            "V_pred": {"definition": "vocabulary minus mask and statistics exclusions", "sha256": prediction_ids_hash},
            "block_size": int(block_size),
            "condition_dim": int(condition_dim),
            "gamma_encoding_version": "sinusoidal-256-mlp-1024-ada6-v1",
            "scheduler": {**scheduler.metadata(), "t_min": float(t_min), "nodes": "linspace(1,t_min,NFE)"},
            "training_sampling": {
                "unit": "noisy_block",
                "uniform_t_fraction": 1.0 - float(uniform_gamma_fraction),
                "uniform_gamma_fraction": float(uniform_gamma_fraction),
            },
            "input_embedding_frozen": True,
            "parallelism": {
                "tensor_parallelism": 1,
                "pipeline_parallelism": 1,
                "use_megatron_fsdp": False,
                "use_distributed_optimizer": True,
            },
            "optimizer_state_precision": "zero_sharded_fp32_master_and_moments",
            "optimizer_resume_migration": "fresh_fp32_state_from_model_for_legacy_bf16_shards",
            "posterior_mode": posterior_mode,
            "posterior_k": int(posterior_k),
            "inference_nfe": int(nfe),
            "sample_protocol": {
                "sample_every": int(sample_every),
                "prompt_set_sha256": stable_json_sha256(list(sample_prompts)),
                "terminal_seeds": list(map(int, terminal_seeds)),
                "block_size": int(block_size),
                "max_blocks": int(sample_max_blocks),
                "nfe": int(nfe),
                "t_min": float(t_min),
            },
            "data_manifest_sha256": data_manifest_hash,
            "data_conversion_version": data_manifest.get("conversion_version", data_manifest.get("version")),
            "training_data_loader": {
                "seed": int(data_seed),
                "batch_size_per_rank": int(batch_size),
                "gradient_accumulation_steps": int(grad_accum),
                "max_length": int(max_length),
                "shuffle_buffer": int(shuffle_buffer),
            },
            "data_split": {
                "algorithm": data_manifest.get("split_algorithm"),
                "seed": data_manifest.get("seed"),
                "validation_ratio": data_manifest.get("val_ratio"),
            },
            "data_partial": bool(
                data_manifest.get("partial") or data_manifest.get("counts", {}).get("partial_conversion")
            ),
            "optimizer_groups": optimizer_groups or [],
        }
    )
    if statistics_manifest is not None:
        metadata.update(statistics_manifest.metadata())
    return metadata


def validate_metadata(metadata: dict, expected: dict, *, inference_only=False):
    """Reject any semantic mismatch between a checkpoint and the current run."""
    exact_keys = (
        "version",
        "geometry_type",
        "mask_token_id",
        "vocab_size",
        "terminal_mode",
        "model_config_sha256",
        "input_embedding_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "V_rms",
        "V_id",
        "V_pred",
        "block_size",
        "condition_dim",
        "gamma_encoding_version",
        "scheduler",
        "input_embedding_frozen",
        "parallelism",
        "optimizer_state_precision",
        "optimizer_resume_migration",
        "posterior_mode",
        "posterior_k",
        "inference_nfe",
        "statistics_manifest_sha256",
        "statistics_fit_sha256",
    )
    if not inference_only:
        exact_keys += (
            "training_sampling",
            "sample_protocol",
            "data_manifest_sha256",
            "data_conversion_version",
            "data_split",
            "optimizer_groups",
        )
    for key in exact_keys:
        if key == "optimizer_state_precision" and metadata.get(key) is None:
            # Version-3 checkpoints from the initial BF16 optimizer can be
            # migrated in-place by the FP32-master optimizer on first step.
            continue
        if key == "optimizer_resume_migration" and metadata.get(key) is None:
            continue
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"checkpoint metadata mismatch for {key}")
    if not inference_only:
        saved_loader = metadata.get("training_data_loader", {})
        expected_loader = expected.get("training_data_loader", {})
        for key in ("seed", "max_length", "shuffle_buffer"):
            if saved_loader.get(key) != expected_loader.get(key):
                raise ValueError(f"checkpoint metadata mismatch for training_data_loader.{key}")
        saved_records = int(saved_loader.get("batch_size_per_rank", 0)) * int(
            saved_loader.get("gradient_accumulation_steps", 0)
        )
        expected_records = int(expected_loader.get("batch_size_per_rank", 0)) * int(
            expected_loader.get("gradient_accumulation_steps", 0)
        )
        if saved_records <= 0 or saved_records != expected_records:
            raise ValueError("checkpoint metadata mismatch for effective records per optimizer step")
    for key in ("R", "r_mask", "b_noise"):
        if not math.isclose(
            float(metadata.get(key, float("nan"))),
            float(expected.get(key, float("nan"))),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(f"checkpoint metadata mismatch for {key}")


__all__ = ["build_metadata", "load_checkpoint", "stable_json_sha256", "tokenizer_files_sha256", "validate_metadata"]
