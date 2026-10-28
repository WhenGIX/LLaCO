# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Prepared-only data provider for spherical Nemotron training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from .spherical_mixture import TAIL_BLOCK_PROTOCOL, PreparedMixtureDataset, load_manifest
from ..nemotron_labs_diffusion_spherical_protocol import SPHERICAL_PROTOCOL_VERSION


@dataclass(kw_only=True)
class PreparedDatasetConfig(DatasetProvider):
    """Train from an offline-tokenized manifest; padding happens in the collator."""

    prepared_manifest: str
    seq_length: int = 2688
    target_max_tokens: int = 768
    block_size: int = 8
    protocol_version: str = SPHERICAL_PROTOCOL_VERSION
    tail_block_protocol: str = TAIL_BLOCK_PROTOCOL
    eos_token_id: int | None = None
    manifest_fingerprint: str | None = None
    dataloader_type: str = "single"
    train_samples: int = 0
    valid_samples: int = 0
    test_samples: int = 0
    mixture_seed: int = 20260925
    mixture_cycle_size: int = 1000
    length_bucket_group: int = 64
    length_ordering: str = "legacy"
    length_ordering_start: int = 0
    length_ordering_window: int = 4000
    length_ordering_microbatch: int = 4
    length_ordering_global_batch: int = 32
    dynamic_padding: bool = True

    def __post_init__(self) -> None:
        if not self.prepared_manifest:
            raise ValueError("prepared_manifest is required; run scripts/prepare_spherical_data.py first")
        if self.tail_block_protocol != TAIL_BLOCK_PROTOCOL or self.protocol_version != SPHERICAL_PROTOCOL_VERSION:
            raise ValueError("Unsupported spherical data protocol")
        if self.block_size <= 0 or self.target_max_tokens <= 0 or self.target_max_tokens % self.block_size:
            raise ValueError("target_max_tokens must be a positive multiple of block_size")
        if self.seq_length <= 2 * self.target_max_tokens:
            raise ValueError("seq_length must leave room for context and two targets")
        if self.dataloader_type != "single":
            raise ValueError("Prepared mixture owns shuffling and requires dataloader_type=single")
        if self.mixture_cycle_size <= 0 or self.length_bucket_group <= 0:
            raise ValueError("Mixture cycle and bucket group must be positive")
        limits = load_manifest(self.prepared_manifest)["selection"]["filters"]
        if (self.target_max_tokens != limits["target_max_tokens"]
                or self.seq_length != limits["context_max_tokens"] + 2 * limits["target_max_tokens"]):
            raise ValueError("Dataset lengths must match the prepared manifest")

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        self.__post_init__()
        if context.tokenizer is None:
            raise ValueError("Bridge tokenizer is required for prepared training")
        # Keep the absolute sample budget valid after a resumed batch resize.
        dataset = PreparedMixtureDataset(self, context.tokenizer, max(context.train_samples, self.train_samples))
        return dataset, None, None


__all__ = ["PreparedDatasetConfig"]
