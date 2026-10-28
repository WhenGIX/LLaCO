# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Map-style stem Parquet source for the spherical Nemotron compatibility path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(map(int, value))


class StemParquetDataset(Dataset):
    """Map-style, row-group backed assistant-turn dataset.

    Only row-group/row/turn coordinates are indexed in memory.  Messages are
    tokenized when a sample is fetched, so no token shard is required.  The
    coordinates are stable under ``sorted(stem-*.parquet)`` and therefore work
    with Bridge's deterministic distributed sampler and consumed-sample resume.
    """

    def __init__(self, root: str | Path, tokenizer: Any, *, max_length: int, block_size: int = 32):
        import pyarrow.parquet as pq

        if block_size != 32:
            raise ValueError("spherical stem provider requires block_size=32")
        self.root = Path(root)
        self.files = [self.root] if self.root.is_file() else sorted(self.root.glob("stem-*.parquet"))
        if not self.files or any(path.suffix != ".parquet" for path in self.files):
            raise ValueError("provider requires one or more stem-*.parquet files")
        if any(not path.name.startswith("stem-") for path in self.files):
            raise ValueError("provider is restricted to the stem split")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.block_size = block_size
        self._parquet = [pq.ParquetFile(path) for path in self.files]
        self._row_groups: list[tuple[int, int, int, int]] = []
        self._eligible: list[tuple[int, int, int, int]] = []
        self._cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for file_index, parquet in enumerate(self._parquet):
            for group_index in range(parquet.num_row_groups):
                rows = parquet.read_row_group(group_index, columns=["messages"]).to_pylist()
                for row_index, row in enumerate(rows):
                    for turn_index, message in enumerate(row.get("messages", ())):
                        if message.get("role") == "assistant" and message.get("content"):
                            self._row_groups.append((file_index, group_index, row_index, turn_index))
        if not self._row_groups:
            raise ValueError("stem parquet contains no assistant turns")
        # Validate complete targets during indexing so the distributed sampler
        # never receives an index that later raises IndexError.  This is also
        # the deterministic drop point used by resume fingerprints.
        for coordinate in self._row_groups:
            try:
                self._build_sample(coordinate)
            except IndexError:
                continue
            self._eligible.append(coordinate)
        self._row_groups = self._eligible
        if not self._row_groups:
            raise ValueError("no stem assistant turns fit the configured spherical sequence length")
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        payload = {"files": []}
        for path in self.files:
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    file_digest.update(chunk)
            file_hash = file_digest.hexdigest()
            payload["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash})
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()

    def __len__(self) -> int:
        return len(self._row_groups)

    def _row(self, file_index: int, group_index: int, row_index: int) -> dict[str, Any]:
        key = (file_index, group_index)
        rows = self._cache.get(key)
        if rows is None:
            rows = self._parquet[file_index].read_row_group(group_index).to_pylist()
            self._cache[key] = rows
        return rows[row_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._build_sample(self._row_groups[index])

    def _build_sample(self, coordinate: tuple[int, int, int, int]) -> dict[str, Any]:
        file_index, group_index, row_index, turn_index = coordinate
        row = self._row(file_index, group_index, row_index)
        messages = row["messages"]
        context = messages[:turn_index]
        target_message = messages[turn_index]
        prefix = _token_ids(self.tokenizer.apply_chat_template(context, tokenize=True, add_generation_prompt=True))
        complete = _token_ids(self.tokenizer.apply_chat_template(
            context + [target_message], tokenize=True, add_generation_prompt=False
        ))
        if complete[: len(prefix)] != prefix:
            raise ValueError("chat template completion does not preserve its generation prefix")
        target = complete[len(prefix) :]
        padded_width = ((len(target) + self.block_size - 1) // self.block_size) * self.block_size
        prompt_cap = self.max_length - 2 * padded_width
        if prompt_cap < 1:
            raise IndexError("complete target does not fit spherical layout")
        prefix = prefix[-prompt_cap:]
        input_ids = prefix + target + [int(self.tokenizer.pad_token_id)] * (padded_width - len(target))
        start = len(prefix)
        return {
            "input_ids": input_ids,
            "assistant_loss_mask": [0] * start + [1] * len(target) + [0] * (padded_width - len(target)),
            "target_start": start,
            "target_valid_end": start + len(target),
            "target_end": len(input_ids),
            "conversation_id": str(row.get("uuid", f"{self.files[file_index].name}:{row_index}")),
            "turn": turn_index,
            "source_shard": self.files[file_index].name,
            "source_row": row_index,
        }


__all__ = ["StemParquetDataset"]
