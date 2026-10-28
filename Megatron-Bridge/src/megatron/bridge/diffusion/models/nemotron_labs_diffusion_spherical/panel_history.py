# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Durable, run-scoped inference tables across training process restarts."""

import fcntl
import json
import os
import tempfile
from pathlib import Path

PANEL_COLUMNS = ("step", "prompt", "standard_answer", "result")


def merge_panel_rows(*groups: list[dict]) -> list[dict]:
    """Merge cumulative snapshots without discarding conflicting generations."""
    merged = {}
    for group in groups:
        for row in group:
            if set(row) != set(PANEL_COLUMNS) or not isinstance(row["step"], int):
                raise ValueError("Invalid inference history row")
            key = (row["step"], row["prompt"])
            if key in merged and merged[key] != row:
                raise ValueError(f"Conflicting inference results at step {row['step']}")
            merged[key] = dict(row)
    return sorted(merged.values(), key=lambda row: row["step"])


def persist_panel_rows(directory: str | Path, run_path: str, rows: list[dict]) -> list[dict]:
    """Atomically merge rows into an entity/project/run-specific local journal."""
    parts = run_path.split("/")
    if len(parts) != 3 or any(not part or part in (".", "..") for part in parts):
        raise ValueError("Expected entity/project/run_id for inference history")
    target = Path(directory).joinpath(*parts).with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"run_path": run_path, "rows": []}
        if previous["run_path"] != run_path:
            raise ValueError("Inference history belongs to another W&B run")
        merged = merge_panel_rows(previous["rows"], rows)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"run_path": run_path, "rows": merged}, stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return merged
