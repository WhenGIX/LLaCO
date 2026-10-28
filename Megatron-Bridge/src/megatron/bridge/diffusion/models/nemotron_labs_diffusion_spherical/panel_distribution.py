# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Assign questions to model replicas and merge their results exactly once."""

import torch.distributed as dist


def panel_indices(count: int, *, replica_rank: int, replica_count: int) -> range:
    """Return the stable question indices owned by one data-parallel replica."""
    if count < 0 or replica_count < 1 or not 0 <= replica_rank < replica_count:
        raise ValueError("invalid inference panel replica assignment")
    return range(replica_rank, count, replica_count)


def merge_panel_payloads(payloads: list[dict], *, count: int) -> tuple[list[dict], list[dict], float]:
    """Validate collective success and coverage, then restore question order."""
    errors = [f"rank {rank}: {payload['error']}" for rank, payload in enumerate(payloads) if payload["error"]]
    if errors:
        raise RuntimeError("inference panel failed: " + "; ".join(errors))
    records = [record for payload in payloads for record in payload["records"]]
    records.sort(key=lambda record: record["index"])
    if [record["index"] for record in records] != list(range(count)):
        raise RuntimeError("inference panel has missing or duplicate question results")
    return (
        [record["row"] for record in records],
        [record["detail"] for record in records],
        max(payload["elapsed"] for payload in payloads),
    )


def gather_panel_results(payload: dict, *, count: int) -> tuple[list[dict], list[dict], float]:
    """Collect one contribution per TP leader while every rank participates."""
    payloads = [payload]
    if dist.is_initialized():
        payloads = [None] * dist.get_world_size()
        dist.all_gather_object(payloads, payload)
    return merge_panel_payloads(payloads, count=count)
