# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RNG-preserving helpers for collective fixed-panel inference."""

from __future__ import annotations

import random
from contextlib import contextmanager
from copy import deepcopy

import torch


def _state_equal(left: object, right: object) -> bool:
    """Compare nested scheduler/sampler state without tensor truth coercion."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_state_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def capture_training_state(*, scheduler=None, sampler=None, optimizer=None) -> dict[str, object]:
    """Capture RNG and small cursors without cloning an 8B optimizer state."""
    state: dict[str, object] = {
        "python_rng": random.getstate(),
        "torch_rng": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    for name, value in (("scheduler", scheduler), ("sampler", sampler)):
        if value is not None and hasattr(value, "state_dict"):
            state[name] = deepcopy(value.state_dict())
            state[f"{name}_identity"] = id(value)
    # The panel never calls optimizer or sampler methods. Retain object identity as
    # a guard, rather than copying multi-billion-parameter optimizer tensors.
    state["optimizer_identity"] = id(optimizer) if optimizer is not None else None
    return state


def restore_training_state(state: dict[str, object], *, scheduler=None, sampler=None, optimizer=None) -> None:
    """Restore the small training state captured around panel inference."""
    random.setstate(state["python_rng"])
    torch.random.set_rng_state(state["torch_rng"])
    if "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    for name, value in (("scheduler", scheduler), ("sampler", sampler)):
        if value is not None and name in state:
            if id(value) != state[f"{name}_identity"]:
                raise RuntimeError(f"fixed panel replaced the training {name}")
            if not _state_equal(value.state_dict(), state[name]):
                raise RuntimeError(f"fixed panel mutated the training {name}")
    if state["optimizer_identity"] is not None and id(optimizer) != state["optimizer_identity"]:
        raise RuntimeError("fixed panel replaced the training optimizer")


@contextmanager
def preserved_training_state(*, scheduler=None, sampler=None, optimizer=None):
    """Run evaluation without advancing training-owned state."""
    snapshot = capture_training_state(scheduler=scheduler, sampler=sampler, optimizer=optimizer)
    try:
        yield
    finally:
        restore_training_state(snapshot, scheduler=scheduler, sampler=sampler, optimizer=optimizer)


__all__ = ["capture_training_state", "preserved_training_state", "restore_training_state"]
