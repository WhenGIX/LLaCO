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

"""Sharded FP32-master optimizer support for the HF RMS pilot."""

from __future__ import annotations

import copy

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


class _FP32MasterAdamW(torch.optim.Optimizer):
    """AdamW whose ZeRO-local BF16 parameters have FP32 masters and moments."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def load_state_dict(self, state_dict):
        """Restore FP32 state without PyTorch casting it to BF16 param dtype."""
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("optimizer checkpoint has a different number of parameter groups")
        saved_ids = []
        current_params = []
        for current_group, saved_group in zip(self.param_groups, saved_groups):
            if len(current_group["params"]) != len(saved_group.get("params", [])):
                raise ValueError("optimizer checkpoint parameter-group size mismatch")
            current_params.extend(current_group["params"])
            saved_ids.extend(saved_group["params"])
            for key, value in saved_group.items():
                if key != "params":
                    current_group[key] = copy.deepcopy(value)

        saved_state = state_dict.get("state", {})
        self.state.clear()
        fp32_keys = {"step", "master_param", "exp_avg", "exp_avg_sq"}
        for parameter, saved_id in zip(current_params, saved_ids):
            values = saved_state.get(saved_id)
            if not values:
                continue
            restored = {}
            for key, value in values.items():
                if torch.is_tensor(value):
                    dtype = torch.float32 if key in fp32_keys else value.dtype
                    restored[key] = value.to(device=parameter.device, dtype=dtype)
                else:
                    restored[key] = copy.deepcopy(value)
            self.state[parameter] = restored

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("FP32 master AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                    state["master_param"] = parameter.detach().float().clone()
                    state["exp_avg"] = torch.zeros_like(state["master_param"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master_param"])
                else:
                    # Checkpoints produced by the former native BF16 AdamW do
                    # not contain a master. Preserve their moments and begin
                    # FP32 accumulation from the checkpoint model value.
                    state.setdefault("master_param", parameter.detach().float().clone())
                    state["master_param"] = state["master_param"].float()
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()
                    state["step"] = torch.as_tensor(state["step"], dtype=torch.float32, device=parameter.device)
                master = state["master_param"]
                grad = parameter.grad.detach().float()
                state["step"].add_(1)
                step = float(state["step"])
                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                state["exp_avg"].lerp_(grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = state["exp_avg_sq"].sqrt().div_(bias_correction2**0.5).add_(eps)
                master.addcdiv_(state["exp_avg"], denom, value=-lr / bias_correction1)
                parameter.copy_(master.to(dtype=parameter.dtype))
        return loss


class ShardedFP32AdamW:
    """Thin, checkpoint-friendly wrapper around native ZeRO AdamW.

    ``ZeroRedundancyOptimizer`` owns parameter partitioning and synchronization.
    Its local optimizer owns FP32 masters and FP32 moments only for this rank's
    shard, so neither precision nor memory is silently changed by BF16 model
    parameters.
    """

    def __init__(self, param_groups, *, process_group=None):
        self.zero = ZeroRedundancyOptimizer(
            param_groups, optimizer_class=_FP32MasterAdamW, process_group=process_group
        )

    @property
    def optim(self):
        """Expose the local optimizer for backward-compatible checkpoint I/O."""
        return self.zero.optim

    @property
    def param_groups(self):
        # The local optimizer owns the groups consumed by ``step``.  PyTorch's
        # ZeRO wrapper keeps a separate global-group list; exposing that list
        # makes scheduler updates silently ineffective on the rank-local AdamW.
        return self.zero.optim.param_groups

    def sync_master_params_from_model(self):
        """Materialize/migrate local FP32 state before activation memory is allocated."""
        for group in self.optim.param_groups:
            for parameter in group["params"]:
                state = self.optim.state.get(parameter)
                if not state:
                    continue
                state["master_param"] = parameter.detach().float().clone()
                state["exp_avg"] = state["exp_avg"].float()
                state["exp_avg_sq"] = state["exp_avg_sq"].float()
                state["step"] = torch.as_tensor(state["step"], dtype=torch.float32, device=parameter.device)

    def load_resume_state(self, state_dict, *, migrate_from_bf16=False):
        """Restore ZeRO state without a GPU-wide BF16-to-FP32 peak.

        Old pilot shards contain native BF16 AdamW moments.  Recreating those
        moments in FP32 is both numerically ambiguous and too memory-heavy at
        Nemotron scale, so migration starts fresh FP32 moments while retaining
        the optimizer step counter.
        """
        if not migrate_from_bf16:
            self.optim.load_state_dict(state_dict)
            return
        saved_groups = state_dict.get("param_groups", [])
        for current, saved in zip(self.optim.param_groups, saved_groups):
            if "step" in saved:
                current["_resume_step"] = int(saved["step"])
        for group in self.optim.param_groups:
            for parameter in group["params"]:
                state = self.optim.state.get(parameter)
                if state:
                    state.clear()
        self._migrated_from_bf16 = True

    def step(self):
        """Run ZeRO's native sharded AdamW update and parameter sync."""
        return self.zero.step()

    def zero_grad(self, set_to_none: bool = True):
        self.zero.zero_grad(set_to_none=set_to_none)


class _BF16AdamW(torch.optim.AdamW):
    """Native BF16-state AdamW with Nemotron's Bridge beta values.

    The official Bridge Nemotron recipe uses Megatron Core's distributed Adam
    and sets ``adam_beta2=0.95``.  This native optimizer is only the compatible
    implementation for the spherical trainer's HF/DDP loop; it is not the
    official Megatron optimizer.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2, **kwargs):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )


class ShardedBF16AdamW:
    """Local HF/DDP compatibility optimizer with BF16 parameters and states.

    ``ZeroRedundancyOptimizer`` provides parameter sharding and synchronization,
    while native AdamW keeps moving averages in the BF16 parameter dtype.  This
    is intentionally an experiment requested by the spherical trainer and is
    not a drop-in replacement for Bridge's Megatron Core optimizer.
    """

    def __init__(self, param_groups, *, process_group=None):
        self.zero = ZeroRedundancyOptimizer(
            param_groups, optimizer_class=_BF16AdamW, process_group=process_group
        )

    @property
    def optim(self):
        return self.zero.optim

    @property
    def param_groups(self):
        # Keep scheduler mutations on the optimizer that actually performs the
        # local update (ZeRO's global groups are a distinct list).
        return self.zero.optim.param_groups

    def load_resume_state(self, state_dict, *, migrate_from_bf16=False):
        # Native AdamW already restores floating state in the parameter dtype.
        self.optim.load_state_dict(state_dict)

    def step(self):
        return self.zero.step()

    def zero_grad(self, set_to_none: bool = True):
        self.zero.zero_grad(set_to_none=set_to_none)


__all__ = ["ShardedFP32AdamW", "ShardedBF16AdamW"]
