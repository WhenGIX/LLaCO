# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Distributed acceptance, monitoring, panel, and model-only checkpoint hooks."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.utils.checkpoint_utils import is_checkpoint_iteration_directory

from .checkpoint import (
    SphericalCheckpointMetadata,
    load_model_only_checkpoint,
    load_spherical_metadata,
    parameter_fingerprint,
    save_model_only_checkpoint,
    warn_fingerprint_mismatch,
    write_spherical_metadata,
)
from .conditioning import conditioning_input
from .fixed_panel import FIXED_PANEL
from .decode_protocol import decode_fields, resolve_decode_endpoint
from .inference import spherical_block_sample
from .panel import preserved_training_state
from .panel_distribution import gather_panel_results, panel_indices
from .panel_history import persist_panel_rows
from .scheduler import gamma_from_t


# A run-scoped marker permits append-only historical backfills without reusing
# W&B's monotonically increasing event counter as the physical training step.
DECODE_HISTORY_CONFIG_KEY = "decode_metric_history_v1"
DECODE_HISTORY_STEP_METRIC = "progress/cumulative_step"


def decode_wandb_metrics(loss_dict: Mapping | None, *, enabled: bool) -> dict[str, float]:
    """Select already-reduced decoder diagnostics only for the opt-in objective."""
    if not enabled or not loss_dict:
        return {}
    return {
        key: float(value)
        for key, value in loss_dict.items()
        if key.startswith(("decode/", "denoise/"))
        or key in {"sampling/decode_block_fraction", "sampling/decode_token_fraction"}
    }


logger = logging.getLogger(__name__)


def _set_uniform_group_lr(optimizer, scheduler) -> None:
    """Override restored group LR limits without resetting moments or schedule progress."""
    for group in optimizer.param_groups:
        group["max_lr"] = scheduler.max_lr
        group["min_lr"] = scheduler.min_lr
    scheduler.step(0)


def _warmup_optimizer_collectives(optimizer, device: torch.device) -> None:
    """Allocate gradient-statistics NCCL communicators before activation peaks."""
    if not torch.distributed.is_initialized():
        return
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    groups = set()
    for inner in optimizers:
        getter = getattr(inner, "get_grad_stats_parallel_group", None)
        if getter is None:
            continue
        group = getter()
        if id(group) not in groups:
            torch.distributed.all_reduce(torch.zeros(1, device=device), group=group)
            groups.add(id(group))


def _core(model):
    model = model[0] if isinstance(model, list) else model
    core = unwrap_model(model)
    return getattr(core, "language_model", core)


def _global_max(value: torch.Tensor) -> float:
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return float(value.item())


def _replicated_parameters(core) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    adapter = list(core.spherical_adapter.parameters())
    ada = list(core.time_conditioner.parameters())
    for layer in core.decoder.layers:
        ada.extend(layer.spherical_ada_projection.parameters())
    return adapter, ada


def _replicated_tp_max_diff(core) -> float:
    group = parallel_state.get_tensor_model_parallel_group()
    src = parallel_state.get_tensor_model_parallel_src_rank()
    maximum = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    for parameter in sum(_replicated_parameters(core), []):
        if not getattr(parameter, "average_gradients_across_tp_domain", False):
            raise RuntimeError("replicated spherical parameter is not registered with MCore TP grad finalization")
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=src, group=group)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).float().abs().max())
    return _global_max(maximum)


def _parameter_checksum(parameters) -> tuple[float, float]:
    """Return a deterministic update sentinel without materializing full fp64 copies."""
    total = torch.zeros(2, device=torch.cuda.current_device(), dtype=torch.float64)
    for parameter in parameters:
        flat = parameter.detach().reshape(-1)
        stride = max(1, flat.numel() // 256)
        value = flat[::stride][:256].double()
        total[0] += value.sum()
        total[1] += value.square().sum()
    return float(total[0]), float(total[1])


def _grad_norm(parameters) -> torch.Tensor:
    total = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    for parameter in parameters:
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is not None:
            total += gradient.detach().float().square().sum()
    total.sqrt_()
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.MAX)
    return total


def _ada_grad_details(core) -> tuple[float, float]:
    conditioner = float(_grad_norm(core.time_conditioner.parameters()))
    layer_norms = torch.stack(
        [_grad_norm(layer.spherical_ada_projection.parameters()) for layer in core.decoder.layers]
    )
    return conditioner, float(layer_norms.min())


def _panel_input_ids(encoded, device: torch.device) -> torch.Tensor:
    """Normalize tokenizer tensor, list, and BatchEncoding return conventions."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("chat template result is missing input_ids")
        encoded = encoded["input_ids"]
    ids = encoded if isinstance(encoded, torch.Tensor) else torch.as_tensor(encoded, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError(f"chat template input_ids must be rank 2, got {tuple(ids.shape)}")
    return ids.to(device)


def _panel_special_ids(tokenizer) -> tuple[int, int]:
    """Read special ids from either MCore or HuggingFace tokenizer interfaces."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eod", None)
    if eos_id is None:
        raise ValueError("panel tokenizer does not define an EOS/EOD token")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "pad", None)
    return int(eos_id), int(eos_id if pad_id is None else pad_id)


def _panel_decode(tokenizer, token_ids: list[int]) -> str:
    """Decode through the explicit MCore or HuggingFace tokenizer contract."""
    if hasattr(tokenizer, "detokenize"):
        return tokenizer.detokenize(token_ids, skip_special_tokens=True)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    raise ValueError("panel tokenizer does not provide detokenize/decode")


@torch.no_grad()
def _full_clean_parity(core) -> tuple[float, float]:
    """Compare both embedding and full local logits for native vs E_geometry input."""
    if core.config.hidden_dropout != 0 or getattr(core.config, "use_mup", False):
        raise RuntimeError("exact clean parity requires hidden_dropout=0 and use_mup=False")
    device = core.spherical_geometry.E_geometry.device
    ids = torch.tensor([[1, 7, 11, int(core.config.mask_token_id)]], device=device)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    mask = torch.full(
        (1, 1, ids.shape[1], ids.shape[1]),
        torch.finfo(core.spherical_geometry.E_geometry.dtype).min,
        device=device,
        dtype=core.spherical_geometry.E_geometry.dtype,
    )
    mask.masked_fill_(torch.ones(ids.shape[1], ids.shape[1], device=device, dtype=torch.bool).tril(), 0)
    condition = core.time_conditioner(torch.zeros_like(ids, dtype=core.spherical_geometry.E_geometry.dtype))
    was_training = core.training
    core.eval()
    native_embedding = core.embedding(input_ids=ids, position_ids=positions)
    geometry_embedding = core.spherical_geometry.clean(ids).transpose(0, 1).contiguous()
    embedding_error = (native_embedding.float() - geometry_embedding.float()).abs().max()
    common = {
        "input_ids": ids,
        "position_ids": positions,
        "attention_mask": mask,
        "extra_block_kwargs": {
            "spherical_time_condition": condition,
            "spherical_position_ids": positions,
        },
        "runtime_gather_output": False,
    }
    native_logits = core(**common)
    geometry_logits = core(**common, decoder_input=geometry_embedding)
    native_logits = native_logits[0] if isinstance(native_logits, tuple) else native_logits
    geometry_logits = geometry_logits[0] if isinstance(geometry_logits, tuple) else geometry_logits
    logit_error = (native_logits.float() - geometry_logits.float()).abs().max()
    core.train(was_training)
    return _global_max(embedding_error), _global_max(logit_error)


class SphericalAcceptanceCallback(Callback):
    """Enforce distributed invariants and emit truthful runtime diagnostics."""

    def __init__(
        self,
        summary_path: str | None = None,
        *,
        panel_interval: int = 100,
        panel_prompts: tuple[str, ...] | None = None,
        panel_examples: tuple[tuple[str, str], ...] | None = None,
        panel_gen_length: int = 512,
        panel_steps: int = 32,
        model_only_interval: int = 2000,
        model_only_warmup_interval: int = 0,
        model_only_warmup_steps: int = 0,
        model_only_load_path: str | None = None,
        wandb_step_offset: int = 0,
    ):
        if model_only_interval < 0 or model_only_warmup_interval < 0 or model_only_warmup_steps < 0:
            raise ValueError("model-only checkpoint intervals and warmup steps must be non-negative")
        if wandb_step_offset < 0:
            raise ValueError("wandb_step_offset must be non-negative")
        self.wandb_step_offset = wandb_step_offset
        self.summary_path = summary_path
        self.panel_interval = panel_interval
        if panel_examples is not None:
            self.panel_examples = panel_examples
        elif panel_prompts is not None:
            self.panel_examples = tuple((prompt, "") for prompt in panel_prompts)
        else:
            self.panel_examples = tuple((case["prompt"], case["reference_answer"]) for case in FIXED_PANEL)
        if panel_examples is None and panel_prompts is None:
            self.panel_sample_ids = tuple(case["sample_id"] for case in FIXED_PANEL)
            self.panel_seeds = tuple(case["seed"] for case in FIXED_PANEL)
        else:
            self.panel_sample_ids = tuple(f"custom:{index}" for index in range(len(self.panel_examples)))
            self.panel_seeds = tuple(1234 + index for index in range(len(self.panel_examples)))
        self.panel_gen_length = panel_gen_length
        self.panel_steps = panel_steps
        self.model_only_interval = model_only_interval
        self.model_only_warmup_interval = model_only_warmup_interval
        self.model_only_warmup_steps = model_only_warmup_steps
        self.model_only_load_path = model_only_load_path
        self._decode_metric_logging = False
        self._wandb_training_axis_defined = False

    def _log_wandb(self, wandb, metrics: dict, completed_step: int) -> None:
        """Offset presentation only; optimizer and checkpoint cursors remain local."""
        if wandb is None:
            return
        display_step = completed_step + self.wandb_step_offset
        payload = {**metrics, "progress/local_step": completed_step, "progress/cumulative_step": display_step}
        event_steps = False
        if self._decode_metric_logging:
            run = getattr(wandb, "run", None)
            config = getattr(run, "config", {})
            history = config.get(DECODE_HISTORY_CONFIG_KEY, {})
            event_steps = (
                isinstance(history, Mapping)
                and history.get("event_steps") is True
                and history.get("run_id") == getattr(run, "id", None)
            )
        if event_steps:
            if not self._wandb_training_axis_defined:
                wandb.define_metric(DECODE_HISTORY_STEP_METRIC)
                wandb.define_metric("*", step_metric=DECODE_HISTORY_STEP_METRIC, step_sync=True)
                self._wandb_training_axis_defined = True
            # The run may now contain appended historical measurements. Let W&B
            # choose the next event index, while plots use the real training step.
            wandb.log(payload)
        else:
            wandb.log(payload, step=display_step)

    @staticmethod
    @torch.no_grad()
    def _conditioning_metrics(core) -> dict[str, float]:
        device = next(core.time_conditioner.parameters()).device
        times = torch.tensor([[0.0, 0.5, 1.0]], device=device)
        coordinate = getattr(getattr(core, "config", None), "spherical_condition_coordinate", "t")
        inputs = conditioning_input(times, gamma_from_t(times), coordinate)
        condition = core.time_conditioner(inputs)
        from .conditioning import ada_condition_input

        modulation_input = ada_condition_input(
            condition, getattr(getattr(core, "config", None), "spherical_ada_activation", "identity")
        )
        norm = condition.float().norm(dim=-1)
        raw_condition = core.time_conditioner.raw_embed(inputs) if hasattr(core.time_conditioner, "raw_embed") else condition
        metrics = {
            "conditioning/raw_output_norm_max": float(raw_condition.float().norm(dim=-1).max()),
            "conditioning/raw_output_rms_max": float(raw_condition.float().square().mean(dim=-1).sqrt().max()),
            "conditioning/output_norm_max": float(norm.max()),
            "conditioning/clean_output_norm": float(norm[0, 0]),
            "conditioning/ada_input_norm_max": float(modulation_input.float().norm(dim=-1).max()),
            "conditioning/input_at_t05": float(inputs[0, 1]),
            "adapter/beta": float(core.spherical_adapter.beta),
            "adapter/bias_norm": float(core.spherical_adapter.bias.float().norm()),
        }
        for layer_index in (0, len(core.decoder.layers) - 1):
            values = core.decoder.layers[layer_index].spherical_ada_projection(modulation_input).float().chunk(6, dim=-1)
            for name, value in zip(("attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"), values):
                metrics[f"conditioning/layer{layer_index}/{name}_max_abs"] = float(value.abs().max())
        return metrics

    def on_train_start(self, context: CallbackContext) -> None:
        core = _core(context.model)
        self._decode_metric_logging = bool(getattr(core.config, "spherical_decode_training", False))
        wandb = context.state.wandb_logger
        if (self._decode_metric_logging and os.environ.get("SPHERICAL_WANDB_DECODE_EVENT_STEPS") == "1"
                and wandb is not None and getattr(wandb, "run", None) is not None):
            history = wandb.run.config.get(DECODE_HISTORY_CONFIG_KEY, {})
            if not isinstance(history, Mapping) or history.get("run_id") != wandb.run.id:
                history = {}
            wandb.config.update({DECODE_HISTORY_CONFIG_KEY: {
                **history, "version": 1, "event_steps": True, "run_id": wandb.run.id,
                "axis": DECODE_HISTORY_STEP_METRIC,
            }}, allow_val_change=True)
        logger.info("Ada condition activation: %s", getattr(core.config, "spherical_ada_activation", "identity"))
        logger.info("Conditioning time scale=%s compute dtype=%s",
                    getattr(core.config, "spherical_time_embedding_scale", 1.0),
                    getattr(core.config, "spherical_condition_compute_dtype", "model"))
        logger.info("Conditioner input coordinate: %s; hard RMS bound: %s; L2 weight: %s",
                    getattr(context.state.cfg.model, "spherical_condition_coordinate", "t"),
                    getattr(context.state.cfg.model, "spherical_condition_rms_max", None),
                    getattr(context.state.cfg.model, "spherical_condition_l2_weight", 0.0))
        logger.info("Noise sampler gamma-uniform block probability: %s",
                    getattr(context.state.cfg.model, "spherical_gamma_uniform_probability", 0.0))
        if all(
            getattr(context.state.cfg.model, name, 10.0) == 1.0
            for name in ("conditioner_lr_mult", "ada_lr_mult", "adapter_lr_mult")
        ):
            _set_uniform_group_lr(context.optimizer, context.scheduler)
            logger.info("All optimizer LR groups aligned to backbone at scheduler sample step %d", context.scheduler.num_steps)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info("optimizer LR groups: %s", [
                {key: group.get(key) for key in ("max_lr", "min_lr", "lr", "wd_mult")}
                for group in getattr(context.optimizer, "param_groups", [])
            ])
        _warmup_optimizer_collectives(context.optimizer, next(core.parameters()).device)
        loaded_step = int(context.state.train_state.step)
        if loaded_step and context.state.cfg.checkpoint.load:
            load_root = Path(context.state.cfg.checkpoint.load)
            iteration_dir = (
                load_root
                if is_checkpoint_iteration_directory(str(load_root))
                else load_root / f"iter_{loaded_step:07d}"
            )
            resume_metadata = load_spherical_metadata(iteration_dir)
            if resume_metadata.checkpoint_type != "resume":
                raise RuntimeError("training resume did not load checkpoint_type=resume")
            warn_fingerprint_mismatch(
                "resume parameter", resume_metadata.parameter_fingerprint, parameter_fingerprint(context.model)
            )
        if self.model_only_load_path:
            expected_fingerprint = parameter_fingerprint(context.model)
            declared_metadata = load_spherical_metadata(self.model_only_load_path)
            if declared_metadata.checkpoint_type != "model_only":
                raise RuntimeError("checkpoint sidecar is not checkpoint_type=model_only")
            if declared_metadata.spherical_condition_dim != context.state.cfg.model.spherical_condition_dim:
                raise RuntimeError("model-only spherical_condition_dim does not match the allocated model")
            if loaded_step and declared_metadata.completed_step != loaded_step:
                raise RuntimeError("cannot combine resumed optimizer state with model-only weights from a different step")
            warn_fingerprint_mismatch("model-only parameter", declared_metadata.parameter_fingerprint, expected_fingerprint)
            warn_fingerprint_mismatch(
                "model-only data", declared_metadata.data_manifest_fingerprint,
                context.state.cfg.dataset.manifest_fingerprint,
            )
            metadata = load_model_only_checkpoint(context.model, self.model_only_load_path)
            if metadata != declared_metadata:
                raise RuntimeError("model-only sidecar and distributed artifact metadata differ")
            if not loaded_step and context.optimizer is not None:
                from megatron.bridge.training.optim import sync_hybrid_device_optimizer_fp32_master_copies

                context.optimizer.reload_model_params()
                sync_hybrid_device_optimizer_fp32_master_copies(context.optimizer)
            logger.info("loaded model-only checkpoint at step %d: %s", metadata.completed_step, self.model_only_load_path)
        embedding_error, logit_error = _full_clean_parity(core)
        replicated_diff = _replicated_tp_max_diff(core)
        if embedding_error != 0.0 or logit_error != 0.0 or replicated_diff != 0.0:
            raise RuntimeError(
                "spherical validation failed: "
                f"embedding={embedding_error}, logits={logit_error}, replicated={replicated_diff}"
            )
        adapter, ada = _replicated_parameters(core)
        context.user_state["spherical_checksums_before"] = {
            "adapter": _parameter_checksum(adapter),
            "ada": _parameter_checksum(ada),
        }
        context.user_state["spherical_acceptance"] = {
            "clean_embedding_max_abs": embedding_error,
            "clean_logits_max_abs": logit_error,
            "replicated_tp_max_abs_before": replicated_diff,
            "topology": {"world_size": int(os.environ.get("WORLD_SIZE", "8")), "tp": int(os.environ.get("SPHERICAL_TP_SIZE", "4")), "dp": int(os.environ.get("SPHERICAL_DP_SIZE", "2")), "pp": int(os.environ.get("SPHERICAL_PP_SIZE", "1")), "cp": int(os.environ.get("SPHERICAL_CP_SIZE", "1"))},
            "completed_step": int(context.state.train_state.step),
            "data_manifest_fingerprint": str(context.state.cfg.dataset.manifest_fingerprint),
            "protocol_version": str(context.state.cfg.dataset.protocol_version),
            "tail_block_protocol": getattr(context.state.cfg.dataset, "tail_block_protocol", "original_length_v0"),
            "tail_eos_token_id": getattr(context.state.cfg.dataset, "eos_token_id", None),
        }
        logger.info(
            "spherical clean parity passed: embedding_max_abs=%.6g logits_max_abs=%.6g replicated_tp_max_abs=%.6g",
            embedding_error,
            logit_error,
            replicated_diff,
        )
        from ...data.spherical_mixture import load_manifest

        data_manifest = load_manifest(context.state.cfg.dataset.prepared_manifest)
        context.user_state["spherical_acceptance"]["data_valid_targets"] = sum(
            source["counts"]["train_samples"] for source in data_manifest["sources"]
        )
        context.user_state["spherical_start_step"] = int(context.state.train_state.step)
        context.user_state["spherical_start_time"] = time.perf_counter()
        context.user_state["spherical_rolling"] = []
        torch.cuda.reset_peak_memory_stats()

        original_step = context.optimizer.step
        original_ready_step = context.optimizer.step_with_ready_grads

        def step_with_post_clip_capture(*args, **kwargs):
            # Native optimizer has unscaled/clipped main gradients at this point.
            # Reuse its distributed norm reduction, not a formula from pre-clip norm.
            context.user_state["spherical_post_clip"] = {
                "step": int(context.state.train_state.step) + 1,
                "norm": float(context.optimizer.get_grad_norm()),
            }
            return original_ready_step(*args, **kwargs)

        context.optimizer.step_with_ready_grads = step_with_post_clip_capture

        def step_with_grad_capture(*args, **kwargs):
            completed_step = int(context.state.train_state.step) + 1
            log_interval = int(getattr(context.state.cfg.logger, "log_interval", 10) or 10)
            should_capture = completed_step - context.user_state["spherical_start_step"] <= 3 or completed_step % log_interval == 0
            if should_capture:
                adapter_parameters, ada_parameters = _replicated_parameters(core)
                conditioner_norm, layer_min = _ada_grad_details(core)
                context.user_state["spherical_pre_step_grads"] = {
                    "step": completed_step,
                    "adapter": float(_grad_norm(adapter_parameters)),
                    "ada": float(_grad_norm(ada_parameters)),
                    "conditioner": conditioner_norm,
                    "ada_layer_min": layer_min,
                }
            return original_step(*args, **kwargs)

        context.optimizer.step = step_with_grad_capture

    def on_train_step_end(self, context: CallbackContext) -> None:
        core = _core(context.model)
        summary = context.user_state["spherical_acceptance"]
        completed_step = int(context.state.train_state.step) + 1
        initial_steps = completed_step - context.user_state["spherical_start_step"] <= 3
        if initial_steps or completed_step % 100 == 0:
            replicated_diff = _replicated_tp_max_diff(core)
            if replicated_diff != 0.0:
                raise RuntimeError(f"spherical parameters diverged across TP ranks: max_abs={replicated_diff}")
            summary["replicated_tp_max_abs_after"] = replicated_diff
        summary["completed_step"] = completed_step
        summary["skipped"] = bool(context.skipped_iter)
        summary["grad_norm"] = float(context.grad_norm) if context.grad_norm is not None else None
        post_clip = context.user_state.get("spherical_post_clip", {})
        if post_clip.get("step") == completed_step:
            summary["post_clip_grad_norm"] = post_clip["norm"]
            pre = summary["grad_norm"]
            summary["grad_clip_ratio"] = post_clip["norm"] / pre if pre and pre > 0 else 1.0
            logger.info("gradient clipping local=%d cumulative=%d pre=%.9g post=%.9g ratio=%.9g threshold=%.9g",
                        completed_step, completed_step + self.wandb_step_offset,
                        pre, post_clip["norm"], summary["grad_clip_ratio"], context.state.cfg.optimizer.clip_grad)
        grads = context.user_state.get("spherical_pre_step_grads", {})
        if grads.get("step") == completed_step:
            summary["adapter_grad_norm"] = grads.get("adapter")
            summary["ada_grad_norm"] = grads.get("ada")
            summary["conditioner_grad_norm"] = grads.get("conditioner")
            summary["ada_layer_min_grad_norm"] = grads.get("ada_layer_min")
            summary["conditioner_grad_norm_max"] = max(
                float(summary.get("conditioner_grad_norm_max", 0.0)), float(grads.get("conditioner", 0.0))
            )
            summary["ada_layer_min_grad_norm_min"] = min(
                float(summary.get("ada_layer_min_grad_norm_min", float("inf"))),
                float(grads.get("ada_layer_min", 0.0)),
            )
        if initial_steps or completed_step % 100 == 0:
            adapter, ada = _replicated_parameters(core)
            before = context.user_state["spherical_checksums_before"]
            current_adapter = _parameter_checksum(adapter)
            current_ada = _parameter_checksum(ada)
            summary["adapter_parameter_checksum_delta"] = max(
                abs(current_adapter[index] - before["adapter"][index]) for index in range(2)
            )
            summary["ada_parameter_checksum_delta"] = max(
                abs(current_ada[index] - before["ada"][index]) for index in range(2)
            )
        summary["adapter_beta_norm"] = float(core.spherical_adapter.log_beta.detach().float().norm())
        summary["adapter_bias_norm"] = float(core.spherical_adapter.bias.detach().float().norm())
        summary["elapsed_seconds"] = time.perf_counter() - context.user_state["spherical_start_time"]
        summary["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        summary["max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        conditioning_metrics = {}
        if initial_steps or completed_step % 100 == 0:
            conditioning_metrics = self._conditioning_metrics(core)
            summary["conditioning_metrics"] = conditioning_metrics
            logger.info("conditioning monitor local=%d cumulative=%d: %s",
                        completed_step, completed_step + self.wandb_step_offset, conditioning_metrics)
        if context.loss_dict:
            for key in (
                "lm loss",
                "target_tokens",
                "sampling/t",
                "sampling/gamma",
                "sampling/gamma_uniform_fraction",
                "sampling/t_uniform_fraction",
            ):
                if key in context.loss_dict:
                    summary[key.replace(" ", "_").replace("/", "_")] = float(context.loss_dict[key])
            rolling = context.user_state["spherical_rolling"]
            rolling.append(
                {
                    key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                    for key, value in context.loss_dict.items()
                    if key == "lm loss" or key.startswith("t_stage/")
                }
            )
            del rolling[:-100]

        if self.panel_interval and completed_step % self.panel_interval == 0:
            self._run_panel(context, completed_step)
        if self._should_save_model_only(completed_step):
            self._save_model_only(context, completed_step)

        wandb = context.state.wandb_logger
        if wandb is not None:
            groups = getattr(context.optimizer, "param_groups", [])
            wandb_metrics = {
                **conditioning_metrics,
                "conditioning/grad_norm": summary.get("conditioner_grad_norm"),
                "train/loss": summary.get("lm_loss"),
                "train/actual_lr": max((group.get("lr", 0.0) for group in groups), default=0.0),
                "train/global_grad_norm": summary["grad_norm"],
                "train/post_clip_grad_norm": summary.get("post_clip_grad_norm"),
                "train/grad_clip_ratio": summary.get("grad_clip_ratio"),
                "train/max_memory_allocated_gib": summary["max_memory_allocated_gib"],
                "train/max_memory_reserved_gib": summary["max_memory_reserved_gib"],
                "sampling/gamma_uniform_fraction": summary.get("sampling_gamma_uniform_fraction"),
                "sampling/t_uniform_fraction": summary.get("sampling_t_uniform_fraction"),
                "debug/sampling_t_mean": summary.get("sampling_t"),
                "debug/sampling_gamma_mean": summary.get("sampling_gamma"),
                "data/manifest_fingerprint": summary["data_manifest_fingerprint"],
                "data/protocol_version": summary["protocol_version"],
                "data/drop_rate": summary.get("data_drop_rate", 0.0),
            }
            if context.loss_dict:
                for key, value in context.loss_dict.items():
                    if key.startswith(("target_region/", "data/")):
                        wandb_metrics[key] = float(value)
            wandb_metrics.update(decode_wandb_metrics(
                context.loss_dict,
                enabled=bool(getattr(core.config, "spherical_decode_training", False)),
            ))
            wandb_metrics["data/tail_block_protocol"] = getattr(context.state.cfg.dataset, "tail_block_protocol", "original_length_v0")
            # Log each distinct effective optimizer-group LR instead of only the maximum.
            for key in ("conditioning/input_mean", "conditioning/l2", "conditioning/l2_weighted", "loss/total"):
                if context.loss_dict and key in context.loss_dict:
                    wandb_metrics[key] = float(context.loss_dict[key])
            for group in groups:
                peak = group.get("max_lr", context.state.cfg.optimizer.lr)
                wandb_metrics[f"lr_groups/peak_{peak:.8g}"] = group.get("lr", 0.0)
            rolling = context.user_state["spherical_rolling"]
            if rolling:
                wandb_metrics.update(
                    {key: value for key, value in rolling[-1].items() if key.startswith("t_stage/")}
                )
                for key in rolling[-1]:
                    values = [item[key] for item in rolling if key in item]
                    if key.startswith("t_stage/"):
                        count_key = key.rsplit("/", 1)[0] + "/valid_tokens"
                        count = sum(item.get(count_key, 0.0) for item in rolling)
                        value = count if key == count_key else sum(
                            item.get(key, 0.0) * item.get(count_key, 0.0) for item in rolling
                        ) / max(count, 1.0)
                    else:
                        value = sum(values) / len(values)
                    wandb_metrics[f"rolling100/{key.replace(' ', '_')}"] = value
            self._log_wandb(wandb, wandb_metrics, completed_step)

    def _should_save_model_only(self, completed_step: int) -> bool:
        return completed_step > 0 and bool(
            (self.model_only_interval and completed_step % self.model_only_interval == 0)
            or (
                self.model_only_warmup_interval
                and completed_step <= self.model_only_warmup_steps
                and completed_step % self.model_only_warmup_interval == 0
            )
        )

    def _run_panel(self, context: CallbackContext, completed_step: int) -> None:
        if parallel_state.model_parallel_is_initialized():
            if (
                parallel_state.get_pipeline_model_parallel_world_size() != 1
                or parallel_state.get_context_parallel_world_size() != 1
            ):
                raise RuntimeError("fixed panel supports TP + DP with PP=CP=1")
            replica_rank = parallel_state.get_data_parallel_rank()
            replica_count = parallel_state.get_data_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            if (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                raise RuntimeError(
                    "distributed fixed panel requires initialized model-parallel groups"
                )
            replica_rank, replica_count, tp_rank, tp_size = 0, 1, 0, 1
        assigned = panel_indices(
            len(self.panel_examples), replica_rank=replica_rank, replica_count=replica_count
        )
        core = _core(context.model)
        tokenizer = getattr(context.state.tokenizer, "tokenizer", context.state.tokenizer)
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        records = []
        error = None
        torch.cuda.synchronize()
        panel_start = time.perf_counter()
        was_training = core.training
        core.eval()
        try:
            with preserved_training_state(
                scheduler=context.scheduler, optimizer=context.optimizer
            ):
                if apply_template is None:
                    raise RuntimeError(
                        "fixed panel requires the same explicit chat template as training"
                    )
                eos_id, pad_id = _panel_special_ids(tokenizer)
                for prompt_index in assigned:
                    prompt, standard_answer = self.panel_examples[prompt_index]
                    question_start = time.perf_counter()
                    encoded = apply_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        truncate_history_thinking=True,
                        return_tensors="pt",
                    )
                    ids = _panel_input_ids(
                        encoded, core.spherical_geometry.E_geometry.device
                    )
                    output, nfe = spherical_block_sample(
                        core,
                        core.spherical_geometry,
                        core.spherical_adapter,
                        ids,
                        gen_length=self.panel_gen_length,
                        block_size=4,
                        steps=4,
                        mask_token_id=int(core.config.mask_token_id),
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        seed=self.panel_seeds[prompt_index],
                        deterministic=True,
                    )
                    generated = output[0, ids.shape[1] :]
                    eos_hits = (generated == eos_id).nonzero(as_tuple=False)
                    token_count = (
                        int(eos_hits[0]) + 1 if eos_hits.numel() else generated.numel()
                    )
                    row = {
                        "step": completed_step + self.wandb_step_offset,
                        "prompt": prompt,
                        "standard_answer": standard_answer,
                        "result": _panel_decode(
                            tokenizer, generated[:token_count].tolist()
                        ),
                    }
                    if tp_rank == 0:
                        records.append(
                            {
                                "index": prompt_index,
                                "row": row,
                                "detail": {
                                    "sample_id": self.panel_sample_ids[prompt_index],
                                    "replica_rank": replica_rank,
                                    "tp_size": tp_size,
                                    "seed": self.panel_seeds[prompt_index],
                                    "steps": self.panel_steps,
                                    "max_new_tokens": self.panel_gen_length,
                                    "generated_tokens": token_count,
                                    "eos_reached": bool(eos_hits.numel()),
                                    "nfe": nfe,
                                    "seconds": time.perf_counter() - question_start,
                                },
                            }
                        )
                        if resolve_decode_endpoint(core.config):
                            records[-1]["detail"].update(
                                endpoint_t=0.0, decode_extra_forward=True,
                                nfe_per_full_block=self.panel_steps + 1,
                            )
                        logger.info(
                            "inference question %s complete: DP rank %d, %d tokens, %d NFE, %.3fs",
                            self.panel_sample_ids[prompt_index],
                            replica_rank,
                            token_count,
                            nfe,
                            records[-1]["detail"]["seconds"],
                        )
        except Exception as exc:
            # Surface ordinary local failures on every DP replica before any
            # replica resumes training. Fatal errors inside TP collectives are
            # still handled by the distributed launcher's failure propagation.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            core.train(was_training)
        torch.cuda.synchronize()
        rows, details, elapsed = gather_panel_results(
            {
                "records": records,
                "error": error,
                "elapsed": time.perf_counter() - panel_start,
            },
            count=len(self.panel_examples),
        )
        total_nfe = sum(detail["nfe"] for detail in details)
        context.user_state["spherical_acceptance"]["panel_last_nfe"] = (
            details[-1]["nfe"] if details else 0
        )
        context.user_state["spherical_acceptance"]["panel_details"] = details
        context.user_state["spherical_acceptance"]["panel_replica_count"] = replica_count
        context.user_state["spherical_acceptance"]["panel_elapsed_seconds"] = elapsed
        context.user_state["spherical_acceptance"]["panel_total_nfe"] = total_nfe
        logger.info(
            "inference panel at step %d: %.3fs, %d NFE, %d questions, DP=%d TP=%d, local indices=%s",
            completed_step,
            elapsed,
            total_nfe,
            len(rows),
            replica_count,
            tp_size,
            list(assigned),
        )
        context.user_state.setdefault("spherical_panel", []).extend(rows)
        context.user_state["spherical_acceptance"]["panel_last"] = rows
        wandb = context.state.wandb_logger
        # GlobalState creates the logger only on the last global rank. All
        # ranks have gathered the complete panel, so its owner can log it once.
        if wandb is not None:
            columns = ["step", "prompt", "standard_answer", "result"]
            history_directory = os.environ.get("SPHERICAL_PANEL_HISTORY_DIR")
            if history_directory and wandb.run is not None:
                context.user_state["spherical_panel"] = persist_panel_rows(
                    history_directory,
                    f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}",
                    context.user_state["spherical_panel"],
                )
            self._log_wandb(
                wandb,
                {
                    "inference/sample": wandb.Table(
                        columns=columns,
                        data=[
                            [row[key] for key in columns]
                            for row in context.user_state["spherical_panel"]
                        ],
                    ),
                    "inference/elapsed_seconds": elapsed,
                    "inference/total_nfe": total_nfe,
                    "inference/replica_count": replica_count,
                    "inference/question_count": len(rows),
                    "inference/steps": self.panel_steps,
                    "inference/max_new_tokens": self.panel_gen_length,
                    "inference/eos_fraction": sum(item["eos_reached"] for item in details)
                    / max(len(details), 1),
                    "inference/details": wandb.Table(
                        columns=list(details[0]) if details else ["sample_id"],
                        data=[list(detail.values()) for detail in details],
                    ),
                },
                completed_step=completed_step,
            )

    def _save_model_only(self, context: CallbackContext, completed_step: int) -> None:
        cfg = context.state.cfg
        metadata = SphericalCheckpointMetadata(
            checkpoint_type="model_only",
            completed_step=completed_step,
            tp_size=int(os.environ.get("SPHERICAL_TP_SIZE", "4")),
            dp_size=int(os.environ.get("SPHERICAL_DP_SIZE", "2")),
            parameter_fingerprint=parameter_fingerprint(context.model),
            data_manifest_fingerprint=str(cfg.dataset.manifest_fingerprint),
            protocol_version=str(cfg.model.spherical_protocol_metadata["version"]),
            block_size=int(cfg.model.block_size),
            tail_block_protocol=getattr(cfg.dataset, "tail_block_protocol", "original_length_v0"),
            tail_eos_token_id=getattr(cfg.dataset, "eos_token_id", None),
            spherical_condition_dim=int(cfg.model.spherical_condition_dim),
            spherical_condition_coordinate=cfg.model.spherical_condition_coordinate,
            spherical_ada_activation=getattr(cfg.model, "spherical_ada_activation", "identity"),
            spherical_time_embedding_scale=getattr(cfg.model, "spherical_time_embedding_scale", 1.0),
            spherical_condition_compute_dtype=getattr(cfg.model, "spherical_condition_compute_dtype", "model"),
            spherical_condition_norm=getattr(cfg.model, "spherical_condition_norm", "none"),
            **decode_fields(cfg.model),
        )
        target = Path(cfg.checkpoint.save) / "model_only" / f"iter_{completed_step:07d}"
        logger.info("saving model-only inference checkpoint at step %d: %s", completed_step, target)
        save_model_only_checkpoint(context.model, target, metadata)
        logger.info("saved model-only inference checkpoint at step %d: %s (retained)", completed_step, target)

    def on_checkpoint_save(self, context: CallbackContext) -> None:
        """Attach strict spherical protocol metadata to Bridge's native resume artifact."""
        if torch.distributed.get_rank() != 0:
            return
        cfg = context.state.cfg
        completed_step = int(context.state.train_state.step)
        metadata = SphericalCheckpointMetadata(
            checkpoint_type="resume",
            completed_step=completed_step,
            tp_size=int(os.environ.get("SPHERICAL_TP_SIZE", "4")),
            dp_size=int(os.environ.get("SPHERICAL_DP_SIZE", "2")),
            parameter_fingerprint=parameter_fingerprint(context.model),
            data_manifest_fingerprint=str(cfg.dataset.manifest_fingerprint),
            protocol_version=str(cfg.model.spherical_protocol_metadata["version"]),
            block_size=int(cfg.model.block_size),
            tail_block_protocol=getattr(cfg.dataset, "tail_block_protocol", "original_length_v0"),
            tail_eos_token_id=getattr(cfg.dataset, "eos_token_id", None),
            spherical_condition_dim=int(cfg.model.spherical_condition_dim),
            spherical_condition_coordinate=cfg.model.spherical_condition_coordinate,
            spherical_ada_activation=getattr(cfg.model, "spherical_ada_activation", "identity"),
            spherical_time_embedding_scale=getattr(cfg.model, "spherical_time_embedding_scale", 1.0),
            spherical_condition_compute_dtype=getattr(cfg.model, "spherical_condition_compute_dtype", "model"),
            spherical_condition_norm=getattr(cfg.model, "spherical_condition_norm", "none"),
            **decode_fields(cfg.model),
        )
        target = Path(cfg.checkpoint.save) / f"iter_{completed_step:07d}"
        write_spherical_metadata(target, metadata)

    def on_train_end(self, context: CallbackContext) -> None:
        summary = context.user_state["spherical_acceptance"]
        peaks = torch.tensor(
            [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
            device=next(_core(context.model).parameters()).device, dtype=torch.float64,
        )
        torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
        summary["all_ranks_max_memory_allocated_gib"], summary["all_ranks_max_memory_reserved_gib"] = (
            peaks / 2**30
        ).tolist()
        if int(summary.get("completed_step", 0)) == int(context.user_state["spherical_start_step"]):
            summary["resume_load_only"] = True
            self._write_summary(summary)
            return
        required_positive = [
            "adapter_grad_norm",
            "ada_grad_norm",
            "ada_layer_min_grad_norm_min",
        ]
        steps_run = int(summary.get("completed_step", 0)) - int(context.user_state["spherical_start_step"])
        summary["verification_steps"] = steps_run
        summary["parameter_update_verified"] = bool(
            float(summary.get("adapter_parameter_checksum_delta", 0.0)) > 0.0
            and float(summary.get("ada_parameter_checksum_delta", 0.0)) > 0.0
        )
        summary["conditioner_gradient_verified"] = bool(
            float(summary.get("conditioner_grad_norm_max", 0.0)) > 0.0
        )
        if steps_run >= 2:
            required_positive.extend(
                ("adapter_parameter_checksum_delta", "ada_parameter_checksum_delta")
            )
        failed = [name for name in required_positive if not float(summary.get(name, 0.0)) > 0.0]
        # The cosine scheduler starts at LR=0. Step 1 therefore cannot move the
        # zero Ada projection; step 2 moves it, and step 3 is the first forward
        # in which the conditioner can receive a nonzero gradient.
        if int(summary.get("completed_step", 0)) >= 3 and not float(summary.get("conditioner_grad_norm_max", 0.0)) > 0.0:
            failed.append("conditioner_grad_norm_max")
        if failed:
            raise RuntimeError(f"spherical parameters did not receive gradients/update: {failed}")
        self._write_summary(summary)

    def _write_summary(self, summary: dict) -> None:
        if self.summary_path is None or torch.distributed.get_rank() != 0:
            return
        target = Path(self.summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


__all__ = ["SphericalAcceptanceCallback"]
