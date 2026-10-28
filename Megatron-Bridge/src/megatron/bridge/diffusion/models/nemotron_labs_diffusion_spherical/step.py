# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Bridge forward step for spherical noisy-position CE."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import partial
from typing import Iterable

import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_model_config, unwrap_model

from .geometry import normalize, random_sphere, slerp, terminal_noise
from .conditioning import conditioning_input
from .inference import deterministic_topk, merge_tp_topk, tp_vocab_range
from .layout import build_decoder_input, continuous_position_ids, time_layout, three_segment_attention_mask
from .scheduler import SphericalScheduler


logger = logging.getLogger(__name__)


def conditioner_l2_sum(condition: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Sum per-sample mean-square conditions; native schedule averages samples.

    All context/clean/noisy positions (including padding) contribute within a
    valid sample. Replicated TP gradients are averaged by conditioner metadata.
    """
    work = condition if condition.dtype == torch.float64 else condition.float()
    per_sample = work.square().mean(dim=(1, 2))
    return per_sample[loss_mask.sum(dim=-1) > 0].sum()


def _spherical_loss_callback(
    valid: torch.Tensor,
    report: dict[str, tuple[torch.Tensor, torch.Tensor]],
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Return the standard MCore loss callback tuple for the schedule."""
    return output_tensor, valid, report


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move tensor fields to the active rank without altering metadata fields."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def spherical_ce_loss(
    core_model,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    check_finite: bool = True,
    return_token_losses: bool = False,
):
    """Compute per-sample vocab-parallel CE only at valid noisy positions."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            "native spherical training expects batch-first logits [B,S,V] and labels [B,S]; "
            f"got logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    token_losses = core_model.compute_language_model_loss(labels, logits.transpose(0, 1).contiguous()).float()
    mask = loss_mask.float()
    token_counts = mask.sum(dim=-1)
    valid_samples_mask = token_counts > 0
    sample_losses = (token_losses * mask).sum(dim=-1) / token_counts.clamp_min(1.0)
    loss_sum = sample_losses[valid_samples_mask].sum()
    valid_samples = valid_samples_mask.sum().detach().to(torch.int)
    if check_finite and not torch.isfinite(loss_sum):
        raise FloatingPointError("spherical CE is non-finite")
    result = (loss_sum, valid_samples, {"lm loss": torch.stack((loss_sum.detach(), valid_samples.float()))})
    return (*result, token_losses) if return_token_losses else result


@torch.no_grad()
def _global_predictions(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Return global vocab ids for vocab-sharded logits, preserving TP tie rules."""
    values, local_ids = deterministic_topk(logits.detach(), 1, deterministic=False)
    vocab_start, _ = tp_vocab_range(vocab_size)
    _, predictions = merge_tp_topk(
        values, local_ids, vocab_offset=vocab_start, k=1, deterministic=False
    )
    # Training core output is batch-first; the loss helper transposes it for
    # MCore's vocab-parallel CE implementation.
    return predictions.squeeze(-1).contiguous()


def _scalar_batch_value(value, *, default: int) -> int:
    """Accept an absent value or a batch of identical scalar protocol ids."""
    if value is None:
        return default
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() == 0 or not torch.equal(tensor, tensor[:1].expand_as(tensor)):
        raise ValueError("protocol token ids must be identical throughout a microbatch")
    return int(tensor[0].item())


def _stable_batch_seed(seed: int, step: int, dp_rank: int, batch: dict) -> int:
    """Derive resume-stable noise from checkpointed step and sampled coordinates."""
    identity = {}
    for key in ("source_shard", "conversation_id", "turn_id"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, tuple):
            value = list(value)
        identity[key] = value
    payload = json.dumps(
        {"seed": seed, "step": step, "dp_rank": dp_rank, "batch": identity},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


_TIME_BINS = (
    (0.0, 0.2, "0.0_0.2"),
    (0.2, 0.4, "0.2_0.4"),
    (0.4, 0.6, "0.4_0.6"),
    (0.6, 0.8, "0.6_0.8"),
    (0.8, 0.9, "0.8_0.9"),
    (0.9, 0.95, "0.9_0.95"),
    (0.95, 0.98, "0.95_0.98"),
    (0.98, 1.0, "0.98_1.0"),
)
# Overlapping summaries are reported alongside the disjoint bins.
_TIME_SUMMARY_BINS = ((0.9, 1.0, "0.9_1.0"),)


@torch.no_grad()
def _global_time_metrics(
    time: torch.Tensor,
    losses: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    count_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Report sums/counts for Bridge to reduce once over DP and microbatches."""
    metrics = {}
    for lower, upper, name in _TIME_BINS + _TIME_SUMMARY_BINS:
        stage_mask = (time >= lower) & ((time < upper) if upper < 1.0 else (time <= upper))
        stage_losses = losses[stage_mask]
        stage_count = stage_mask.sum().to(losses.dtype)
        correct = ((predictions == labels) & stage_mask).sum().to(losses.dtype)
        metrics[f"t_stage/{name}/loss"] = torch.stack((stage_losses.sum(), stage_count))
        metrics[f"t_stage/{name}/accuracy"] = torch.stack((correct, stage_count))
        metrics[f"t_stage/{name}/valid_tokens"] = torch.stack((stage_count, stage_count.new_tensor(count_weight)))
    return metrics


@torch.no_grad()
def target_region_metrics(losses, predictions, labels, original_lengths, diffusion_lengths, *, eos_id, count_weight=1.0):
    """Token-weighted region metrics; counts distinguish absent EOS from accuracy zero."""
    offsets = torch.arange(labels.shape[1], device=labels.device)[None, :]
    original = offsets < original_lengths[:, None]
    valid = offsets < diffusion_lengths[:, None]
    fill = valid & ~original
    original_eos = original & (labels == eos_id)
    # Last EOS in the original target is the termination token, even before a newline.
    last_eos = torch.where(original_eos, offsets, -1).amax(dim=-1, keepdim=True)
    terminal = original_eos & (offsets == last_eos)
    report = {}
    for name, mask in (("original_target", original), ("eos_fill", fill),
                       ("original_terminal_eos", terminal), ("original_body", original & ~original_eos)):
        count = mask.sum().float()
        report[f"target_region/{name}/loss"] = torch.stack((losses[mask].sum(), count))
        report[f"target_region/{name}/accuracy"] = torch.stack((((predictions == labels) & mask).sum().float(), count))
        report[f"target_region/{name}/valid_tokens"] = torch.stack((count, count.new_tensor(count_weight)))
    for name, mask in (("original_target_tokens", original), ("eos_fill_tokens", fill),
                       ("supervised_target_tokens", valid), ("outer_padding_tokens", ~valid)):
        count = mask.sum().float()
        report[f"data/{name}"] = torch.stack((count, count.new_tensor(count_weight)))
    return report


class SphericalDGPTStep:
    """Bridge forward-step functor implementing the three-segment protocol."""

    def __init__(self, *, seed: int = 1234, scheduler: SphericalScheduler | None = None):
        self.seed = seed
        self.scheduler = scheduler or SphericalScheduler()
        self._scheduler_from_config = scheduler is None

    def __call__(self, state, data_iterator: Iterable, model, return_schedule_plan: bool = False):
        """Run one native MCore forward and return Bridge's loss callback."""
        batch = next(data_iterator)
        device = next(parameter for parameter in model.parameters() if parameter is not None).device
        batch = _move_batch(batch, device)
        core = unwrap_model(model)
        if hasattr(core, "language_model"):
            core = core.language_model
        geometry = getattr(core, "spherical_geometry", None)
        adapter = getattr(core, "spherical_adapter", None)
        if geometry is None or adapter is None:
            raise RuntimeError("native model is missing spherical_geometry/spherical_adapter")
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _stable_batch_seed(
                self.seed,
                int(state.train_state.step),
                parallel_state.get_data_parallel_rank(),
                batch,
            )
        )
        context_ids = batch["context_ids"].long()
        clean_ids = batch["clean_ids"].long()
        context_lengths = batch["context_lengths"].long()
        target_lengths = batch.get("diffusion_target_lengths", batch["target_lengths"]).long()
        original_lengths = batch.get("original_target_lengths", target_lengths).long()
        if "diffusion_target_lengths" in batch:
            offsets = torch.arange(clean_ids.shape[1], device=device)[None, :]
            expected_mask = offsets < target_lengths[:, None]
            if (not torch.equal(batch["target_lengths"], target_lengths)
                    or not torch.equal(batch["loss_mask"], expected_mask.float())):
                raise ValueError("Diffusion length, target_lengths alias and supervision must agree")
        batch_size, target_width = clean_ids.shape
        block_size = int(getattr(core, "spherical_block_size", getattr(get_model_config(core), "block_size", 32)))
        if "diffusion_target_lengths" in batch:
            expected_lengths = ((original_lengths + block_size - 1) // block_size) * block_size
            if not torch.equal(target_lengths, expected_lengths) or (original_lengths < 1).any():
                raise ValueError("Tail diffusion lengths must be the per-sample original block ceiling")
        context_width = context_ids.shape[1]
        blocks = target_width // block_size
        if self._scheduler_from_config:
            probability = float(getattr(core.config, "spherical_gamma_uniform_probability", 0.0))
            enabled = bool(getattr(core.config, "spherical_decode_training", False))
            options = {
                "gamma_uniform_probability": probability,
                "decode_enabled": enabled,
                "decode_probability": getattr(core.config, "spherical_decode_probability", 0.2),
                "decode_gamma_min": getattr(core.config, "spherical_decode_gamma_min", 0.85),
                "decode_gamma_max": getattr(core.config, "spherical_decode_gamma_max", 0.95),
            }
            if any(getattr(self.scheduler, key) != value for key, value in options.items()):
                self.scheduler = SphericalScheduler(**options)
        sample = self.scheduler.sample((batch_size, blocks), device=device, generator=generator)
        clean_state = normalize(geometry.clean(clean_ids))
        mask_id = _scalar_batch_value(batch.get("mask_token_id"), default=int(getattr(core.config, "mask_token_id", 100)))
        mask_direction = geometry.codeword(mask_id, device=device, dtype=clean_state.dtype).view(1, 1, -1)
        terminal = terminal_noise(
            random_sphere(clean_state.shape, device=device, dtype=clean_state.dtype, generator=generator),
            mask_direction,
            generator=generator,
        )
        noisy_gamma = sample.gamma.repeat_interleave(block_size, dim=-1)
        # Gamma follows the formula's direct noise coefficient: 0=clean, 1=noise.
        noisy_state = slerp(clean_state, terminal, noisy_gamma)
        decoder_input = build_decoder_input(
            geometry,
            adapter,
            context_ids,
            clean_ids,
            noisy_state,
        )
        block_condition = conditioning_input(
            sample.t, sample.gamma, getattr(core.config, "spherical_condition_coordinate", "t")
        )
        semantic_time = time_layout(
            block_condition, context_width=context_width, target_width=target_width, block_size=block_size
        )
        attention_mask = three_segment_attention_mask(
            context_lengths,
            target_lengths,
            context_width=context_width,
            target_width=target_width,
            block_size=block_size,
            dtype=decoder_input.dtype,
        )
        input_ids = torch.cat((context_ids, clean_ids, clean_ids), dim=1)
        semantic_position_ids = continuous_position_ids(
            context_lengths,
            target_lengths,
            context_width=context_width,
            target_width=target_width,
        )
        conditioner = getattr(core, "time_conditioner", None)
        if conditioner is None:
            raise RuntimeError("native model is missing the semantic-time conditioner")
        spherical_time_condition = conditioner(semantic_time)
        output = core(
            input_ids=input_ids,
            position_ids=semantic_position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input.transpose(0, 1).contiguous(),
            extra_block_kwargs={
                "spherical_time_condition": spherical_time_condition,
                "spherical_position_ids": semantic_position_ids,
            },
            runtime_gather_output=False,
        )
        logits = output[0] if isinstance(output, tuple) else output
        # Context/clean positions never contribute to CE. Do not build FP32
        # vocab-parallel softmax buffers for that unused majority of the sequence.
        logits = logits[:, -target_width:]
        labels = clean_ids
        loss_mask = batch["loss_mask"].float()
        loss_sum, valid_samples, report, token_losses = spherical_ce_loss(
            core,
            logits,
            labels,
            loss_mask,
            check_finite=state.cfg.rerun_state_machine.check_for_nan_in_loss,
            return_token_losses=True,
        )
        regularizer_weight = float(getattr(core.config, "spherical_condition_l2_weight", 0.0))
        if regularizer_weight:
            penalty = conditioner_l2_sum(spherical_time_condition, loss_mask)
            weighted_penalty = regularizer_weight * penalty
            loss_sum = loss_sum + weighted_penalty
            denominator = valid_samples.float()
            report["conditioning/l2"] = torch.stack((penalty.detach(), denominator))
            report["conditioning/l2_weighted"] = torch.stack((weighted_penalty.detach(), denominator))
            report["loss/total"] = torch.stack((loss_sum.detach(), denominator))
        noisy_losses = token_losses[:, -target_width:]
        noisy_mask = batch["loss_mask"].float()
        gamma_tokens = noisy_gamma.float()
        valid_losses = noisy_losses[noisy_mask.bool()]
        valid_gamma = gamma_tokens[noisy_mask.bool()]
        time_tokens = sample.t.repeat_interleave(block_size, dim=-1).float()
        valid_time = time_tokens[noisy_mask.bool()]
        # Keep the batch-first logits aligned with the [batch, target] mask.
        predictions = _global_predictions(logits[:, -target_width:].contiguous(), int(core.config.vocab_size))
        valid_predictions = predictions[noisy_mask.bool()]
        valid_labels = clean_ids[noisy_mask.bool()]
        count_weight = 1.0 / (get_num_microbatches() * parallel_state.get_data_parallel_world_size())
        decode_blocks = getattr(sample, "decode", None)
        valid_decode = (
            decode_blocks.repeat_interleave(block_size, dim=-1)[noisy_mask.bool()]
            if decode_blocks is not None else torch.zeros_like(valid_time, dtype=torch.bool)
        )
        # t=0 is a task label for corrupted decode blocks, not their physical noise.
        # Exclude those blocks from the existing denoising time-bin curves.
        metrics = _global_time_metrics(
            valid_time[~valid_decode], valid_losses[~valid_decode],
            valid_predictions[~valid_decode], valid_labels[~valid_decode], count_weight=count_weight,
        )
        if decode_blocks is not None:
            for name, selected in (("decode", valid_decode), ("denoise", ~valid_decode)):
                count = selected.sum().to(valid_losses.dtype)
                correct = ((valid_predictions == valid_labels) & selected).sum().to(valid_losses.dtype)
                report[f"{name}/loss"] = torch.stack((valid_losses[selected].sum().detach(), count))
                report[f"{name}/accuracy"] = torch.stack((correct, count))
                report[f"{name}/valid_tokens"] = torch.stack((count, count.new_tensor(count_weight)))
            decode_count = valid_decode.sum().float()
            report["decode/gamma"] = torch.stack((valid_gamma[valid_decode].sum(), decode_count))
            report["sampling/decode_token_fraction"] = torch.stack((decode_count, valid_decode.new_tensor(valid_decode.numel(), dtype=torch.float32)))
            active_blocks = torch.arange(blocks, device=device)[None, :] * block_size < target_lengths[:, None]
            report["sampling/decode_block_fraction"] = torch.stack((
                (decode_blocks & active_blocks).sum().float(), active_blocks.sum().float()
            ))
        eos_id = _scalar_batch_value(batch.get("eos_token_id"), default=-1)
        if "original_target_lengths" in batch and eos_id < 0:
            raise ValueError("Region metrics require the resolved EOS ID")
        report.update(target_region_metrics(noisy_losses.detach(), predictions, labels, original_lengths,
                                            target_lengths, eos_id=eos_id, count_weight=count_weight))
        count_denominator = valid_samples.float().new_tensor(count_weight)
        valid_tokens = loss_mask.sum().detach().to(torch.float)
        report["target_tokens"] = torch.stack((valid_tokens, count_denominator))
        report["target_samples"] = torch.stack((valid_samples.float(), count_denominator))
        if getattr(state.cfg.dataset, "prepared_manifest", None):
            report["data/context_tokens"] = torch.stack((context_lengths.sum().float(), count_denominator))
            report["data/physical_tokens"] = torch.stack((valid_tokens.new_tensor(batch_size * (context_width + 2 * target_width)), count_denominator))
            report["data/padding_tokens"] = torch.stack((valid_tokens.new_tensor(batch_size * (context_width + 2 * target_width)) - context_lengths.sum() - 2 * valid_tokens, count_denominator))
            source_ids = core.config.spherical_protocol_metadata["data_mixture"]["source_ids"]
            for source_index, source_name in enumerate(source_ids):
                selected = batch["source_id"] == source_index
                report[f"data/source/{source_name}/samples"] = torch.stack((selected.sum().float(), count_denominator))
                report[f"data/source/{source_name}/supervised_tokens"] = torch.stack(((loss_mask.sum(-1) * selected).sum(), count_denominator))
        for name, value in metrics.items():
            report[name] = value.detach()
        gamma_uniform = sample.gamma_uniform.repeat_interleave(block_size, dim=-1)[noisy_mask.bool()][~valid_decode]
        t_tokens = valid_time[~valid_decode]
        denoise_tokens = (~valid_decode).sum().float()
        report["sampling/t"] = torch.stack((t_tokens.float().sum(), denoise_tokens))
        report["sampling/gamma"] = torch.stack((valid_gamma[~valid_decode].float().sum(), denoise_tokens))
        valid_condition_input = block_condition.repeat_interleave(block_size, dim=-1)[noisy_mask.bool()]
        report["conditioning/input_mean"] = torch.stack((valid_condition_input.float().sum(), valid_tokens))
        report["sampling/gamma_uniform_fraction"] = torch.stack(
            (gamma_uniform.float().sum(), denoise_tokens)
        )
        report["sampling/t_uniform_fraction"] = torch.stack(
            (denoise_tokens - gamma_uniform.float().sum(), denoise_tokens)
        )
        return loss_sum, partial(_spherical_loss_callback, valid_samples, report)


__all__ = ["SphericalDGPTStep", "spherical_ce_loss"]
