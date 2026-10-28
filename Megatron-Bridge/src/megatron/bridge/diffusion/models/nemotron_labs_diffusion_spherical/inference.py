# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native spherical inference with MCore KV cache and TP top-k posterior."""

from __future__ import annotations

import torch
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.utils import unwrap_model

from .geometry import normalize, random_sphere, slerp, terminal_noise
from .decode_protocol import resolve_decode_endpoint
from .scheduler import gamma_from_t
from .conditioning import conditioning_input


def deterministic_topk(
    logits: torch.Tensor, k: int, *, deterministic: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k without full-vocabulary tie tensors.

    The default is the training fast path and returns ``torch.topk`` directly.
    Deterministic mode only reorders the returned candidates by token id before
    sorting by score; it intentionally does not resolve ties outside that
    candidate set at the top-k boundary.
    """
    if k <= 0 or k > logits.shape[-1]:
        raise ValueError("k must be within the vocabulary dimension")
    values, order = torch.topk(logits, k, dim=-1, largest=True, sorted=True)
    if not deterministic:
        return values, order
    # Tie handling is restricted to the k candidates returned by topk.
    id_order = torch.argsort(order, dim=-1, stable=True)
    values = values.gather(-1, id_order)
    order = order.gather(-1, id_order)
    value_order = torch.argsort(values, dim=-1, descending=True, stable=True)
    return values.gather(-1, value_order), order.gather(-1, value_order)


def _tp_details() -> tuple[object | None, int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None, 0, 1
    return (
        parallel_state.get_tensor_model_parallel_group(),
        parallel_state.get_tensor_model_parallel_rank(),
        parallel_state.get_tensor_model_parallel_world_size(),
    )


def tp_vocab_range(vocab_size: int) -> tuple[int, int]:
    """Return the exact global vocabulary range owned by the current TP rank."""
    _, rank, world = _tp_details()
    return tuple(VocabUtility.vocab_range_from_global_vocab_size(vocab_size, rank, world))


def merge_tp_topk(
    local_values: torch.Tensor,
    local_ids: torch.Tensor,
    *,
    vocab_offset: int | None = None,
    k: int = 512,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge local TP candidates with a low-memory top-k selection.

    Deterministic mode only sorts the already gathered TP candidates.
    """
    tp_group, _, tp_world_size = _tp_details()
    if vocab_offset is None:
        if tp_world_size != 1:
            raise ValueError("vocab_offset is required when merging TP vocabulary shards")
        vocab_offset = 0
    values = local_values
    ids = local_ids.long() + vocab_offset
    if tp_world_size > 1:
        gathered_values = [torch.empty_like(values) for _ in range(tp_world_size)]
        gathered_ids = [torch.empty_like(ids) for _ in range(tp_world_size)]
        torch.distributed.all_gather(gathered_values, values, group=tp_group)
        torch.distributed.all_gather(gathered_ids, ids, group=tp_group)
        values = torch.cat(gathered_values, dim=-1)
        ids = torch.cat(gathered_ids, dim=-1)
    if not deterministic:
        values, value_order = torch.topk(values, min(k, values.shape[-1]), dim=-1, sorted=True)
        return values, ids.gather(-1, value_order)
    id_order = torch.argsort(ids, dim=-1, stable=True)
    values = values.gather(-1, id_order)
    ids = ids.gather(-1, id_order)
    value_order = torch.argsort(values, dim=-1, descending=True, stable=True)
    value_order = value_order[..., : min(k, values.shape[-1])]
    return values.gather(-1, value_order), ids.gather(-1, value_order)


def distributed_x0_hat(
    local_logits: torch.Tensor,
    geometry,
    *,
    k: int = 512,
    deterministic: bool = False,
) -> torch.Tensor:
    """Compute the top-k posterior expectation after merging TP candidates.

    ``deterministic=False`` is used by the training rollout and keeps all
    temporary tensors bounded by the requested candidate count.
    """
    vocab_start, vocab_end = tp_vocab_range(geometry.vocab_size)
    expected_width = vocab_end - vocab_start
    if local_logits.shape[-1] != expected_width:
        raise ValueError(
            f"local logits width {local_logits.shape[-1]} does not match TP vocabulary {expected_width}"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    local_k = min(k, local_logits.shape[-1])
    local_values, local_ids = deterministic_topk(local_logits, local_k, deterministic=deterministic)
    values, global_ids = merge_tp_topk(
        local_values,
        local_ids,
        vocab_offset=vocab_start,
        k=k,
        deterministic=deterministic,
    )
    probabilities = torch.softmax(values.float(), dim=-1)
    expectation = torch.zeros(*global_ids.shape[:-1], geometry.hidden_size, device=local_logits.device, dtype=torch.float32)
    for start in range(0, global_ids.shape[-1], 64):
        codebook = geometry.codewords(global_ids[..., start:start + 64], device=local_logits.device, dtype=torch.float32)
        expectation.add_(torch.matmul(probabilities[..., start:start + 64].unsqueeze(-2), codebook).squeeze(-2))
    return normalize(expectation).to(local_logits.dtype)


class NativeKVCache:
    """Persistent MCore static context containing prompt and accepted clean blocks."""

    def __init__(self, *, batch_size: int, max_sequence_length: int):
        self.context = StaticInferenceContext(batch_size, max_sequence_length)
        self.context.config.materialize_only_last_token_logits = False

    @property
    def length(self) -> int:
        return int(self.context.sequence_len_offset)

    def advance(self, tokens: int) -> None:
        self.context.increment_sequence_len_offset(tokens)

    def fork(self) -> StaticInferenceContext:
        """Share storage but keep the speculative cursor separate from the committed prefix.

        Static KV writes only touch positions at/after the cursor. Denoising
        overwrites that uncommitted suffix; accepting a block overwrites it with
        clean KV before advancing the committed cursor. Calls are sequential.
        """
        scratch = StaticInferenceContext(self.context.max_batch_size, self.context.max_sequence_length)
        scratch.config.materialize_only_last_token_logits = False
        scratch.sequence_len_offset = self.context.sequence_len_offset
        scratch.batch_size_offset = self.context.batch_size_offset
        scratch.key_value_memory_dict.update(self.context.key_value_memory_dict)
        return scratch


def _cached_attention_mask(
    batch_size: int,
    prefix_length: int,
    query_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    query_bidirectional: bool,
) -> torch.Tensor:
    total = prefix_length + query_length
    mask = torch.zeros((batch_size, 1, query_length, total), device=device, dtype=dtype)
    if not query_bidirectional:
        future = torch.ones((query_length, query_length), device=device, dtype=torch.bool).triu(1)
        mask[:, :, :, prefix_length:].masked_fill_(future, torch.finfo(dtype).min)
    return mask


def _cached_forward(
    core,
    cache: StaticInferenceContext,
    decoder_input: torch.Tensor,
    token_ids: torch.Tensor,
    condition: torch.Tensor,
    *,
    query_bidirectional: bool,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, query_length = token_ids.shape
    prefix_length = int(cache.sequence_len_offset)
    if attention_mask is None:
        attention_mask = _cached_attention_mask(
            batch_size,
            prefix_length,
            query_length,
            device=decoder_input.device,
            dtype=decoder_input.dtype,
            query_bidirectional=query_bidirectional,
        )
    positions = torch.arange(prefix_length, prefix_length + query_length, device=token_ids.device)
    semantic_positions = positions.unsqueeze(0).expand(batch_size, -1)
    output = core(
        input_ids=token_ids,
        position_ids=semantic_positions,
        attention_mask=attention_mask,
        # MCore's decoder consumes [sequence, batch, hidden].
        decoder_input=decoder_input.transpose(0, 1).contiguous(),
        inference_context=cache,
        extra_block_kwargs={
            "spherical_time_condition": condition,
            "spherical_position_ids": semantic_positions,
        },
        runtime_gather_output=False,
    )
    output = output[0] if isinstance(output, tuple) else output
    # GPTModel already returns [batch, sequence, vocab] when labels are absent.
    return output.contiguous()


@torch.no_grad()
def spherical_block_sample(
    model,
    geometry,
    adapter,
    prompt_ids: torch.Tensor,
    *,
    gen_length: int,
    block_size: int,
    steps: int,
    mask_token_id: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    seed: int = 1234,
    t_min: float = 0.01,
    deterministic: bool = False,
    decode_endpoint: bool | None = None,
) -> tuple[torch.Tensor, int]:
    """Run reverse updates while committing only accepted clean blocks to KV cache.

    ``deterministic`` is intended for validation and diagnostics; training uses
    the default fast top-k path.

    ``steps`` is the number of model evaluations for each generated block.  The
    final evaluation decodes the block; preceding evaluations update the state.
    With decode training (or explicit ``decode_endpoint=True``), ``steps`` counts
    positive-time denoising forwards, followed by one extra t=0 decoding forward.
    The endpoint is then fixed at zero and the legacy ``t_min`` is not used.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [batch, positive_length]")
    if block_size <= 0 or gen_length <= 0 or gen_length % block_size:
        raise ValueError("gen_length must be a positive multiple of block_size")
    block_count = gen_length // block_size
    if steps <= 0:
        raise ValueError("steps must be positive per denoising block")
    if not 0.0 <= t_min < 1.0:
        raise ValueError("t_min must be in [0, 1)")
    if eos_token_id is not None and pad_token_id is None:
        raise ValueError("pad_token_id is required when eos_token_id is configured")
    generator = torch.Generator(device=prompt_ids.device)
    generator.manual_seed(seed)
    core = unwrap_model(model)
    core = getattr(core, "language_model", core)
    endpoint_decode = resolve_decode_endpoint(getattr(core, "config", None), decode_endpoint)
    time_conditioner = getattr(core, "time_conditioner", None)
    if time_conditioner is None:
        raise RuntimeError("native spherical inference requires time_conditioner")

    batch_size = prompt_ids.shape[0]
    cache = NativeKVCache(batch_size=batch_size, max_sequence_length=prompt_ids.shape[1] + gen_length)
    zeros = torch.zeros(prompt_ids.shape, device=prompt_ids.device, dtype=geometry.E_geometry.dtype)
    _cached_forward(
        core,
        cache.context,
        geometry.clean(prompt_ids),
        prompt_ids,
        time_conditioner(zeros),
        query_bidirectional=False,
    )
    cache.advance(prompt_ids.shape[1])

    output = prompt_ids.clone()
    finished = torch.zeros(batch_size, device=prompt_ids.device, dtype=torch.bool)
    steps_per_block = steps
    times = torch.linspace(
        1.0,
        0.0 if endpoint_decode else t_min,
        steps_per_block + 1 if endpoint_decode else steps_per_block,
        device=prompt_ids.device,
        dtype=torch.float32,
    )
    gamma_coordinates = gamma_from_t(times)
    gammas = gamma_coordinates.to(dtype=geometry.E_geometry.dtype)
    # Keep the original GEMM batch shape: changing it can change BF16 rounding.
    # Conditions still need to be computed only once for all generated blocks.
    coordinates = conditioning_input(
        times, gamma_coordinates, getattr(getattr(core, "config", None), "spherical_condition_coordinate", "t")
    )
    conditions = [time_conditioner(value.expand(batch_size, block_size)) for value in coordinates]
    zero_condition = time_conditioner(zeros.new_zeros((batch_size, block_size)))
    mask_direction = geometry.codeword(
        mask_token_id, device=prompt_ids.device, dtype=geometry.E_geometry.dtype
    ).view(1, 1, -1)
    vocab_start, _ = tp_vocab_range(geometry.vocab_size)
    nfe = 0
    for block_index in range(block_count):
        clean = torch.full(
            (batch_size, block_size), mask_token_id, device=prompt_ids.device, dtype=torch.long
        )
        state = terminal_noise(
            random_sphere(
                (batch_size, block_size, geometry.hidden_size),
                device=prompt_ids.device,
                dtype=geometry.E_geometry.dtype,
                generator=generator,
            ),
            mask_direction,
            generator=generator,
        )
        scratch = cache.fork()
        attention_mask = _cached_attention_mask(
            batch_size, cache.length, block_size, device=state.device,
            dtype=state.dtype, query_bidirectional=True,
        )
        final_logits = None
        for index in range(steps_per_block):
            final_logits = _cached_forward(
                core,
                scratch,
                adapter(state),
                clean,
                conditions[index],
                query_bidirectional=True,
                attention_mask=attention_mask,
            )
            nfe += 1
            if endpoint_decode or index + 1 < steps_per_block:
                x0_hat = distributed_x0_hat(
                    final_logits, geometry, k=512, deterministic=deterministic,
                )
                ratio = 1.0 - gammas[index + 1] / gammas[index].clamp_min(1e-6)
                state = slerp(state, x0_hat, ratio)

        if endpoint_decode:
            # The last denoising update reached gamma=0. Decode this still-soft
            # state once, without injecting noise or committing speculative KV.
            final_logits = _cached_forward(
                core, scratch, adapter(state), clean, conditions[-1],
                query_bidirectional=True, attention_mask=attention_mask,
            )
            nfe += 1

        local_logits = final_logits
        local_values, local_ids = deterministic_topk(local_logits, 1, deterministic=deterministic)
        _, global_ids = merge_tp_topk(local_values, local_ids, vocab_offset=vocab_start, k=1, deterministic=deterministic)
        clean = global_ids.squeeze(-1)
        if eos_token_id is not None:
            positions = torch.arange(block_size, device=clean.device).unsqueeze(0)
            eos_hits = clean == eos_token_id
            first_eos = torch.where(eos_hits, positions, block_size).amin(dim=-1, keepdim=True)
            clean = clean.masked_fill(finished.unsqueeze(-1) | (positions > first_eos), pad_token_id)
            finished = finished | eos_hits.any(dim=-1)

        output = torch.cat((output, clean), dim=1)
        if eos_token_id is not None and finished.all():
            remaining = gen_length - (output.shape[1] - prompt_ids.shape[1])
            if remaining:
                output = torch.cat((output, clean.new_full((batch_size, remaining), pad_token_id)), dim=1)
            break
        # No later query consumes the final accepted block's KV.
        if block_index + 1 < block_count:
            _cached_forward(
                core,
                cache.context,
                geometry.clean(clean),
                clean,
                zero_condition,
                query_bidirectional=False,
            )
            cache.advance(block_size)
    return output, nfe


__all__ = [
    "NativeKVCache",
    "deterministic_topk",
    "distributed_x0_hat",
    "merge_tp_topk",
    "spherical_block_sample",
    "tp_vocab_range",
]
