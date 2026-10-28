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

"""Euclidean reverse solver for RMS continuous Nemotron states."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from .attention import install_additive_attention, run_encoder_with_attention_mask
from .scheduler import IdentityScheduler
from .step import _posterior_clean_state


def _cached_context_length(prompt_ids: torch.Tensor, past_key_values) -> int:
    """Return accepted-prefix length without re-encoding cached tokens."""
    if past_key_values is None:
        return int(prompt_ids.shape[1])
    length = int(past_key_values.get_seq_length())
    if length <= 0:
        raise ValueError("past_key_values must contain the accepted context")
    return length


def euclidean_reverse_step(state, x0_hat, gamma, next_gamma, eps=1e-8):
    """Advance a linear-path state using the documented Euclidean velocity."""
    gamma = torch.as_tensor(gamma, device=state.device, dtype=state.dtype)
    next_gamma = torch.as_tensor(next_gamma, device=state.device, dtype=state.dtype)
    while gamma.ndim < state.ndim:
        gamma = gamma.unsqueeze(-1)
    while next_gamma.ndim < state.ndim:
        next_gamma = next_gamma.unsqueeze(-1)
    ratio = 1.0 - next_gamma / gamma.clamp_min(eps)
    updated = state + ratio * (x0_hat - state)
    return torch.where(gamma <= eps, x0_hat, updated)


def _allowed_vocab_mask(vocab_size, device, valid_token_ids, mask_token_id):
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    allowed[valid_token_ids.to(device=device, dtype=torch.long)] = True
    if mask_token_id is not None:
        allowed[int(mask_token_id)] = False
    return allowed


def _pad_after_eos(tokens, eos_token_id, pad_token_id):
    if eos_token_id is None or pad_token_id is None:
        return tokens
    out = tokens.clone()
    for row in range(out.shape[0]):
        eos = (out[row] == eos_token_id).nonzero(as_tuple=False)
        if eos.numel():
            out[row, int(eos[0]) + 1 :] = pad_token_id
    return out


@torch.no_grad()
def rms_block_sample(
    model,
    prompt_ids: torch.Tensor,
    *,
    block_size: int = 32,
    nfe: int = 64,
    t_min: float = 0.01,
    valid_token_ids: torch.Tensor | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
    posterior_mode: str = "topk",
    posterior_k: int = 512,
    past_key_values=None,
    return_cache: bool = False,
    return_trajectory: bool = False,
    terminal_seed: int | None = None,
    generator: torch.Generator | None = None,
    block_index: int = 0,
):
    """Generate one block with linear-path Euclidean updates.

    ``nfe`` includes the final logits-only decode forward.  Reverse updates
    use ``delta_gamma`` in RMS coordinates, never semantic-time differences.
    """
    base = model.base
    geometry = model.geometry
    embed = base.get_input_embeddings()
    device = prompt_ids.device
    dtype = embed.weight.dtype
    valid_token_ids = valid_token_ids if valid_token_ids is not None else model.valid_token_ids
    if valid_token_ids is None:
        raise ValueError("valid_token_ids (V_pred) is required")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if nfe < 2:
        raise ValueError("nfe must be >= 2 so the schedule includes an update and final decode")
    if not 0.0 <= t_min < 1.0:
        raise ValueError("t_min must be in [0, 1)")
    if posterior_mode not in {"topk", "full"}:
        raise ValueError("posterior_mode must be 'topk' or 'full'")
    if posterior_mode == "topk" and posterior_k < 1:
        raise ValueError("posterior_k must be >= 1")
    valid_token_ids = valid_token_ids.to(device=device, dtype=torch.long)
    if valid_token_ids.numel() == 0:
        raise ValueError("V_pred must contain at least one token")
    if valid_token_ids.unique().numel() != valid_token_ids.numel():
        raise ValueError("V_pred token ids must be unique")
    if int(valid_token_ids.min()) < 0 or int(valid_token_ids.max()) >= embed.num_embeddings:
        raise ValueError("V_pred contains a token id outside the embedding vocabulary")
    mask_token_id = model.mask_id if mask_token_id is None else mask_token_id
    for layer in base.encoder.layers:
        install_additive_attention(layer.self_attn)
    prompt_cache = past_key_values if past_key_values is not None else DynamicCache()
    if past_key_values is None:
        prompt = embed(prompt_ids).detach()
        positions = torch.arange(prompt.shape[1], device=device)
        model.set_gamma0_condition(prompt.shape[0], device)
        try:
            base.encoder(
                input_ids=prompt_ids,
                position_ids=positions.unsqueeze(0).expand(prompt.shape[0], -1),
                past_key_values=prompt_cache,
                use_cache=True,
                use_causal_mask=True,
                cache_position=positions,
            )
        finally:
            model.clear_gamma_condition()
        context_length = prompt.shape[1]
    else:
        # ``prompt_ids`` is the caller's full accepted sequence, while the
        # cache already contains that sequence.  Re-encoding it here would
        # append duplicate keys and shift every subsequent block position.
        context_length = _cached_context_length(prompt_ids, prompt_cache)
    batch_size = prompt_ids.shape[0]
    hidden_size = embed.weight.shape[-1]
    if generator is not None and terminal_seed is not None:
        raise ValueError("pass either generator or terminal_seed, not both")
    if generator is None and terminal_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(terminal_seed))
    state = geometry.sample_terminal(
        (batch_size, block_size, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    terminal_rms = float(state.square().mean(dim=-1).sqrt().mean())
    times = torch.linspace(1.0, t_min, nfe, device=device, dtype=torch.float32)
    scheduler = IdentityScheduler.for_terminal(geometry.terminal_mode)
    gammas = scheduler.gamma_from_t(times)
    final_logits = None
    trajectory = []
    for index, gamma in enumerate(gammas):
        noisy = geometry.to_model(state).to(dtype=dtype)
        total = context_length + block_size
        min_value = torch.finfo(dtype).min
        mask = torch.full((batch_size, 1, total, total), min_value, device=device, dtype=dtype)
        prompt_mask = torch.full((context_length, context_length), min_value, device=device, dtype=dtype)
        mask[:, :, :context_length, :context_length] = torch.tril(torch.zeros_like(prompt_mask)) + torch.triu(
            prompt_mask, diagonal=1
        )
        mask[:, :, context_length:, :context_length] = 0
        mask[:, :, context_length:, context_length:] = 0
        inputs = noisy.to(dtype=next(base.parameters()).dtype)
        pos = torch.cat(
            [
                (context_length + torch.arange(block_size, device=device)).expand(prompt_ids.shape[0], -1),
            ],
            dim=1,
        )
        model.set_gamma_condition(gamma.expand(batch_size, block_size).to(dtype=dtype), 0, block_size)
        try:
            hidden = run_encoder_with_attention_mask(
                base.encoder,
                inputs_embeds=inputs,
                attention_mask=mask[:, :, context_length:, :],
                position_ids=pos,
                past_key_values=prompt_cache,
                use_cache=False,
                cache_position=torch.arange(context_length, context_length + block_size, device=device),
                ada_condition=model._active_ada,
            ).last_hidden_state
        finally:
            model.clear_gamma_condition()
        logits = base.diffusion_head(hidden).float()
        allowed = _allowed_vocab_mask(logits.shape[-1], device, valid_token_ids, mask_token_id)
        filtered = logits.masked_fill(~allowed.view(1, 1, -1), -torch.inf)
        top_values, top1 = filtered.max(dim=-1)
        log_z = torch.logsumexp(filtered, dim=-1)
        top1_probability = (top_values - log_z).exp()
        posterior_entropy = torch.distributions.Categorical(logits=filtered).entropy()
        state_rms = state.square().mean(dim=-1).sqrt()
        state_l2 = state.norm(dim=-1)
        trajectory.append(
            {
                "block_index": int(block_index),
                "nfe_index": index,
                "t": float(times[index]),
                "gamma": float(gamma),
                "top1_token_id": top1.detach().cpu(),
                "top1_probability": top1_probability.detach().cpu(),
                "posterior_entropy": posterior_entropy.detach().cpu(),
                "x0_hat_cosine": float("nan"),
                "state_rms": float(state_rms.mean()),
                "state_l2": float(state_l2.mean()),
                "terminal_rms": terminal_rms,
            }
        )
        if index == nfe - 1:
            final_logits = filtered
            break
        x0_hat = _posterior_clean_state(
            logits, embed.weight, geometry.rms, valid_token_ids, mode=posterior_mode, k=posterior_k
        ).to(dtype=state.dtype)
        state = euclidean_reverse_step(state, x0_hat, gamma, gammas[index + 1])
    tokens = _pad_after_eos(final_logits.argmax(dim=-1), eos_token_id, pad_token_id)
    result = (tokens, final_logits)
    if return_cache:
        result += (prompt_cache,)
    if return_trajectory:
        result += (trajectory,)
    return result


__all__ = ["euclidean_reverse_step", "rms_block_sample"]
