"""Three-region attention mask for continuous block training."""

from types import SimpleNamespace

import torch


def continuous_position_ids(
    prompt_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    prompt_width: int,
    target_width: int,
) -> torch.Tensor:
    """Build semantic RoPE positions independent of batch padding widths."""
    device = prompt_lengths.device
    context_offset = torch.arange(prompt_width, device=device)[None, :]
    response_offset = torch.arange(target_width, device=device)[None, :]
    context = context_offset.expand(prompt_lengths.shape[0], -1)
    response = prompt_lengths[:, None] + response_offset
    context = torch.where(context_offset < prompt_lengths[:, None], context, 0)
    response = torch.where(response_offset < target_lengths[:, None], response, 0)
    return torch.cat([context, response, response], dim=1)


def continuous_block_attention_mask(prompt_length: int, target_length: int, block_size: int, device=None):
    """Return True for allowed attention edges in [context, clean, noisy] layout."""
    total = prompt_length + 2 * target_length
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    # Context is causal; clean response is causal over context + prior clean tokens.
    for q in range(prompt_length):
        mask[q, : q + 1] = True
    for q in range(target_length):
        clean_q = prompt_length + q
        noisy_q = prompt_length + target_length + q
        mask[clean_q, : prompt_length + q + 1] = True
        block_start = (q // block_size) * block_size
        mask[noisy_q, :prompt_length] = True
        mask[noisy_q, prompt_length : prompt_length + block_start] = True
        mask[noisy_q, prompt_length + target_length + block_start : prompt_length + 2 * target_length] = False
        mask[noisy_q, prompt_length + target_length + block_start : prompt_length + target_length + block_start + min(block_size, target_length - block_start)] = True
    return mask


def additive_block_attention_mask(
    prompt_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    prompt_width: int,
    target_width: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [batch, 1, total, total] additive masks for HF attention."""
    batch = prompt_lengths.numel()
    total = prompt_width + 2 * target_width
    result = torch.full((batch, 1, total, total), torch.finfo(dtype).min, device=prompt_lengths.device, dtype=dtype)
    for b in range(batch):
        p = int(prompt_lengths[b])
        r = int(target_lengths[b])
        allowed = continuous_block_attention_mask(p, r, block_size, device=prompt_lengths.device)
        # Map each sample into the fixed padded layout.  p/r only describe
        # valid rows and columns; they must not shift the clean/noisy segments.
        clean_start = prompt_width
        noisy_start = prompt_width + target_width
        result[b, 0, :p, :p] = torch.where(allowed[:p, :p], 0, torch.finfo(dtype).min)
        result[b, 0, clean_start : clean_start + r, :p] = torch.where(
            allowed[p : p + r, :p], 0, torch.finfo(dtype).min
        )
        result[b, 0, clean_start : clean_start + r, clean_start : clean_start + r] = torch.where(
            allowed[p : p + r, p : p + r], 0, torch.finfo(dtype).min
        )
        result[b, 0, noisy_start : noisy_start + r, :p] = torch.where(
            allowed[p + r : p + 2 * r, :p], 0, torch.finfo(dtype).min
        )
        result[b, 0, noisy_start : noisy_start + r, clean_start : clean_start + r] = torch.where(
            allowed[p + r : p + 2 * r, p : p + r], 0, torch.finfo(dtype).min
        )
        result[b, 0, noisy_start : noisy_start + r, noisy_start : noisy_start + r] = torch.where(
            allowed[p + r : p + 2 * r, p + r : p + 2 * r], 0, torch.finfo(dtype).min
        )
        # Padding queries remain finite and can only read the first valid key.
        first_key = 0
        result[b, 0, :prompt_width, first_key] = torch.where(
            torch.arange(prompt_width, device=prompt_lengths.device) < p, result[b, 0, :prompt_width, first_key], 0
        )
        result[b, 0, clean_start + r : noisy_start, first_key] = 0
        result[b, 0, noisy_start + r :, first_key] = 0
    return result


def run_encoder_with_attention_mask(
    encoder,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values=None,
    use_cache: bool = False,
    cache_position: torch.Tensor | None = None,
):
    """Run the checkpoint encoder without letting it re-impose a causal mask.

    The bundled checkpoint's encoder converts ``use_causal_mask=True`` into a
    triangular mask and drops a custom mask when it is false.  Continuous
    training needs a non-causal noisy block, so call decoder layers directly
    and pass the already-built additive mask through unchanged.
    """
    if cache_position is None:
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_len, past_len + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    position_embeddings = encoder.rotary_emb(inputs_embeds, position_ids=position_ids)
    hidden_states = inputs_embeds
    for layer in encoder.layers[: encoder.config.num_hidden_layers]:
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    hidden_states = encoder.norm(hidden_states)
    return SimpleNamespace(last_hidden_state=hidden_states, past_key_values=past_key_values)
