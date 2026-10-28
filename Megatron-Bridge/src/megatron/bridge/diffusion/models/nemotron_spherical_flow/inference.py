"""Reference continuous reverse sampler for Nemotron spherical flow."""

import torch
from transformers.cache_utils import DynamicCache

from .attention import run_encoder_with_attention_mask
from .geometry import normalize, random_sphere, slerp, terminal_noise
from .scheduler import gamma_from_t


def token_lengths_through_eos(tokens: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """Count emitted tokens through the first EOS without changing padded blocks."""
    lengths = torch.full(
        tokens.shape[:-1], tokens.shape[-1], dtype=torch.long, device=tokens.device
    )
    if eos_token_id is None:
        return lengths
    eos = tokens.eq(eos_token_id)
    return torch.where(eos.any(dim=-1), eos.long().argmax(dim=-1) + 1, lengths)


@torch.no_grad()
def spherical_block_sample(
    model,
    prompt_ids: torch.Tensor,
    mask_direction: torch.Tensor,
    block_size: int = 32,
    nfe: int = 64,
    iterations: int | None = None,
    t_min: float = 0.01,
    valid_token_ids: torch.Tensor | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
    special_token_ids: torch.Tensor | None = None,
    past_key_values=None,
    return_cache: bool = False,
):
    """Generate one block using the same conditioned three-part model as training."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    if iterations is not None and iterations <= 0:
        raise ValueError("iterations must be positive")
    base = getattr(model, "base", model)
    adapter = getattr(model, "adapter", None)
    if adapter is None:
        raise ValueError("model must expose .adapter")
    if valid_token_ids is None:
        valid_token_ids = getattr(model, "valid_token_ids", None)
    if valid_token_ids is None:
        raise ValueError("valid_token_ids (fixed V_id) is required for spherical decoding")
    full_vocab = torch.arange(embed_vocab := base.get_input_embeddings().num_embeddings, device=valid_token_ids.device)
    if valid_token_ids.numel() != embed_vocab or not torch.equal(
        valid_token_ids.to(device=full_vocab.device, dtype=torch.long), full_vocab
    ):
        raise ValueError("spherical decoding requires the complete vocabulary V_id")
    device = prompt_ids.device
    embed = base.get_input_embeddings()
    dtype = embed.weight.dtype
    for layer in base.encoder.layers:
        layer.self_attn.diffusion_lm = False
    prompt = embed(prompt_ids).detach()
    # Reuse the checkpoint's native DynamicCache path: prompt KV is prefetched
    # once, while every denoising query reads it without updating the cache.
    prompt_cache = past_key_values if past_key_values is not None else DynamicCache()
    if past_key_values is None:
        prompt_positions = torch.arange(prompt.shape[1], device=device)
        if hasattr(model, "set_gamma0_condition"):
            model.set_gamma0_condition(prompt.shape[0], device)
        try:
            base.encoder(
                input_ids=prompt_ids,
                position_ids=prompt_positions.unsqueeze(0).expand(prompt.shape[0], -1),
                past_key_values=prompt_cache,
                use_cache=True,
                use_causal_mask=True,
                cache_position=prompt_positions,
            )
        finally:
            if hasattr(model, "clear_gamma_condition"):
                model.clear_gamma_condition()
    # The caller may append completed blocks to the same cache between calls.
    # Current queries must therefore start after the full cached prefix, not
    # after the original prompt alone.
    past_len = prompt_cache.get_seq_length()
    state = terminal_noise(random_sphere((prompt_ids.shape[0], block_size, prompt.shape[-1]), device=device, dtype=dtype), mask_direction)
    # One node is one model evaluation. The final node only decodes, so an NFE
    # budget of N performs N-1 reverse updates and one final forward.
    iterations = iterations if iterations is not None else max(1, nfe)
    times = torch.linspace(1.0, t_min, iterations, device=device)
    gammas = gamma_from_t(times).to(dtype)
    final_logits = None
    for index in range(iterations):
        noisy = adapter(state)
        total = past_len + block_size
        mask = torch.full((prompt.shape[0], 1, total, total), torch.finfo(dtype).min, device=device, dtype=dtype)
        prefix_mask = torch.full((past_len, past_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask[:, :, :past_len, :past_len] = torch.tril(torch.zeros_like(prefix_mask)) + torch.triu(prefix_mask, diagonal=1)
        mask[:, :, past_len:, :past_len] = 0
        mask[:, :, past_len:, past_len:] = 0
        # Prompt and noisy block share the same response position convention as
        # training; the clean response is represented by the denoising target.
        inputs = torch.cat([prompt, noisy], dim=1).to(dtype=next(base.parameters()).dtype)
        position_ids = (past_len + torch.arange(block_size, device=device)).expand(prompt.shape[0], -1)
        cond_gamma = gammas[index].expand(prompt.shape[0], block_size)
        if hasattr(model, "set_gamma_condition"):
            model.set_gamma_condition(cond_gamma, 0, block_size)
        try:
            hidden = run_encoder_with_attention_mask(
                base.encoder,
                inputs_embeds=inputs[:, -block_size:],
                attention_mask=mask[:, :, -block_size:, :],
                position_ids=position_ids[:, -block_size:],
                past_key_values=prompt_cache,
                use_cache=False,
                cache_position=torch.arange(
                    past_len, past_len + block_size, device=device
                ),
            ).last_hidden_state[:, -block_size:]
        finally:
            if hasattr(model, "clear_gamma_condition"):
                model.clear_gamma_condition()
        logits = base.diffusion_head(hidden).float()
        if index == iterations - 1:
            final_logits = logits
            break
        allowed = _allowed_vocab_mask(logits.shape[-1], device, valid_token_ids, mask_token_id, special_token_ids)
        filtered_logits = logits.masked_fill(~allowed.view(1, 1, -1), torch.finfo(logits.dtype).min)
        top_values, top_indices = deterministic_topk(filtered_logits, min(512, filtered_logits.shape[-1]))
        probs = torch.softmax(top_values, dim=-1)
        code = adapter.codebook[top_indices].float()
        x0_hat = normalize((probs.unsqueeze(-1) * code).sum(dim=-2)).to(dtype)
        ratio = 1.0 - gammas[index + 1] / gammas[index].clamp_min(1e-6)
        state = slerp(state, x0_hat, ratio)
    allowed = _allowed_vocab_mask(final_logits.shape[-1], device, valid_token_ids, mask_token_id, special_token_ids)
    final_logits = final_logits.masked_fill(~allowed.view(1, 1, -1), torch.finfo(final_logits.dtype).min)
    tokens = _pad_after_eos(final_logits.argmax(dim=-1), eos_token_id, pad_token_id)
    if return_cache:
        return tokens, final_logits, prompt_cache
    return tokens, final_logits


def _allowed_vocab_mask(vocab_size, device, valid_token_ids, mask_token_id, special_token_ids):
    """Return the fixed inference vocabulary mask.

    Spherical Nemotron uses the complete model vocabulary for ``V_id``.  The
    mask token and other special ids remain legal outputs; EOS termination and
    padding are handled after argmax.  The legacy arguments are retained for
    caller compatibility but intentionally do not remove logits.
    """
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if valid_token_ids is not None:
        ids = valid_token_ids.to(device=device, dtype=torch.long)
        expected = torch.arange(vocab_size, device=device)
        if ids.numel() != vocab_size or not torch.equal(ids, expected):
            raise ValueError("spherical decoding requires the complete vocabulary V_id")
    return allowed


def deterministic_topk(logits: torch.Tensor, k: int):
    """Return top-k logits with smaller token ids winning exact ties."""
    if k <= 0 or k > logits.shape[-1]:
        raise ValueError("k must be in the logits vocabulary range")
    # Stable sorting preserves the original ascending token-id order for ties.
    order = torch.argsort(-logits, dim=-1, stable=True)[..., :k]
    return logits.gather(-1, order), order


def _pad_after_eos(tokens: torch.Tensor, eos_token_id: int | None, pad_token_id: int | None):
    if eos_token_id is None or pad_token_id is None:
        return tokens
    result = tokens.clone()
    for row in range(result.shape[0]):
        eos = (result[row] == eos_token_id).nonzero(as_tuple=False)
        if eos.numel():
            result[row, int(eos[0]) + 1 :] = pad_token_id
    return result
