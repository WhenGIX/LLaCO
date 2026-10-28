"""Fixed-budget Plan 3 posterior solver."""
from __future__ import annotations
import math
import torch
from transformers.cache_utils import DynamicCache

from .attention import install_additive_attention, run_encoder_with_attention_mask
from .noise import add_observation_noise, observation_sigma

SOLVER_STEPS_PER_BLOCK = 32
SOLVER_UPDATE_COUNT = 32
SOLVER_NFE = 33
FINAL_T = 0.01
FINAL_NODE_INDEX = round(FINAL_T * 999)
NODE_SELECTOR_VERSION = "endpoint_round_half_up_final_t001_v2"

def node_indices(steps_per_block: int | None = None):
    if steps_per_block is not None and int(steps_per_block) != SOLVER_STEPS_PER_BLOCK:
        raise ValueError("Plan 3 fixes steps_per_block=32")
    n = SOLVER_NFE
    span = 999 - FINAL_NODE_INDEX
    return tuple(
        FINAL_NODE_INDEX + math.floor(((n - 1 - q) * span + (n - 1) / 2) / (n - 1))
        for q in range(n)
    )

def posterior_clean_state(logits, embedding_weight, rms, valid_ids, *, mode="full", k=None):
    ids = torch.as_tensor(valid_ids, device=logits.device, dtype=torch.long)
    selected = logits.float().index_select(-1, ids)
    if mode == "full":
        probs = torch.softmax(selected, dim=-1)
        return probs @ (embedding_weight.index_select(0, ids).float() / float(rms))
    if mode != "topk" or k is None or int(k) <= 0:
        raise ValueError("posterior must be full or topk with positive k")
    values, order = selected.topk(min(int(k), ids.numel()), dim=-1)
    probs = torch.softmax(values, dim=-1)
    rows = embedding_weight.index_select(0, ids.index_select(0, order.reshape(-1))).float().reshape(*order.shape, -1) / float(rms)
    return (probs.unsqueeze(-1) * rows).sum(dim=-2)

def reverse_update(state, x0_hat, gamma, next_gamma, eps=1e-8):
    g = torch.as_tensor(gamma, device=state.device, dtype=state.dtype)
    ng = torch.as_tensor(next_gamma, device=state.device, dtype=state.dtype)
    while g.ndim < state.ndim: g, ng = g.unsqueeze(-1), ng.unsqueeze(-1)
    return torch.where(g <= eps, x0_hat, state + (1 - ng / g.clamp_min(eps)) * (x0_hat - state))


def _allowed_vocab_logits(logits, valid_ids):
    allowed = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
    allowed[torch.as_tensor(valid_ids, device=logits.device, dtype=torch.long)] = True
    return logits.masked_fill(~allowed.view(1, 1, -1), -torch.inf)


def _pad_after_eos(tokens, eos_token_id, pad_token_id):
    if eos_token_id is None or pad_token_id is None:
        return tokens
    result = tokens.clone()
    for row in range(result.shape[0]):
        eos = (result[row] == eos_token_id).nonzero(as_tuple=False)
        if eos.numel():
            result[row, int(eos[0]) + 1 :] = pad_token_id
    return result


def _cache_prompt(model, prompt_ids, past_key_values):
    base = model.base
    if past_key_values is not None:
        length = int(past_key_values.get_seq_length())
        if length <= 0:
            raise ValueError("cached Plan 3 prompt must contain accepted context")
        return past_key_values, length
    cache = DynamicCache()
    positions = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
    model.set_gamma0_condition(prompt_ids.shape[0], prompt_ids.device)
    try:
        base.encoder(
            inputs_embeds=torch.nn.functional.embedding(prompt_ids, model.frozen_embedding),
            position_ids=positions.unsqueeze(0).expand(prompt_ids.shape[0], -1),
            past_key_values=cache,
            use_cache=True,
            use_causal_mask=True,
            cache_position=positions,
        )
    finally:
        model.clear_gamma_condition()
    return cache, int(prompt_ids.shape[1])


def commit_clean_block(model, tokens, cache, start_position):
    """Append accepted clean tokens after, never during, solver updates."""
    positions = torch.arange(start_position, start_position + tokens.shape[1], device=tokens.device)
    model.set_gamma0_condition(tokens.shape[0], tokens.device)
    try:
        model.base.encoder(
            inputs_embeds=torch.nn.functional.embedding(tokens, model.frozen_embedding),
            position_ids=positions.unsqueeze(0).expand(tokens.shape[0], -1),
            past_key_values=cache,
            use_cache=True,
            use_causal_mask=True,
            cache_position=positions,
        )
    finally:
        model.clear_gamma_condition()


@torch.no_grad()
def plan3_block_sample(
    model,
    prompt_ids,
    schedule,
    *,
    mode,
    inference_mode,
    seed,
    block_size=32,
    eos_token_id=None,
    pad_token_id=None,
    posterior_mode="full",
    posterior_k=None,
    past_key_values=None,
    block_index=0,
):
    """Generate one 32-token block with the fixed Plan 3 solver contract."""
    if block_size != SOLVER_STEPS_PER_BLOCK:
        raise ValueError("Plan 3 fixes block_size=32")
    if inference_mode not in {"clean", "noisy"}:
        raise ValueError("inference_mode must be clean or noisy")
    base = model.base
    embedding = model.frozen_embedding
    for layer in base.encoder.layers:
        install_additive_attention(layer.self_attn)
    cache, context_length = _cache_prompt(model, prompt_ids, past_key_values)
    state = model.geometry.mask_state.to(prompt_ids.device, torch.float32).expand(
        prompt_ids.shape[0], block_size, -1
    ).clone()
    rows = schedule.lookup_indices(torch.tensor(node_indices(), device="cpu"))
    generator = None
    if inference_mode == "noisy":
        generator = torch.Generator(device=state.device).manual_seed(int(seed))
    trajectory = []
    final_logits = None
    for index in range(SOLVER_NFE):
        row = {key: value[index] for key, value in rows.items() if key != "role"}
        t = row["t"].to(state.device, dtype=state.dtype)
        gamma = row["gamma"].to(state.device, dtype=state.dtype)
        sigma10 = row["sigma_10"].to(state.device, dtype=state.dtype)
        observation = state if inference_mode == "clean" else add_observation_noise(
            state, mode, t, sigma10, generator=generator
        )
        positions = torch.arange(context_length, context_length + block_size, device=state.device)
        attention = torch.zeros(
            (state.shape[0], 1, block_size, context_length + block_size),
            device=state.device,
            dtype=embedding.dtype,
        )
        model.set_gamma_condition(t.expand(state.shape[0], block_size), 0, block_size)
        try:
            hidden = run_encoder_with_attention_mask(
                base.encoder,
                model.geometry.to_model(observation).to(dtype=embedding.dtype),
                attention,
                positions.unsqueeze(0).expand(state.shape[0], -1),
                past_key_values=cache,
                use_cache=False,
                cache_position=positions,
                ada_condition=model._active_ada,
            ).last_hidden_state
        finally:
            model.clear_gamma_condition()
        logits = _allowed_vocab_logits(base.diffusion_head(hidden).float(), model.valid_token_ids)
        trajectory.append(
            {
                "block_index": int(block_index),
                "node_index": int(rows["index"][index]),
                "t": float(t),
                "gamma": float(gamma),
                "sigma_10": float(sigma10),
                "sigma_obs": float(observation_sigma(mode, t, sigma10)),
                "inference_mode": inference_mode,
            }
        )
        if index == SOLVER_UPDATE_COUNT:
            final_logits = logits
            break
        x0_hat = posterior_clean_state(
            logits,
            embedding,
            model.geometry.rms,
            model.valid_token_ids,
            mode=posterior_mode,
            k=posterior_k,
        )
        state = reverse_update(state, x0_hat.to(state.dtype), gamma, rows["gamma"][index + 1].to(state))
    if final_logits is None:
        raise RuntimeError("Plan 3 solver did not execute the required final t=0.01 decode")
    return _pad_after_eos(final_logits.argmax(dim=-1), eos_token_id, pad_token_id), cache, trajectory

@torch.no_grad()
def solve_block(model_forward, initial_state, schedule, embedding_weight, rms, valid_ids, *, mode, inference_mode="clean", posterior_mode="full", posterior_k=None, seed=None, block_index=0):
    if inference_mode not in {"clean", "noisy"}:
        raise ValueError("inference_mode must be clean or noisy")
    nodes = node_indices()
    rows = schedule.lookup_indices(torch.tensor(nodes))
    state = initial_state.clone()
    generator = None
    if inference_mode == "noisy":
        generator = torch.Generator(device=state.device).manual_seed(int(seed))
    trajectory = []
    final_logits = None
    for q in range(SOLVER_NFE):
        t, gamma, sigma10 = rows["t"][q], rows["gamma"][q], rows["sigma_10"][q]
        obs = state if inference_mode == "clean" else add_observation_noise(state, mode, t.to(state), sigma10.to(state), generator=generator)
        logits = model_forward(schedule.lookup_indices(torch.tensor([nodes[q]])), obs)
        trajectory.append({"block_index": int(block_index), "node_index": int(nodes[q]), "t": float(t), "gamma": float(gamma), "sigma_10": float(sigma10), "sigma_obs": float(observation_sigma(mode, t, sigma10)), "inference_mode": inference_mode})
        if q == SOLVER_UPDATE_COUNT:
            final_logits = logits
            break
        x0_hat = posterior_clean_state(logits, embedding_weight, rms, valid_ids, mode=posterior_mode, k=posterior_k)
        state = reverse_update(state, x0_hat.to(state.dtype), gamma, rows["gamma"][q + 1])
    return final_logits, state, trajectory

__all__ = ["FINAL_T", "NODE_SELECTOR_VERSION", "SOLVER_NFE", "SOLVER_STEPS_PER_BLOCK", "SOLVER_UPDATE_COUNT", "commit_clean_block", "node_indices", "observation_sigma", "plan3_block_sample", "posterior_clean_state", "reverse_update", "solve_block"]
