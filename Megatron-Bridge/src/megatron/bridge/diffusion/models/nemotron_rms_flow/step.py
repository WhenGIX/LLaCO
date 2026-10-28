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

"""HF Nemotron training wrapper for RMS continuous denoising."""

from __future__ import annotations

import types
import os

import torch
import torch.nn.functional as F
from torch import nn

from .attention import install_additive_attention, run_encoder_with_attention_mask
from .conditioning import GammaConditioner
from .geometry import RMSGeometry


def _posterior_clean_state(logits, embedding_weight, rms, valid_ids, mode="topk", k=512):
    """Compute a posterior clean state without normalizing the result."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    ids = valid_ids.to(device=flat.device, dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("V_pred must contain at least one token")
    if ids.unique().numel() != ids.numel():
        raise ValueError("V_pred token ids must be unique")
    if int(ids.min()) < 0 or int(ids.max()) >= flat.shape[-1]:
        raise ValueError("V_pred contains a token id outside the logits vocabulary")
    selected_logits = flat.index_select(-1, ids)
    if mode == "full":
        probs = torch.softmax(selected_logits, dim=-1)
        chunks = []
        for start in range(0, ids.numel(), 32768):
            rows = embedding_weight[ids[start : start + 32768]].float() / rms.float()
            chunks.append(probs[:, start : start + rows.shape[0]] @ rows)
        return torch.stack(chunks, dim=0).sum(dim=0).reshape(*logits.shape[:-1], -1)
    if mode != "topk":
        raise ValueError("posterior mode must be 'topk' or 'full'")
    if int(k) <= 0:
        raise ValueError("posterior k must be positive")
    k = min(int(k), ids.numel())
    values, order = selected_logits.topk(k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    chosen = ids[order]
    state = embedding_weight[chosen].float() / rms.float()
    return (probs.unsqueeze(-1) * state).sum(dim=-2).reshape(*logits.shape[:-1], -1)


def _bounded_x0_cosine(logits, clean_state, valid, embedding_weight, rms, valid_ids, limit=64, k=512):
    result = torch.full(valid.shape, torch.nan, device=logits.device, dtype=torch.float32)
    selected = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if selected.numel() == 0:
        return result
    if selected.numel() > limit:
        offsets = torch.linspace(0, selected.numel() - 1, limit, device=selected.device).long()
        selected = selected[offsets]
    posterior = _posterior_clean_state(
        logits.flatten(0, 1)[selected], embedding_weight, rms, valid_ids, mode="topk", k=k
    )
    targets = clean_state.flatten(0, 1)[selected]
    result.view(-1)[selected] = F.cosine_similarity(posterior.float(), targets.float(), dim=-1)
    return result


def _terminal_mask_top1(terminal, embedding_weight, mask_id, valid, limit=1, chunk_size=32768):
    selected = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if selected.numel() == 0:
        zero = terminal.new_zeros((), dtype=torch.float32)
        return zero, zero
    selected = selected[:limit]
    state = F.normalize(terminal.flatten(0, 1)[selected].float(), dim=-1)
    top_score = torch.full((selected.numel(),), -torch.inf, device=terminal.device)
    top_id = torch.zeros((selected.numel(),), dtype=torch.long, device=terminal.device)
    for start in range(0, embedding_weight.shape[0], chunk_size):
        code = F.normalize(embedding_weight[start : start + chunk_size].float(), dim=-1)
        score, offset = (state @ code.T).max(dim=-1)
        replace = score > top_score
        top_score = torch.where(replace, score, top_score)
        top_id = torch.where(replace, offset + start, top_id)
    return top_id.eq(mask_id).sum().float(), selected.new_tensor(selected.numel(), dtype=torch.float32)


class RMSContinuousDenoiser(nn.Module):
    """Nemotron encoder with direct continuous embedding inputs."""

    def __init__(
        self,
        base,
        mask_id: int,
        *,
        terminal_mode="scale-matched",
        condition_dim=1024,
        vocab_rms_ids=None,
        statistics_manifest=None,
    ):
        super().__init__()
        self.base = base
        self.mask_id = int(mask_id)
        embedding = base.get_input_embeddings()
        embedding.weight.requires_grad_(False)
        if statistics_manifest is not None:
            statistics_manifest.validate_embedding(embedding.weight, self.mask_id)
            vocab_rms_ids = statistics_manifest.rms_ids(device=embedding.weight.device)
        self.geometry = RMSGeometry.from_embedding(
            embedding.weight,
            self.mask_id,
            vocab_ids=vocab_rms_ids,
            terminal_mode=terminal_mode,
            b_noise=statistics_manifest.b_noise if statistics_manifest is not None else None,
        )
        self.valid_token_ids = (
            statistics_manifest.prediction_ids()
            if statistics_manifest is not None
            else torch.tensor([i for i in range(embedding.num_embeddings) if i != self.mask_id], dtype=torch.long)
        )
        self.statistics_manifest = statistics_manifest
        for layer in self.base.encoder.layers:
            install_additive_attention(layer.self_attn)
        self.gamma_conditioner = GammaConditioner(
            base.config.hidden_size, condition_dim=condition_dim, layers=len(self.base.encoder.layers)
        ).to(dtype=embedding.weight.dtype)
        self._active_ada = None
        self._install_ada_layers()

    def _install_ada_layers(self):
        for layer_idx, layer in enumerate(self.base.encoder.layers):
            layer._rms_ada_idx = layer_idx

            def ada_forward(layer_self, hidden_states, *args, **kwargs):
                # Pass the condition through the checkpoint closure. Reading
                # mutable module state here would make backward recomputation
                # observe the cleared condition and produce a different graph.
                active = kwargs.pop("_rms_ada", None)
                if active is None:
                    active = self._active_ada
                if active is None:
                    return layer_self._rms_original_forward(hidden_states, *args, **kwargs)
                layer_idx = layer_self._rms_ada_idx
                noisy_ada, zero_ada, start, end = active
                shift_0a, scale_0a, gate_0a, shift_0m, scale_0m, gate_0m = zero_ada[layer_idx]
                residual = hidden_states
                normed = layer_self.input_layernorm(hidden_states) * (1 + scale_0a) + shift_0a
                if noisy_ada is not None:
                    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = noisy_ada[layer_idx]
                    mod = layer_self.input_layernorm(hidden_states[:, start:end]) * (1 + scale_a) + shift_a
                    normed = torch.cat([normed[:, :start], mod, normed[:, end:]], dim=1)
                attn_out, _ = layer_self.self_attn(hidden_states=normed, **kwargs)
                hidden_states = residual + (1 + gate_0a) * attn_out
                if noisy_ada is not None:
                    mod = residual[:, start:end] + (1 + gate_a) * attn_out[:, start:end]
                    hidden_states = torch.cat([hidden_states[:, :start], mod, hidden_states[:, end:]], dim=1)
                residual = hidden_states
                normed = layer_self.post_attention_layernorm(hidden_states) * (1 + scale_0m) + shift_0m
                if noisy_ada is not None:
                    mod = layer_self.post_attention_layernorm(hidden_states[:, start:end]) * (1 + scale_m) + shift_m
                    normed = torch.cat([normed[:, :start], mod, normed[:, end:]], dim=1)
                mlp_out = layer_self.mlp(normed)
                hidden_states = residual + (1 + gate_0m) * mlp_out
                if noisy_ada is not None:
                    mod = residual[:, start:end] + (1 + gate_m) * mlp_out[:, start:end]
                    hidden_states = torch.cat([hidden_states[:, :start], mod, hidden_states[:, end:]], dim=1)
                return hidden_states

            layer._rms_original_forward = layer.forward
            layer.forward = types.MethodType(ada_forward, layer)

    def set_gamma_condition(self, gamma, noisy_start, total_length):
        noisy_projected = self.gamma_conditioner(gamma)
        zero_gamma = torch.zeros((gamma.shape[0], 1), device=gamma.device, dtype=gamma.dtype)
        self._active_ada = (
            noisy_projected,
            self.gamma_conditioner(zero_gamma),
            noisy_start,
            noisy_start + gamma.shape[1],
        )

    def set_gamma0_condition(self, batch_size, device):
        zero_gamma = torch.zeros((batch_size, 1), device=device)
        self._active_ada = (None, self.gamma_conditioner(zero_gamma), 0, 0)

    def clear_gamma_condition(self):
        self._active_ada = None

    def forward(self, context_ids, clean_ids, labels, attention_mask, position_ids, gamma, target_lengths):
        embed = self.base.get_input_embeddings()
        context = embed(context_ids).detach()
        clean_model = embed(clean_ids).detach()
        clean_state = clean_model.float() / self.geometry.rms.float().to(clean_model.device)
        terminal = self.geometry.sample_terminal(
            tuple(clean_state.shape), dtype=clean_state.dtype, device=clean_state.device
        )
        state = self.geometry.path(clean_state, terminal, gamma)
        noisy = self.geometry.to_model(state).to(dtype=clean_model.dtype)
        # Preserve the exact checkpoint embedding at the clean endpoint.  The
        # explicit branch avoids a needless float32 divide/multiply round trip.
        clean_endpoint = gamma.to(device=clean_model.device).eq(0)
        while clean_endpoint.ndim < noisy.ndim:
            clean_endpoint = clean_endpoint.unsqueeze(-1)
        noisy = torch.where(clean_endpoint, clean_model, noisy)
        inputs = torch.cat([context, clean_model, noisy], dim=1).to(dtype=next(self.base.parameters()).dtype)
        noisy_start = context.shape[1] + clean_model.shape[1]
        self.set_gamma_condition(gamma, noisy_start, inputs.shape[1])
        ada_condition = self._active_ada
        try:
            hidden = run_encoder_with_attention_mask(
                self.base.encoder,
                inputs,
                attention_mask,
                position_ids,
                ada_condition=ada_condition,
            )
        finally:
            self.clear_gamma_condition()
        logits = self.base.diffusion_head(hidden.last_hidden_state[:, noisy_start:]).float()
        flat_logits = logits.flatten(0, 1)
        flat_labels = labels.flatten()
        token_count = flat_labels.ne(-100).sum()
        # V_pred is the only vocabulary used for clean-token prediction.  Keep
        # the mask token (and any statistics exclusions) out of CE as well as
        # posterior/decode diagnostics; otherwise the model could spend
        # probability mass on a token that inference can never emit.
        sample_ids = self.valid_token_ids.to(device=logits.device)
        prediction_allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        prediction_allowed[sample_ids] = True
        valid_labels = flat_labels.ne(-100)
        if valid_labels.any() and not bool(prediction_allowed[flat_labels[valid_labels]].all()):
            excluded = flat_labels[valid_labels][~prediction_allowed[flat_labels[valid_labels]]]
            raise ValueError(
                f"training labels contain tokens outside V_pred; first invalid token id is {int(excluded[0])}"
            )
        allowed_logits = flat_logits.masked_fill(~prediction_allowed.unsqueeze(0), -torch.inf)
        per_token_loss = F.cross_entropy(allowed_logits, flat_labels, ignore_index=-100, reduction="none").view_as(
            labels
        )
        token_loss = per_token_loss.sum()
        valid = labels.ne(-100)
        with torch.no_grad():
            _, top_ids = allowed_logits.detach().topk(min(5, sample_ids.numel()), dim=-1)
            predictions = top_ids[:, 0].view_as(labels)
            correct = predictions.eq(labels) & valid
            top5_correct = top_ids.view(*labels.shape, -1).eq(labels.unsqueeze(-1)).any(dim=-1) & valid
            entropy = torch.distributions.Categorical(logits=allowed_logits.detach()).entropy().view_as(labels)
            # Posterior diagnostics use cosine only for comparison; x0_hat itself is
            # kept in RMS coordinates and is never normalized.
            x0_cosine = _bounded_x0_cosine(
                logits.detach(), clean_state, valid, embed.weight, self.geometry.rms, sample_ids
            )
        edges = logits.new_tensor([0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0001])
        state_rms = state.float().square().mean(dim=-1).sqrt()
        gamma_stats = torch.zeros(7, 8, device=logits.device, dtype=torch.float32)
        for index in range(7):
            in_bin = (gamma >= edges[index]) & (gamma < edges[index + 1]) & valid
            cosine_valid = in_bin & torch.isfinite(x0_cosine)
            gamma_stats[index] = torch.stack(
                [
                    per_token_loss[in_bin].sum(),
                    entropy[in_bin].sum(),
                    correct[in_bin].sum().float(),
                    top5_correct[in_bin].sum().float(),
                    x0_cosine[cosine_valid].sum(),
                    state_rms[in_bin].sum(),
                    in_bin.sum().float(),
                    cosine_valid.sum().float(),
                ]
            )
        state_l2 = state.float().norm(dim=-1)
        terminal_rms = terminal.float().square().mean(dim=-1).sqrt()
        mask_state = self.geometry.mask_state.to(device=terminal.device)
        terminal_mask_cosine = F.cosine_similarity(terminal.float(), mask_state.view(1, 1, -1), dim=-1)
        finite = torch.isfinite(state).all(dim=-1)
        diagnostic_limit = int(os.environ.get("RMS_TERMINAL_TOP1_LIMIT", "0"))
        if diagnostic_limit > 0:
            terminal_top1, terminal_top1_count = _terminal_mask_top1(
                terminal, embed.weight, self.mask_id, valid, limit=diagnostic_limit
            )
        else:
            terminal_top1 = terminal.new_zeros((), dtype=torch.float32)
            terminal_top1_count = terminal.new_zeros((), dtype=torch.float32)
        flow_stats = torch.stack(
            [
                correct[valid].sum().float(),
                top5_correct[valid].sum().float(),
                state_rms[valid].sum(),
                state_l2[valid].sum(),
                (~finite[valid]).sum().float(),
                valid.sum().float(),
                terminal_rms[valid].sum(),
                terminal_mask_cosine[valid].sum(),
                terminal_top1,
                terminal_top1_count,
            ]
        )
        return token_loss, token_count, gamma_stats.detach(), flow_stats.detach()


__all__ = ["RMSContinuousDenoiser", "_posterior_clean_state"]
