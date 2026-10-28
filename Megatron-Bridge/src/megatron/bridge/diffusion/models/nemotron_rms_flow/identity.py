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

"""Cosine identity diagnostics for RMS states.

State vectors remain unnormalized in the model.  Normalization happens only
inside these diagnostic functions, matching the embedding-space specification.
"""

from __future__ import annotations

import torch


def normalized_directions(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def clean_competitor_cosine(embedding_weight, source_ids, valid_ids, chunk_size=32768):
    """Return ``rho_clean[i]`` for each source id with deterministic ties."""
    code = normalized_directions(embedding_weight)
    source = code[source_ids]
    best = torch.full((source_ids.numel(),), -torch.inf, device=source.device)
    for start in range(0, valid_ids.numel(), chunk_size):
        ids = valid_ids[start : start + chunk_size]
        scores = source @ code[ids].T
        equal_source = ids[None, :] == source_ids[:, None]
        scores = scores.masked_fill(equal_source, -torch.inf)
        best = torch.maximum(best, scores.max(dim=-1).values)
    return best


@torch.no_grad()
def cosine_identity_metrics(
    state,
    source_ids,
    embedding_weight,
    valid_ids,
    *,
    rho_clean=None,
    eps_cos=1e-6,
    eps_den=1e-6,
    chunk_size=32768,
):
    """Compute source cosine, competitor, rank, P/A primitives for one gamma.

    ``state`` is not modified.  ``invalid_ratio`` is true when the competitor
    denominator is too close to zero; ``endpoint_degenerate`` is populated by
    callers that supply endpoint-normalized ratio values.
    """
    state_dir = normalized_directions(state)
    code = normalized_directions(embedding_weight)
    source_ids = source_ids.to(device=state.device, dtype=torch.long)
    valid_ids = valid_ids.to(device=state.device, dtype=torch.long)
    source_dir = code[source_ids]
    source_cosine = (state_dir * source_dir).sum(dim=-1)
    competitor = torch.full_like(source_cosine, -torch.inf)
    rank_better = torch.zeros_like(source_ids, dtype=torch.long)
    for start in range(0, valid_ids.numel(), chunk_size):
        ids = valid_ids[start : start + chunk_size]
        scores = state_dir @ code[ids].T
        equal_source = ids[None, :] == source_ids[:, None]
        scores = scores.masked_fill(equal_source, -torch.inf)
        competitor = torch.maximum(competitor, scores.max(dim=-1).values)
        rank_better += (
            (scores > source_cosine[:, None])
            | ((scores == source_cosine[:, None]) & (ids[None, :] < source_ids[:, None]))
        ).sum(dim=-1)
    rank = rank_better + 1
    invalid_ratio = competitor.abs() <= eps_cos
    top1 = rank == 1
    margin = source_cosine - competitor
    result = {
        "source_cosine": source_cosine,
        "competitor_cosine": competitor,
        "margin": margin,
        "rank": rank,
        "top1": top1,
        "invalid_ratio": invalid_ratio,
        "endpoint_degenerate": torch.zeros_like(invalid_ratio),
    }
    if rho_clean is not None:
        rho_clean = rho_clean.to(device=state.device, dtype=source_cosine.dtype)
        score = rho_clean * source_cosine / competitor
        result["identity_score"] = score
        result["invalid_ratio"] = invalid_ratio | ~torch.isfinite(score)
        result["identity_score"] = torch.where(result["invalid_ratio"], torch.full_like(score, torch.nan), score)
    result["P"] = top1.float()
    result["Q"] = 1.0 - (rank.float() - 1.0) / max(valid_ids.numel() - 1, 1)
    return result


def endpoint_normalized_metric(values, endpoint_value, clean_value, eps_den=1e-6):
    """Normalize A/M endpoint values and mark degenerate paths."""
    denominator = clean_value - endpoint_value
    degenerate = denominator.abs() <= eps_den
    normalized = (values - endpoint_value) / denominator
    normalized = torch.where(degenerate, torch.full_like(normalized, torch.nan), normalized)
    return normalized, degenerate


__all__ = [
    "clean_competitor_cosine",
    "cosine_identity_metrics",
    "endpoint_normalized_metric",
    "normalized_directions",
]
