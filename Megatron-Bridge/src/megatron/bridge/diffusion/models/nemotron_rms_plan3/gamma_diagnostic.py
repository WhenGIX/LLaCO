"""Independent Plan 3 clean-to-mask geometry diagnostics."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METRICS = ("A_raw", "P", "M", "Q")


@torch.no_grad()
def clean_to_mask_statistics(states: torch.Tensor, source_ids: torch.Tensor, codebook: torch.Tensor) -> dict[str, float | int | None]:
    """Compute global A_raw/P/M/Q in the Plan 3 RMS coordinate space."""
    if states.ndim != 2 or codebook.ndim != 2 or states.shape[1] != codebook.shape[1]:
        raise ValueError("states and codebook must be rank-2 tensors with matching hidden size")
    ids = torch.as_tensor(source_ids, device=states.device, dtype=torch.long).reshape(-1)
    if ids.numel() != states.shape[0] or ids.numel() == 0:
        raise ValueError("source_ids must contain one non-empty entry per state")
    if bool((ids < 0).any()) or bool((ids >= codebook.shape[0]).any()):
        raise ValueError("source_ids contain an out-of-range codebook index")
    scores = F.normalize(states.float(), dim=-1) @ F.normalize(codebook.float(), dim=-1).T
    rows = torch.arange(ids.numel(), device=states.device)
    positive = scores[rows, ids]
    competitor = scores.clone()
    competitor[rows, ids] = -torch.inf
    negative = competitor.max(dim=-1).values
    codebook_ids = torch.arange(codebook.shape[0], device=states.device)
    rank = 1 + (
        (scores > positive.unsqueeze(-1))
        | ((scores == positive.unsqueeze(-1)) & (codebook_ids < ids.unsqueeze(-1)))
    ).sum(dim=-1)
    denom = max(codebook.shape[0] - 1, 1)
    ratio = positive / negative
    valid_ratio = torch.isfinite(ratio) & (negative.abs() > 1e-12)
    return {
        "count": int(ids.numel()),
        "codebook_count": int(codebook.shape[0]),
        "A_raw": float(ratio[valid_ratio].mean()) if bool(valid_ratio.any()) else None,
        "A_raw_invalid_count": int((~valid_ratio).sum()),
        "A_raw_invalid_fraction": float((~valid_ratio).float().mean()),
        "P": float((rank == 1).float().mean()),
        "M": float((positive - negative).mean()),
        "Q": float((1.0 - (rank.float() - 1.0) / denom).mean()),
    }


def manifest() -> dict:
    return {
        "namespace": "nemotron_rms_plan3",
        "schema_version": "plan3-gamma-diagnostic-v1",
        "metrics": list(METRICS),
        "state_definition": "s_gamma=(1-gamma)y+gamma*m",
        "role": "scheduler_geometry_diagnostic",
        "required_for_training": False,
        "required_for_inference": False,
        "codebook": "ordered V_rms/V_pred from Plan3Geometry",
        "rank_definition": "1 + count(score > positive) + count(score == positive and lower codebook index)",
        "metric_definitions": {
            "A_raw": "positive cosine divided by strongest non-source cosine; invalid near-zero denominators excluded",
            "P": "top-1 recoverability under deterministic tie-break",
            "M": "positive cosine minus strongest non-source cosine",
            "Q": "1 - (rank - 1) / (codebook_count - 1)",
        },
    }


__all__ = ["METRICS", "clean_to_mask_statistics", "manifest"]
