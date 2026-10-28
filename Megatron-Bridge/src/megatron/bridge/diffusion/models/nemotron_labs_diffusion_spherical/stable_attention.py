# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Range-guarded SDPA: fused for small scores, explicit wide math otherwise."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def attention_score_bound(query: Tensor, key: Tensor, mask: Tensor | None) -> float:
    """Bound absolute unmasked logits without materializing the QK matrix.

    Float64 reductions keep the range decision independent of narrow overflow.
    The dtype's minimum value is the spherical layout's blocked-position sentinel.
    Other finite additive biases, including large negative ones, enter the bound.
    """
    with torch.no_grad():
        q_norm = torch.linalg.vector_norm(query, dim=-1, dtype=torch.float64).amax(dim=-1)
        k_norm = torch.linalg.vector_norm(key, dim=-1, dtype=torch.float64).amax(dim=-1)
        bound = (q_norm * k_norm).amax() / math.sqrt(query.shape[-1])
        if mask is not None and mask.dtype != torch.bool:
            sentinel = mask == torch.finfo(mask.dtype).min
            valid_bias = torch.isfinite(mask) & ~sentinel
            bias_bound = torch.where(valid_bias, mask.abs(), 0).amax().double()
            # A finite-sentinel-only row has a huge common offset. Preserve
            # math-SDPA's finite-mask semantics instead of ignoring its LSE risk.
            sentinel_only_row = (sentinel.any(dim=-1) & ~valid_bias.any(dim=-1)).any()
            bias_bound = torch.where(sentinel_only_row, torch.inf, bias_bound)
            bound = bound + bias_bound
        return float(bound.item())


def stable_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    native_score_bound: float = 128.0,
    fp64_score_bound: float = 1024.0,
    max_score_elements: int = 64 * 1024 * 1024,
    query_chunk_size: int = 0,
    recompute: bool = True,
    stats: dict | None = None,
) -> Tensor:
    """Preserve attention mathematics while choosing a safe numerical path.

    Bounds select arithmetic; they never clip scores, Q/K, or learned modulation.
    Small scores retain native fused SDPA. Intermediate scores use math FP32;
    large scores use math FP64, avoiding both absolute-LSE reconstruction and
    coarse FP32 score differences. Output and returned parameter gradients retain
    their original dtype. Large allocations are query-chunked with recomputation;
    the normal TP4/MB1 shape fits one math call and avoids nested recomputation.
    """
    if (not math.isfinite(native_score_bound) or native_score_bound < 0
            or not math.isfinite(fp64_score_bound) or fp64_score_bound <= 0
            or native_score_bound > fp64_score_bound):
        raise ValueError("Require 0 <= finite native bound <= positive finite FP64 bound")
    if max_score_elements <= 0 or query_chunk_size < 0:
        raise ValueError("Attention allocation budget must be positive and chunk size nonnegative")
    if is_causal and attn_mask is not None:
        raise ValueError("Explicit attention mask and is_causal are mutually exclusive")
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p must be in [0, 1]")
    if query.shape[-2] == 0 or key.shape[-2] == 0:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal)
    bound = attention_score_bound(query, key, attn_mask)
    if (query.dtype != torch.float64 and native_score_bound > 0
            and math.isfinite(bound) and bound <= native_score_bound):
        path = "native"
    else:
        path = "math_fp64" if query.dtype == torch.float64 or not math.isfinite(bound) or bound > fp64_score_bound else "math_fp32"
    if stats is not None:
        stats[path] = stats.get(path, 0) + 1
        stats["last_backend"] = path
        stats["last_score_bound"] = bound
        stats["max_score_bound"] = max(stats.get("max_score_bound", 0.0), bound)
    if path == "native":
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal)

    dtype = torch.float64 if path == "math_fp64" else torch.float32
    # Cast once so chunk contributions accumulate in the wide dtype before the
    # single cast-back, rather than accumulating shared K/V gradients in BF16.
    q, k, v = (tensor.to(dtype) for tensor in (query, key, value))
    mask = attn_mask if attn_mask is None or attn_mask.dtype == torch.bool else attn_mask.to(dtype)
    batch_heads = math.prod(torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2]))
    chunk = max(1, min(q.shape[-2], max_score_elements // (batch_heads * k.shape[-2])))
    if query_chunk_size:
        chunk = min(chunk, query_chunk_size)

    def attend(q_part: Tensor, keys: Tensor, values: Tensor, bias: Tensor | None, causal: bool) -> Tensor:
        with torch.autocast(device_type=q_part.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q_part, keys, values, attn_mask=bias,
                                                 dropout_p=dropout_p, is_causal=causal)

    if chunk >= q.shape[-2]:
        return attend(q, k, v, mask, is_causal).to(query.dtype)
    outputs = []
    needs_checkpoint = recompute and torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
    for start in range(0, q.shape[-2], chunk):
        end = min(start + chunk, q.shape[-2])
        bias = mask
        if is_causal:
            rows = torch.arange(start, end, device=q.device).unsqueeze(-1)
            columns = torch.arange(k.shape[-2], device=q.device).unsqueeze(0)
            bias = columns <= rows
        elif mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
            bias = mask[..., start:end, :]
        args = (q[..., start:end, :], k, v, bias, False)
        result = checkpoint(attend, *args, use_reentrant=False, preserve_rng_state=True) if needs_checkpoint else attend(*args)
        outputs.append(result)
    return torch.cat(outputs, dim=-2).to(query.dtype)
