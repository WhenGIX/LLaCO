# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fixed-precision attention using matmul/softmax only; never dispatches SDPA."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def explicit_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    compute_dtype: torch.dtype = torch.float64,
    max_score_elements: int = 64 * 1024 * 1024,
    query_chunk_size: int = 0,
) -> Tensor:
    """Compute softmax(QK^T/sqrt(D) + mask)V in one fixed arithmetic dtype.

    Output uses the query dtype. Query chunks retain every key and accumulate
    K/V gradients before one cast-back. Fully blocked boolean/-inf rows return zero;
    finite biases retain additive-mask semantics. There is
    no value-dependent precision selection and no scaled_dot_product_attention.
    """
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError('Explicit attention compute dtype must be FP32 or FP64')
    if is_causal and attn_mask is not None:
        raise ValueError('Explicit mask and is_causal are mutually exclusive')
    if not 0 <= dropout_p <= 1 or max_score_elements <= 0 or query_chunk_size < 0:
        raise ValueError('Invalid dropout or attention allocation budget')
    if query.is_cuda and compute_dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Explicit FP32 attention requires TF32 disabled')
    q,k,v=(x.to(compute_dtype) for x in (query,key,value))
    mask=attn_mask if attn_mask is None or attn_mask.dtype==torch.bool else attn_mask.to(compute_dtype)
    scale=1/math.sqrt(q.shape[-1])
    if not q.shape[-2] or not k.shape[-2]:
        return ((q @ k.transpose(-1,-2))*scale).softmax(-1).matmul(v).to(query.dtype)
    batch_heads=math.prod(torch.broadcast_shapes(q.shape[:-2],k.shape[:-2],v.shape[:-2]))
    chunk=max(1,min(q.shape[-2],max_score_elements//(batch_heads*k.shape[-2])))
    if query_chunk_size:chunk=min(chunk,query_chunk_size)

    def attend(q_part: Tensor, keys: Tensor, values: Tensor, bias: Tensor | None) -> Tensor:
        with torch.autocast(device_type=q_part.device.type,enabled=False):
            scores=(q_part @ keys.transpose(-1,-2))*scale
            blocked_rows=None
            if bias is not None:
                if bias.dtype==torch.bool:
                    scores=scores.masked_fill(~bias,-torch.inf)
                    candidate=~bias.any(-1,keepdim=True)
                else:
                    scores=scores+bias
                    candidate=torch.isneginf(bias).all(-1,keepdim=True)
                # This branch handles mask semantics only, never precision.
                if bool(candidate.any()):
                    blocked_rows=candidate
                    scores=scores.masked_fill(blocked_rows,0.)
            probability=scores.softmax(-1)
            if blocked_rows is not None:probability=probability.masked_fill(blocked_rows,0.)
            if dropout_p:probability=F.dropout(probability,p=dropout_p,training=True)
            return probability @ values

    output=[]
    recompute=chunk<q.shape[-2] and torch.is_grad_enabled() and (
        any(x.requires_grad for x in (q,k,v)) or (mask is not None and mask.requires_grad))
    for start in range(0,q.shape[-2],chunk):
        end=min(start+chunk,q.shape[-2]);bias=mask
        if is_causal:
            bias=torch.arange(k.shape[-2],device=q.device)[None,:]<=torch.arange(start,end,device=q.device)[:,None]
        elif mask is not None and mask.ndim>=2 and mask.shape[-2]!=1:
            bias=mask[...,start:end,:]
        args=(q[...,start:end,:],k,v,bias)
        output.append(checkpoint(attend,*args,use_reentrant=False,preserve_rng_state=True) if recompute else attend(*args))
    return (output[0] if len(output)==1 else torch.cat(output,dim=-2)).to(query.dtype)
