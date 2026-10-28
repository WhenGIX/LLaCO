# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Replicated full-vocabulary spherical adapter and TP synchronization helpers."""

from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .geometry import EGeometry, normalize


class _StreamingSoftmaxExpectation(torch.autograd.Function):
    """Streaming softmax with wide accumulators and bounded saved tensors."""

    @staticmethod
    def forward(ctx, flat, beta, bias, embedding, chunk_size, return_stats):
        accumulation_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
        maximum = torch.full((flat.shape[0], 1), -torch.inf, device=flat.device, dtype=accumulation_dtype)
        denominator = torch.zeros_like(maximum)
        weighted = torch.zeros_like(flat, dtype=accumulation_dtype)
        mean_logits = torch.zeros_like(maximum)
        for start in range(0, embedding.shape[0], chunk_size):
            end = min(start + chunk_size, embedding.shape[0])
            values = embedding[start:end].to(flat.dtype)
            code = F.normalize(embedding[start:end].to(accumulation_dtype), dim=-1).to(flat.dtype)
            logits = ((beta * flat) @ code.T).to(accumulation_dtype) + bias[start:end].to(accumulation_dtype)
            new_max = torch.maximum(maximum, logits.max(-1, keepdim=True).values)
            old_scale = (maximum - new_max).exp()
            probabilities = (logits - new_max).exp()
            denominator = denominator * old_scale + probabilities.sum(-1, keepdim=True)
            # The vocabulary-wide sums can exceed FP16 range even for finite logits.
            weighted.mul_(old_scale).add_((probabilities.to(flat.dtype) @ values).to(accumulation_dtype))
            if return_stats:
                mean_logits = mean_logits * old_scale + (probabilities * logits).sum(-1, keepdim=True)
            maximum = new_max
        output = weighted / denominator
        log_z = maximum + denominator.log()
        mean_logits = mean_logits / denominator
        entropy = (log_z - mean_logits).squeeze(-1) if return_stats else flat.new_empty(0)
        ctx.chunk_size = chunk_size
        ctx.return_stats = return_stats
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(flat, beta, bias, embedding, output, log_z, mean_logits)
        return output, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_entropy):
        flat, beta, bias, embedding, output, log_z, mean_logits = ctx.saved_tensors
        accumulation_dtype = output.dtype
        grad_flat = torch.zeros_like(flat, dtype=accumulation_dtype) if ctx.needs_input_grad[0] else None
        grad_beta = torch.zeros_like(beta, dtype=accumulation_dtype) if ctx.needs_input_grad[1] else None
        grad_bias = torch.zeros_like(bias, dtype=accumulation_dtype) if ctx.needs_input_grad[2] else None
        if grad_output is not None:
            grad_output = grad_output.to(accumulation_dtype)
        baseline = (grad_output * output).sum(-1, keepdim=True) if grad_output is not None else None
        # Recreate and immediately consume one chunk. No autograd graph spans
        # the vocabulary, including during the transformer's backward pass.
        for start in range(0, embedding.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, embedding.shape[0])
            values = embedding[start:end].to(flat.dtype)
            code = F.normalize(embedding[start:end].to(accumulation_dtype), dim=-1).to(flat.dtype)
            similarities = (flat @ code.T).to(accumulation_dtype)
            # Replay the forward GEMM order; in FP16 beta placement changes rounding.
            logits = ((beta * flat) @ code.T).to(accumulation_dtype) + bias[start:end].to(accumulation_dtype)
            probabilities = (logits - log_z).exp()
            derivative = torch.zeros_like(logits)
            if grad_output is not None:
                # Finite BF16/FP32 upstream gradients can exceed FP16 range.
                derivative.add_(grad_output @ values.to(accumulation_dtype).T - baseline)
            if ctx.return_stats and grad_entropy is not None:
                derivative.add_(grad_entropy.unsqueeze(-1) * (mean_logits - logits))
            derivative.mul_(probabilities)
            if grad_flat is not None:
                grad_flat.addmm_(derivative, code.to(accumulation_dtype))
            if grad_beta is not None:
                grad_beta.add_((derivative * similarities).sum())
            if grad_bias is not None:
                grad_bias[start:end] = derivative.sum(0)
        if grad_flat is not None:
            grad_flat.mul_(beta.to(accumulation_dtype))
        return (
            grad_flat.to(flat.dtype) if grad_flat is not None else None,
            grad_beta.to(beta.dtype) if grad_beta is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None,
        )


class SphericalAdapter(nn.Module):
    """Exact full-vocabulary softmax adapter, replicated on every TP rank."""

    def __init__(
        self, geometry: EGeometry, *, beta_init: float = 16.0, chunk_size: int = 512,
        compute_dtype: torch.dtype = torch.float16,
        inference_chunk_size: int | None = None,
    ):
        super().__init__()
        if beta_init <= 0 or not torch.isfinite(torch.tensor(beta_init)):
            raise ValueError("beta_init must be finite and positive")
        # Geometry belongs to the language model. Keeping a weak reference avoids
        # registering the same 2 GiB of persistent buffers under two state-dict paths.
        object.__setattr__(self, "_geometry_ref", weakref.ref(geometry))
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("compute_dtype must be torch.float16, torch.bfloat16, or torch.float32")
        self.compute_dtype = compute_dtype
        self.inference_chunk_size = self.chunk_size if inference_chunk_size is None else int(inference_chunk_size)
        if self.inference_chunk_size <= 0:
            raise ValueError("inference_chunk_size must be positive")
        device = geometry.E_geometry.device
        self.log_beta = nn.Parameter(
            torch.tensor(float(beta_init), dtype=geometry.E_geometry.dtype, device=device).log()
        )
        self.bias = nn.Parameter(
            torch.zeros(geometry.vocab_size, dtype=geometry.E_geometry.dtype, device=device)
        )

    @property
    def beta(self) -> torch.Tensor:
        """Positive inverse-temperature parameter."""
        return self.log_beta.float().exp()

    @property
    def geometry(self) -> EGeometry:
        geometry = self._geometry_ref()
        if geometry is None:
            raise RuntimeError("spherical geometry was released before its adapter")
        return geometry

    def forward(self, state: torch.Tensor, *, return_stats: bool = False):
        """Map spherical states to embeddings using a numerically stable full softmax."""
        work_dtype = self.compute_dtype
        flat = normalize(state.reshape(-1, state.shape[-1])).to(work_dtype)
        embedding = self.geometry.E_geometry.to(device=flat.device)
        chunk_size = self.chunk_size
        if not self.training and not torch.is_grad_enabled():
            chunk_size = self.inference_chunk_size
        output, entropy = _StreamingSoftmaxExpectation.apply(
            flat, self.beta.to(work_dtype), self.bias.float().to(work_dtype), embedding, chunk_size, return_stats
        )
        output = output.reshape_as(state).to(state.dtype)
        if not return_stats:
            return output
        return output, {"adapter_entropy": entropy.reshape(state.shape[:-1]), "beta": self.beta.detach()}

    def posterior(self, state: torch.Tensor) -> torch.Tensor:
        """Materialize a complete-vocabulary posterior for inference-only use."""
        flat = normalize(state.reshape(-1, state.shape[-1])).float()
        logits = self.beta * flat @ self.geometry.normalized_chunk(0, self.geometry.vocab_size).T + self.bias.float()
        return logits.softmax(-1).reshape(*state.shape[:-1], self.geometry.vocab_size)


def broadcast_replicated_adapter(adapter: SphericalAdapter, *, group=None, src: int | None = None) -> None:
    """Broadcast full adapter parameters across the TP group after initialization."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if group is None or src is None:
        from megatron.core import parallel_state

        group = group or parallel_state.get_tensor_model_parallel_group()
        src = parallel_state.get_tensor_model_parallel_src_rank() if src is None else src
    for parameter in (adapter.log_beta, adapter.bias):
        torch.distributed.broadcast(parameter.data, src=src, group=group)


def adapter_parameter_fingerprint(adapter: SphericalAdapter) -> str:
    """Return a deterministic fingerprint for replicated trainable parameters."""
    import hashlib

    digest = hashlib.sha256()
    for parameter in (adapter.log_beta, adapter.bias):
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "SphericalAdapter",
    "adapter_parameter_fingerprint",
    "broadcast_replicated_adapter",
]
