# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native MCore transformer extensions for explicit per-layer spherical Ada."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor
from torch import Tensor, nn

from .conditioning import ada_condition_input, keep_condition_parameters_fp32

from ...nemotron_labs_diffusion_spherical_protocol import (
    SPHERICAL_CONDITION_DIM,
    SUPPORTED_SPHERICAL_CONDITION_DIMS,
)


class SphericalTransformerLayer(TransformerLayer):
    """Nemotron layer with explicit shift/scale/gate Ada on attention and MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        condition_dim = int(getattr(self.config, "spherical_condition_dim", SPHERICAL_CONDITION_DIM))
        if condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
            raise ValueError(
                "native spherical condition_dim must be one of the supported native/HF widths "
                f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
                f"got {condition_dim}"
            )
        self.spherical_ada_projection = nn.Linear(condition_dim, 6 * self.config.hidden_size)
        nn.init.zeros_(self.spherical_ada_projection.weight)
        nn.init.zeros_(self.spherical_ada_projection.bias)
        if getattr(self.config, "spherical_condition_compute_dtype", "model") == "fp32":
            keep_condition_parameters_fp32(self.spherical_ada_projection)

    @staticmethod
    def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        work = value.float() if scale.dtype == torch.float32 else value
        result = work * (1.0 + scale.transpose(0, 1)) + shift.transpose(0, 1)
        return result.to(value.dtype)

    @staticmethod
    def _gate(output_with_bias, gate: Tensor):
        output, bias = output_with_bias
        if gate.dtype == torch.float32:
            output = output.float()
            bias = bias.float() if bias is not None else None
        if bias is not None:
            output = output + bias
            bias = None
        # The projection is zero-initialized, so 1 + gate preserves the converted
        # pretrained transformer exactly at initialization.
        return output * (1.0 + gate.transpose(0, 1)), bias

    def _residual_add(self, branch, residual: Tensor, bda) -> Tensor:
        dtype = residual.dtype
        with self.bias_dropout_add_exec_handler():
            result = bda(self.training, self.config.bias_dropout_fusion)(
                branch, residual.to(torch.promote_types(branch[0].dtype, residual.dtype)), self.hidden_dropout
            )
        return result.to(dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        spherical_time_condition: Tensor,
        spherical_position_ids: Tensor,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        **kwargs,
    ):
        if packed_seq_params is not None:
            raise ValueError("spherical transformer does not support packed sequences")
        if inference_context is not None and getattr(self.config, "inference_fuse_tp_communication", False):
            raise ValueError("spherical Ada does not support fused TP inference kernels")
        ada = self.spherical_ada_projection(ada_condition_input(
            spherical_time_condition, getattr(self.config, "spherical_ada_activation", "identity")
        ))
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = ada.chunk(6, dim=-1)

        residual = hidden_states
        normalized = apply_module(self.input_layernorm)(hidden_states)
        if isinstance(normalized, tuple):
            normalized, residual = normalized
        normalized = self._modulate(normalized, attn_shift, attn_scale)
        core_attention = self.self_attention.core_attention
        if not hasattr(core_attention, "set_spherical_position_ids"):
            raise RuntimeError("spherical transformer requires SphericalNemotronAttention")
        core_attention.set_spherical_position_ids(spherical_position_ids)
        try:
            attention_output = self.self_attention(
                normalized,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        finally:
            core_attention.clear_spherical_position_ids()
        attention_output = self._gate(attention_output, attn_gate)
        hidden_states = self._residual_add(attention_output, residual, self.self_attn_bda)

        normalized = apply_module(self.pre_mlp_layernorm)(hidden_states)
        residual = hidden_states
        if isinstance(normalized, tuple):
            normalized, residual = normalized
        normalized = self._modulate(normalized, mlp_shift, mlp_scale)
        mlp_output = apply_module(self.mlp)(normalized, padding_mask=padding_mask)
        mlp_output = self._gate(mlp_output, mlp_gate)
        hidden_states = self._residual_add(mlp_output, residual, self.mlp_bda)
        return make_viewless_tensor(hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True), None


class SphericalTransformerBlock(TransformerBlock):
    """Fixed-topology native block forwarding gamma explicitly through recompute."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        *,
        spherical_time_condition: Tensor,
        spherical_position_ids: Tensor,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        **kwargs,
    ) -> Tensor:
        if not self.pre_process:
            hidden_states = self.input_tensor
        hidden_states = make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        # Reentrant checkpointing needs a grad-bearing input even when only
        # internal Ada parameters are trainable and input/conditioner are frozen.
        if (self.training and torch.is_grad_enabled()
                and self.config.recompute_granularity == "full"
                and not hidden_states.requires_grad and not spherical_time_condition.requires_grad):
            hidden_states = hidden_states.detach().requires_grad_(True)
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()

        def run_layer(layer, states, condition, position_ids):
            return layer(
                hidden_states=states,
                attention_mask=attention_mask,
                spherical_time_condition=condition,
                spherical_position_ids=position_ids,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                padding_mask=padding_mask,
            )[0]

        with rng_context:
            for layer in self.layers:
                if self.config.recompute_granularity == "full" and self.training:
                    hidden_states = tensor_parallel.checkpoint(
                        lambda states, condition, position_ids, current=layer: run_layer(
                            current, states, condition, position_ids
                        ),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        spherical_time_condition,
                        spherical_position_ids,
                    )
                else:
                    hidden_states = run_layer(layer, hidden_states, spherical_time_condition, spherical_position_ids)
        if self.final_layernorm is not None:
            hidden_states = apply_module(self.final_layernorm)(hidden_states)
            hidden_states = make_viewless_tensor(hidden_states, requires_grad=True, keep_graph=True)
        return hidden_states


__all__ = ["SphericalTransformerBlock", "SphericalTransformerLayer"]
