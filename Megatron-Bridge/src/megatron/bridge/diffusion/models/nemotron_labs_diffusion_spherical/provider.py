# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Native Nemotron provider with spherical geometry components."""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field

import torch
from megatron.core import parallel_state
from megatron.core.utils import unwrap_model

from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
    NemotronLabsDiffusionModelProvider as NativeNemotronLabsDiffusionModelProvider,
)
from megatron.bridge.models.gpt_provider import ModuleSpec

from ...nemotron_labs_diffusion_spherical_protocol import (
    SPHERICAL_CONDITION_DIM,
    SUPPORTED_SPHERICAL_CONDITION_DIMS,
)
from .adapter import SphericalAdapter, broadcast_replicated_adapter
from .attention import SphericalNemotronAttention
from .conditioning import (
    TimeConditioner, condition_parameter_dtype, keep_condition_parameters_fp32, mark_tp_replicated_parameters,
)
from .geometry import EGeometry
from .transformer import SphericalTransformerBlock, SphericalTransformerLayer


logger = logging.getLogger(__name__)


@dataclass
class NemotronLabsDiffusionModelProvider(NativeNemotronLabsDiffusionModelProvider):
    """MCore Nemotron provider retaining native embeddings and adding spherical modules."""

    block_size: int = 8
    ada_lr_mult: float = 10.0
    adapter_lr_mult: float = 10.0
    conditioner_lr_mult: float = 10.0
    ada_weight_decay: float = 0.0
    adapter_weight_decay: float = 0.0
    freeze_input_embedding: bool = True
    geometry_chunk_size: int = 512
    spherical_compute_dtype: str = "fp32"
    spherical_condition_dim: int = SPHERICAL_CONDITION_DIM
    spherical_condition_coordinate: str = "t"
    spherical_ada_activation: str = "identity"
    spherical_time_embedding_scale: float = 1.0
    spherical_condition_compute_dtype: str = "model"
    spherical_condition_norm: str = "none"
    spherical_attention_backend: str = "adaptive_math"
    spherical_attention_native_score_bound: float = 128.0
    spherical_attention_fp64_score_bound: float = 1024.0
    spherical_attention_max_score_elements: int = 64 * 1024 * 1024
    spherical_explicit_fp64_query_chunk_size: int = 1024
    spherical_gamma_uniform_probability: float = 0.0
    spherical_condition_rms_max: float | None = None
    spherical_condition_l2_weight: float = 0.0
    spherical_trainable_component: str = "all"
    # Accept the neutral QK metadata written by the isolated low-LR experiment.
    # This provider intentionally does not implement the optional RMS QK variant.
    spherical_qk_norm: str = "none"
    spherical_qk_norm_eps: float = 1e-6
    spherical_decode_training: bool = False
    spherical_decode_probability: float = 0.2
    spherical_decode_gamma_min: float = 0.85
    spherical_decode_gamma_max: float = 0.95
    spherical_protocol_metadata: dict = field(default_factory=dict)

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the native MCore GPTModel with spherical attention."""
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            parameters = inspect.signature(transformer_layer_spec).parameters
            transformer_layer_spec = (
                transformer_layer_spec(self, vp_stage=vp_stage)
                if "vp_stage" in parameters
                else transformer_layer_spec(self)
            )
        transformer_layer_spec.submodules.self_attention.submodules.core_attention = SphericalNemotronAttention
        original_spec = self.transformer_layer_spec
        self.transformer_layer_spec = transformer_layer_spec
        model = super(NativeNemotronLabsDiffusionModelProvider, self).provide_language_model(
            pre_process, post_process, vp_stage
        )
        self.transformer_layer_spec = original_spec
        return model

    def validate_fixed_topology(self) -> None:
        """Validate topology and dimensions for the unchanged native checkpoint architecture."""
        if self.spherical_condition_coordinate not in {"t", "gamma"}:
            raise ValueError("spherical_condition_coordinate must be 't' or 'gamma'")
        if not math.isfinite(self.spherical_gamma_uniform_probability) or not 0 <= self.spherical_gamma_uniform_probability <= 1:
            raise ValueError("spherical_gamma_uniform_probability must be finite and in [0, 1]")
        if self.spherical_qk_norm != "none":
            raise ValueError("this provider supports spherical_qk_norm=none only; use the QK experiment implementation")
        if not math.isfinite(self.spherical_qk_norm_eps) or self.spherical_qk_norm_eps <= 0:
            raise ValueError("spherical_qk_norm_eps must be finite and positive")
        expected = {
            "hidden_size": 4096,
            "ffn_hidden_size": 14336,
            "num_layers": 34,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "vocab_size": 131072,
        }
        actual = {
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "vocab_size": self.vocab_size,
        }
        if actual != expected:
            raise ValueError(f"native Nemotron architecture mismatch: expected={expected}, actual={actual}")
        topology = {
            "tp": self.tensor_model_parallel_size,
            "pp": self.pipeline_model_parallel_size,
            "cp": self.context_parallel_size,
        }
        if topology["tp"] <= 0 or topology["pp"] <= 0 or topology["cp"] <= 0:
            raise ValueError(f"native spherical model topology must be positive: got {topology}")
        if self.spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
            raise ValueError(
                "native spherical condition_dim must be one of the supported native/HF widths "
                f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
                f"got {self.spherical_condition_dim}"
            )
        tp = self.tensor_model_parallel_size
        dimensions = {key: actual[key] for key in ("hidden_size", "ffn_hidden_size", "num_attention_heads", "num_query_groups")}
        incompatible = {name: value for name, value in dimensions.items() if value % tp}
        if incompatible:
            detail = ", ".join(f"{name}={value}" for name, value in incompatible.items())
            raise ValueError(
                f"native Nemotron topology is incompatible with TP={tp}: {detail} must be divisible by TP"
            )


def initialize_spherical_components(model, *, block_size: int = 8, chunk_size: int | None = None, compute_dtype: str | None = None):
    """Refresh E_geometry from the already-loaded native checkpoint exactly once."""
    if isinstance(model, list):
        for chunk in model:
            initialize_spherical_components(chunk, block_size=block_size, chunk_size=chunk_size, compute_dtype=compute_dtype)
        return model
    core = model.module if hasattr(model, "module") else model
    embedding = getattr(getattr(core, "embedding", None), "word_embeddings", None)
    if embedding is None:
        raise ValueError("native input embedding is required to initialize E_geometry")
    geometry = EGeometry.from_native_embedding(embedding)
    if chunk_size is None:
        chunk_size = int(core.config.geometry_chunk_size)
    compute_dtype_name = str(compute_dtype or getattr(core.config, "spherical_compute_dtype", "fp16")).lower()
    compute_dtype_value = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(compute_dtype_name)
    if compute_dtype_value is None:
        raise ValueError("spherical_compute_dtype must be fp16, bf16, or fp32")
    adapter = SphericalAdapter(geometry, chunk_size=chunk_size, compute_dtype=compute_dtype_value)
    condition_dim = int(getattr(core.config, "spherical_condition_dim", SPHERICAL_CONDITION_DIM))
    if condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "native spherical condition_dim must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
            f"got {condition_dim}"
        )
    conditioner = TimeConditioner(core.config.hidden_size, core.config.num_layers, condition_dim=condition_dim,
                                  output_rms_max=getattr(core.config, "spherical_condition_rms_max", None),
                                  time_embedding_scale=getattr(core.config, "spherical_time_embedding_scale", 1.0),
                                  output_norm=getattr(core.config, "spherical_condition_norm", "none")).to(
        device=geometry.E_geometry.device, dtype=condition_parameter_dtype(core.config, geometry.E_geometry.dtype)
    )
    if getattr(core.config, "spherical_condition_compute_dtype", "model") == "fp32":
        keep_condition_parameters_fp32(conditioner)
    core.spherical_geometry = geometry
    core.spherical_adapter = adapter
    core.time_conditioner = conditioner
    core.spherical_block_size = block_size
    core.spherical_input_embedding = embedding
    embedding.weight.requires_grad_(False)
    mark_tp_replicated_parameters(adapter)
    mark_tp_replicated_parameters(conditioner)
    trainable = str(getattr(core.config, "spherical_trainable_component", "all")).lower()
    if trainable not in {"all", "ada", "conditioner", "conditioner_ada", "new_modules"}:
        raise ValueError("spherical_trainable_component must be all, ada, conditioner, conditioner_ada, or new_modules")
    return model


def synchronize_spherical_components(model):
    """Verify and broadcast every replicated spherical parameter in its TP group."""
    if isinstance(model, list):
        for chunk in model:
            synchronize_spherical_components(chunk)
        return model
    core = unwrap_model(model)
    group = parallel_state.get_tensor_model_parallel_group()
    src = parallel_state.get_tensor_model_parallel_src_rank()
    adapter = getattr(core, "spherical_adapter", None)
    conditioner = getattr(core, "time_conditioner", None)
    if adapter is None or conditioner is None:
        raise RuntimeError("spherical modules must be attached before TP synchronization")
    broadcast_replicated_adapter(adapter, group=group, src=src)
    replicated_parameters = list(conditioner.parameters())
    for layer in core.decoder.layers:
        replicated_parameters.extend(layer.spherical_ada_projection.parameters())
    for parameter in replicated_parameters:
        if parameter.device.type != "cuda":
            raise RuntimeError("spherical parameters must be on CUDA before TP synchronization")
        torch.distributed.broadcast(parameter.data, src=src, group=group)
    return model


def apply_trainable_component(core):
    """Freeze only after all spherical parameters exist, before DDP/optimizer."""
    component = str(getattr(core.config, "spherical_trainable_component", "all")).lower()
    if component not in {"all", "ada", "conditioner", "conditioner_ada", "new_modules"}:
        raise ValueError("spherical_trainable_component must be all, ada, conditioner, conditioner_ada, or new_modules")
    if component == "all":
        return
    for name, parameter in core.named_parameters():
        trainable = (
            component in {"ada", "conditioner_ada", "new_modules"} and "spherical_ada_projection" in name
        ) or (
            component in {"conditioner", "conditioner_ada", "new_modules"} and name.startswith("time_conditioner.")
        ) or (
            component == "new_modules" and name.startswith("spherical_adapter.")
        )
        parameter.requires_grad_(trainable)
    if not any(parameter.requires_grad for parameter in core.parameters()):
        raise ValueError(f"No parameters selected for {component}-only training")


def install_spherical_transformer(model):
    """Install Ada-aware classes after HF conversion has completed."""
    if isinstance(model, list):
        for chunk in model:
            install_spherical_transformer(chunk)
        return model
    core = model.module if hasattr(model, "module") else model
    core.decoder.__class__ = SphericalTransformerBlock
    for layer in core.decoder.layers:
        _split_fused_layernorm_linears(layer, core.config)
        layer.__class__ = SphericalTransformerLayer
        condition_dim = int(getattr(core.config, "spherical_condition_dim", SPHERICAL_CONDITION_DIM))
        if condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
            raise ValueError(
                "native spherical condition_dim must be one of the supported native/HF widths "
                f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
                f"got {condition_dim}"
            )
        projection = torch.nn.Linear(condition_dim, 6 * core.config.hidden_size)
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.zeros_(projection.bias)
        projection = projection.to(device=next(layer.parameters()).device,
                                   dtype=condition_parameter_dtype(core.config, next(layer.parameters()).dtype))
        if getattr(core.config, "spherical_condition_compute_dtype", "model") == "fp32":
            keep_condition_parameters_fp32(projection)
        layer.spherical_ada_projection = projection
        mark_tp_replicated_parameters(layer.spherical_ada_projection)
    apply_trainable_component(core)
    return model


def _split_fused_layernorm_linears(layer, config) -> None:
    """Split TE fused norms so Ada modulation follows RMSNorm, matching the HF block."""
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        split_te_layernorm_column_parallel_linear,
    )

    qkv = layer.self_attention.linear_qkv
    if isinstance(qkv, TELayerNormColumnParallelLinear):
        input_layernorm, linear_qkv = split_te_layernorm_column_parallel_linear(
            qkv, config=config, tp_group=qkv._tp_group
        )
        layer.input_layernorm = input_layernorm
        layer.self_attention.linear_qkv = linear_qkv
    elif not hasattr(layer.input_layernorm, "weight"):
        raise TypeError("native spherical attention requires a fused or standalone input RMSNorm")

    fc1 = layer.mlp.linear_fc1
    if isinstance(fc1, TELayerNormColumnParallelLinear):
        pre_mlp_layernorm, linear_fc1 = split_te_layernorm_column_parallel_linear(
            fc1, config=config, tp_group=fc1._tp_group
        )
        layer.pre_mlp_layernorm = pre_mlp_layernorm
        layer.mlp.linear_fc1 = linear_fc1
    elif not hasattr(layer.pre_mlp_layernorm, "weight"):
        raise TypeError("native spherical MLP requires a fused or standalone pre-MLP RMSNorm")


__all__ = [
    "NemotronLabsDiffusionModelProvider",
    "initialize_spherical_components",
    "install_spherical_transformer",
    "synchronize_spherical_components",
]
