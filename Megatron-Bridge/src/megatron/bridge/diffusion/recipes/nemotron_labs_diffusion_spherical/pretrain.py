# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ConfigContainer recipe for native Nemotron spherical pretraining."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import ParamKey

from megatron.bridge.training.checkpointing import _resolve_checkpoint_iteration
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.training.utils.checkpoint_utils import is_checkpoint_iteration_directory

from ...data.nemotron_labs_diffusion_spherical import (
    PreparedDatasetConfig,
)
from ...models.nemotron_labs_diffusion_spherical.checkpoint import (
    load_spherical_metadata,
    validate_resume_metadata,
)
from ...models.nemotron_labs_diffusion_spherical.provider import (
    NemotronLabsDiffusionModelProvider,
    initialize_spherical_components,
    install_spherical_transformer,
    synchronize_spherical_components,
)
from ...models.nemotron_labs_diffusion_spherical.scheduler import (
    EPSILON_UNIFORM,
    LOGISTIC_GAMMA_C,
    LOGISTIC_K,
)
from ...nemotron_labs_diffusion_spherical_protocol import (
    LEGACY_SPHERICAL_PROTOCOL_VERSION,
    SPHERICAL_CONDITION_DIM,
    SPHERICAL_PROTOCOL_VERSION,
    SUPPORTED_SPHERICAL_CONDITION_DIMS,
    T_UNIFORM_ENDPOINT_INCLUSIVE,
    T_UNIFORM_GRID_SIZE,
    T_UNIFORM_GRID_STEP,
)


FORMAL_NUM_EPOCHS = 3
FORMAL_GLOBAL_BATCH_SIZE = 72
FORMAL_TRAIN_ITERS = 28_931
logger = logging.getLogger(__name__)


def _topology_from_env() -> tuple[int, int, int, int, int]:
    """Resolve the runtime parallel topology without changing spherical math."""
    tp = int(os.environ.get("SPHERICAL_TP_SIZE", "4"))
    dp = int(os.environ.get("SPHERICAL_DP_SIZE", "2"))
    pp = int(os.environ.get("SPHERICAL_PP_SIZE", "1"))
    cp = int(os.environ.get("SPHERICAL_CP_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(tp * dp * pp * cp)))
    if min(tp, dp, pp, cp, world_size) <= 0:
        raise ValueError("spherical parallel sizes must be positive")
    if world_size != tp * dp * pp * cp:
        raise ValueError(
            "WORLD_SIZE must equal SPHERICAL_TP_SIZE * SPHERICAL_DP_SIZE "
            "* SPHERICAL_PP_SIZE * SPHERICAL_CP_SIZE"
        )
    return tp, dp, pp, cp, world_size


def _epoch_contract(
    *, train_iters: int, global_batch_size: int, valid_targets: int | None,
    resume_step: int = 0, consumed_samples: int = 0,
) -> dict[str, int | float | None]:
    """Describe duration against the index currently selected for this run."""
    if min(resume_step, consumed_samples) < 0 or train_iters < resume_step:
        raise ValueError("training duration must include the nonnegative resume cursor")
    scheduled_samples = consumed_samples + (train_iters - resume_step) * global_batch_size
    resume = {"resume_step": resume_step, "consumed_samples": consumed_samples} if resume_step else {}
    if valid_targets is None:
        return {
            "valid_targets": None,
            "requested_epochs": None,
            "required_samples": None,
            "scheduled_samples": scheduled_samples,
            "effective_epochs": None,
            **resume,
        }
    if valid_targets <= 0:
        raise ValueError("valid_targets must be positive")
    required_samples = valid_targets * FORMAL_NUM_EPOCHS
    return {
        "valid_targets": valid_targets,
        "requested_epochs": FORMAL_NUM_EPOCHS,
        "required_samples": required_samples,
        "scheduled_samples": scheduled_samples,
        "effective_epochs": scheduled_samples / valid_targets,
        **resume,
    }


class SphericalOptimizerOverrideProvider(OptimizerConfigOverrideProvider):
    """Apply independent learning-rate policies to spherical parameter groups."""

    def __init__(
        self, *, base_lr: float, ada_lr_mult: float, adapter_lr_mult: float,
        conditioner_lr_mult: float = 10.0,
    ):
        if any(not math.isfinite(value) or value <= 0 for value in (base_lr, ada_lr_mult, adapter_lr_mult, conditioner_lr_mult)):
            raise ValueError("spherical learning rates and multipliers must be finite and positive")
        self.base_lr = base_lr
        self.ada_lr_mult = ada_lr_mult
        self.adapter_lr_mult = adapter_lr_mult
        self.conditioner_lr_mult = conditioner_lr_mult

    def build_config_overrides(self, context: OptimizerConfigOverrideProviderContext):
        return {
            ParamKey(name="*spherical_adapter.*"): {
                "max_lr": self.base_lr * self.adapter_lr_mult,
                "min_lr": context.optimizer_config.min_lr * self.adapter_lr_mult,
                "start_wd": 0.0,
                "end_wd": 0.0,
            },
            ParamKey(name="*time_conditioner.*"): {
                "max_lr": self.base_lr * self.conditioner_lr_mult,
                "min_lr": context.optimizer_config.min_lr * self.conditioner_lr_mult,
                "start_wd": 0.0,
                "end_wd": 0.0,
            },
            ParamKey(name="*spherical_ada_projection.*"): {
                "max_lr": self.base_lr * self.ada_lr_mult,
                "min_lr": context.optimizer_config.min_lr * self.ada_lr_mult,
                "start_wd": 0.0,
                "end_wd": 0.0,
            },
        }


def nemotron_labs_diffusion_spherical_pretrain_config(
    *,
    hf_path: str | None = None,
    model_checkpoint: str,
    output_dir: str = "artifacts/nemotron_native_spherical_flow",
    run_name: str = "native_spherical_3epoch",
    seq_length: int | None = None,
    target_max_tokens: int | None = None,
    block_size: int = 8,
    micro_batch_size: int = 6,
    global_batch_size: int = FORMAL_GLOBAL_BATCH_SIZE,
    gradient_accumulation_steps: int | None = None,
    train_iters: int = FORMAL_TRAIN_ITERS,
    base_lr: float = 1e-5,
    ada_lr_mult: float = 10.0,
    adapter_lr_mult: float = 10.0,
    conditioner_lr_mult: float = 10.0,
    spherical_compute_dtype: str = "fp32",
    enable_recompute: bool = True,
    spherical_condition_dim: int = SPHERICAL_CONDITION_DIM,
    spherical_condition_coordinate: str = "t",
    spherical_ada_activation: str = "identity",
    spherical_time_embedding_scale: float = 1.0,
    spherical_condition_compute_dtype: str = "model",
    spherical_condition_norm: str = "none",
    spherical_gamma_uniform_probability: float = 0.0,
    spherical_condition_rms_max: float | None = None,
    spherical_trainable_component: str = "all",
    spherical_decode_training: bool = False,
    spherical_decode_probability: float = 0.2,
    spherical_decode_gamma_min: float = 0.85,
    spherical_decode_gamma_max: float = 0.95,
    prepared_manifest: str,
) -> ConfigContainer:
    """Construct the complete native training configuration without starting training."""
    from ...data.spherical_mixture import load_manifest

    limits = load_manifest(prepared_manifest)["selection"]["filters"]
    target_max_tokens = limits["target_max_tokens"] if target_max_tokens is None else target_max_tokens
    seq_length = limits["context_max_tokens"] + 2 * limits["target_max_tokens"] if seq_length is None else seq_length
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    tp, dp, pp, cp, world_size = _topology_from_env()
    denominator = dp * micro_batch_size
    if global_batch_size % denominator:
        raise ValueError("global_batch_size must be divisible by DP * micro_batch_size")
    expected_accumulation = global_batch_size // denominator
    if gradient_accumulation_steps is not None and gradient_accumulation_steps != expected_accumulation:
        raise ValueError("gradient_accumulation_steps must equal global_batch_size / (DP * micro_batch_size)")
    if block_size <= 0 or target_max_tokens % block_size:
        raise ValueError("target_max_tokens must be a positive multiple of block_size")
    if seq_length <= 2 * target_max_tokens:
        raise ValueError("seq_length must leave room for context plus two complete targets")
    if spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "spherical_condition_dim must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; got {spherical_condition_dim}"
        )
    if hf_path is None:
        raise ValueError("direct LLaCO training requires --hf-path for tokenizer/config files")
    from transformers import AutoConfig
    from megatron.bridge.training.model_load_save import load_model_config
    from ...models.nemotron_labs_diffusion_spherical.checkpoint import load_spherical_metadata
    from ...models.nemotron_labs_diffusion_spherical.decode_protocol import copy_decode_fields
    from llaco.runtime import register_checkpoint_config_alias

    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True, local_files_only=True)
    register_checkpoint_config_alias(Path(model_checkpoint), hf_config)
    loaded_provider, _ = load_model_config(model_checkpoint)
    if not isinstance(loaded_provider, NemotronLabsDiffusionModelProvider):
        raise TypeError("LLaCO model-only checkpoint does not contain a spherical model provider")
    provider = loaded_provider
    copy_decode_fields(provider, load_spherical_metadata(model_checkpoint))
    provider.perform_initialization = True
    provider.register_pre_wrap_hook(lambda models: initialize_spherical_components(models, block_size=block_size))
    provider.register_pre_wrap_hook(install_spherical_transformer)
    provider.register_post_wrap_hook(synchronize_spherical_components)
    tokenizer_model = hf_path
    provider.tensor_model_parallel_size = tp
    provider.should_pad_vocab = True
    provider.make_vocab_size_divisible_by = 1
    provider.pipeline_model_parallel_size = pp
    provider.context_parallel_size = cp
    provider.sequence_parallel = False
    # Keep the per-sample CE sum through backward. MCore then SUM-reduces the
    # DP gradients and divides by the global valid-sample count.
    provider.calculate_per_token_loss = True
    provider.seq_length = seq_length
    provider.block_size = block_size
    provider.ada_lr_mult = ada_lr_mult
    provider.adapter_lr_mult = adapter_lr_mult
    provider.conditioner_lr_mult = conditioner_lr_mult
    provider.spherical_compute_dtype = spherical_compute_dtype
    provider.spherical_condition_dim = spherical_condition_dim
    provider.spherical_condition_coordinate = spherical_condition_coordinate
    provider.spherical_ada_activation = spherical_ada_activation
    provider.spherical_time_embedding_scale = spherical_time_embedding_scale
    provider.spherical_condition_compute_dtype = spherical_condition_compute_dtype
    provider.spherical_condition_norm = spherical_condition_norm
    provider.spherical_gamma_uniform_probability = spherical_gamma_uniform_probability
    provider.spherical_condition_rms_max = spherical_condition_rms_max
    provider.spherical_trainable_component = spherical_trainable_component
    provider.spherical_decode_training = spherical_decode_training
    provider.spherical_decode_probability = spherical_decode_probability
    provider.spherical_decode_gamma_min = spherical_decode_gamma_min
    provider.spherical_decode_gamma_max = spherical_decode_gamma_max
    provider.validate_fixed_topology()
    if enable_recompute:
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=base_lr,
        min_lr=base_lr * 0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        bf16=True,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    scheduler = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=500,
        lr_decay_iters=train_iters,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )
    tokenizer = (
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
        if tokenizer_model
        else TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=provider.vocab_size)
    )
    run_dir = os.path.join(output_dir, run_name)
    dataset = PreparedDatasetConfig(
        seq_length=seq_length,
        target_max_tokens=target_max_tokens,
        block_size=block_size,
        prepared_manifest=prepared_manifest,
        dataloader_type="single",
    )
    dataset.manifest_fingerprint = _resolve_data_fingerprint(dataset)
    valid_targets = None
    provider.spherical_protocol_metadata = {
        "version": SPHERICAL_PROTOCOL_VERSION,
        "topology": {"world_size": world_size, "tp": tp, "dp": dp, "pp": pp, "cp": cp},
        "architecture": {
            "hidden_size": 4096,
            "ffn_hidden_size": 14336,
            "num_layers": 34,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "vocab_size": 131072,
        },
        "block_size": block_size,
        "conditioning": {"output_rms_max": spherical_condition_rms_max, "input_coordinate": spherical_condition_coordinate,
                         "ada_activation": spherical_ada_activation, "time_embedding_scale": spherical_time_embedding_scale,
                         "compute_dtype": spherical_condition_compute_dtype, "output_norm": spherical_condition_norm},
        "semantic_time_sampling": {
            "gamma_uniform_fraction": spherical_gamma_uniform_probability,
            "t_uniform_fraction": 1.0 - spherical_gamma_uniform_probability,
            "branch_names": ["gamma_uniform", "t_uniform"] if spherical_gamma_uniform_probability else ["t_uniform"],
            "gamma_range": [0.0, 1.0],
            "t_range": [0.0, 1.0],
            "grid_size": T_UNIFORM_GRID_SIZE,
            "step": T_UNIFORM_GRID_STEP,
            "endpoint_inclusive": T_UNIFORM_ENDPOINT_INCLUSIVE,
            "sampling_method": (
                "per_block_Bernoulli_mixture_uniform_gamma_or_uniform_t_grid_shared_qlog_mapping"
                if spherical_gamma_uniform_probability else "uniform_integer_index_then_divide_by_1000_then_shared_inverse"
            ),
            "sample_unit": "noisy_block",
            "model_input": "gamma" if spherical_condition_coordinate == "gamma" else "semantic_time",
            "scheduler": {
                "type": "endpoint_normalized_logistic_q_log",
                "logistic_k": LOGISTIC_K,
                "logistic_gamma_c": LOGISTIC_GAMMA_C,
                "epsilon_uniform": EPSILON_UNIFORM,
                "inverse": "monotone_piecewise_linear_table",
                "inverse_grid_size": 10001,
            },
            "logged_fields": [
                "t",
                "gamma",
                "gamma_uniform_fraction",
                "t_uniform_fraction",
            ],
        },
        "seq_length": seq_length,
        "target_max_tokens": target_max_tokens,
        "training": {
            "micro_batch_size": micro_batch_size,
            "global_batch_size": global_batch_size,
            "train_iters": train_iters,
            "epoch_contract": _epoch_contract(
                train_iters=train_iters,
                global_batch_size=global_batch_size,
                valid_targets=valid_targets,
            ),
            "base_lr": base_lr,
            "ada_lr_mult": ada_lr_mult,
            "adapter_lr_mult": adapter_lr_mult,
            "conditioner_lr_mult": conditioner_lr_mult,
        },
        "data_manifest_fingerprint": dataset.manifest_fingerprint,
    }
    return ConfigContainer(
        model=provider,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            eval_interval=0,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_config_override_provider=SphericalOptimizerOverrideProvider(
            base_lr=base_lr, ada_lr_mult=ada_lr_mult, adapter_lr_mult=adapter_lr_mult,
            conditioner_lr_mult=conditioner_lr_mult,
        ),
        ddp=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            average_in_collective=False,
            grad_reduce_in_fp32=False,
            check_for_nan_in_grad=True,
        ),
        dataset=dataset,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=os.path.join(run_dir, "tb_logs"),
            # The acceptance callback owns the compact W&B payload. Keep the
            # generic Bridge logger from duplicating every loss_dict field.
            tensorboard_log_interval=10**9,
            log_throughput=False,
            log_memory_interval=10**9,
            # Keep Bridge's default logger aligned with the compact Plan 3
            # dashboard; the acceptance callback still records one global
            # gradient norm in the train/* namespace.
            log_l2_norm_grad_to_tensorboard=False,
            log_loss_scale_to_tensorboard=False,
            wandb_project=os.environ.get("WANDB_PROJECT", "nemotron-spherical-log-Q-attn"),
            wandb_exp_name=run_name,
            wandb_save_dir=os.path.join(run_dir, "wandb"),
        ),
        tokenizer=tokenizer,
        checkpoint=CheckpointConfig(
            save=os.path.join(run_dir, "checkpoints"),
            load=os.path.join(run_dir, "checkpoints"),
            save_interval=500,
            most_recent_k=1,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        mixed_precision=MixedPrecisionConfig(
            bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, grad_reduce_in_fp32=False,
        ),
    )


def validate_runtime_config(cfg: ConfigContainer, *, allow_smoke_config: bool = False) -> None:
    """Validate the post-Hydra configuration so overrides cannot change the protocol."""
    cfg.model.validate_fixed_topology()
    cfg.dataset.__post_init__()
    topology = {
        "world_size": int(os.environ.get("WORLD_SIZE", str(cfg.model.tensor_model_parallel_size * int(os.environ.get("SPHERICAL_DP_SIZE", "2")) * cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size))),
        "tp": cfg.model.tensor_model_parallel_size,
        "dp": int(os.environ.get("SPHERICAL_DP_SIZE", "2")),
        "pp": cfg.model.pipeline_model_parallel_size,
        "cp": cfg.model.context_parallel_size,
    }
    if topology["world_size"] != topology["tp"] * topology["dp"] * topology["pp"] * topology["cp"]:
        raise ValueError(f"WORLD_SIZE does not match spherical topology: {topology}")
    denominator = topology["dp"] * cfg.train.micro_batch_size
    if cfg.train.global_batch_size % denominator:
        raise ValueError("global_batch_size must be divisible by DP * micro_batch_size")
    if getattr(cfg.dataset, "length_ordering", "legacy") == "balanced_v2":
        preserve_ordering_on_resize = (
            os.environ.get("SPHERICAL_ALLOW_DP_RESIZE") == "1"
            and bool(cfg.checkpoint.load)
            and int(os.environ.get("SPHERICAL_RESUME_STEP", "0")) > 0
        )
        if (cfg.dataset.length_ordering_microbatch != cfg.train.micro_batch_size
                or (cfg.dataset.length_ordering_global_batch != denominator and not preserve_ordering_on_resize)
                or cfg.train.global_batch_size != denominator):
            raise ValueError("Balanced ordering requires its declared DP microbatch layout and accumulation=1")
        if cfg.dataset.length_ordering_global_batch != denominator:
            logger.warning("Preserving the checkpoint's %d-sample ordering permutation with new batch %d; "
                           "the sample stream is unchanged but cross-rank length balance may differ.",
                           cfg.dataset.length_ordering_global_batch, denominator)
    if not allow_smoke_config and cfg.train.global_batch_size <= 0:
        raise ValueError("formal global_batch_size must be positive")
    if not allow_smoke_config:
        if cfg.train.train_iters <= 0:
            raise ValueError("formal spherical train_iters must be positive")
        if cfg.scheduler.lr_decay_iters != cfg.train.train_iters:
            raise ValueError("scheduler.lr_decay_iters must match train.train_iters")
    if cfg.model.seq_length != cfg.dataset.seq_length:
        raise ValueError("model and prepared dataset sequence lengths must match")
    if cfg.dataset.protocol_version != SPHERICAL_PROTOCOL_VERSION:
        raise ValueError(f"dataset protocol_version must be {SPHERICAL_PROTOCOL_VERSION}")
    if cfg.model.spherical_protocol_metadata["version"] != SPHERICAL_PROTOCOL_VERSION:
        raise ValueError(f"model protocol version must be {SPHERICAL_PROTOCOL_VERSION}")
    if cfg.dataset.manifest_fingerprint != cfg.model.spherical_protocol_metadata["data_manifest_fingerprint"]:
        raise ValueError("data manifest is not bound into checkpoint metadata")
    if not cfg.model.calculate_per_token_loss:
        raise ValueError("spherical CE requires MCore global sample normalization")
    if cfg.ddp.average_in_collective:
        raise ValueError("global valid-sample normalization requires sum DP collectives")
    if cfg.model.spherical_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "spherical_condition_dim must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
            f"got {cfg.model.spherical_condition_dim}"
        )


def bind_runtime_protocol(cfg: ConfigContainer, *, hf_path: str | None) -> None:
    """Rebind derived metadata after declarative CLI overrides are applied."""
    # The provider is created before CLI overrides; rebuild it from resolved values.
    cfg.optimizer_config_override_provider = SphericalOptimizerOverrideProvider(
        base_lr=cfg.optimizer.lr,
        ada_lr_mult=cfg.model.ada_lr_mult,
        adapter_lr_mult=cfg.model.adapter_lr_mult,
        conditioner_lr_mult=cfg.model.conditioner_lr_mult,
    )
    logger.info(
        "spherical LR peaks: backbone=%.8g conditioner=%.8g Ada=%.8g adapter=%.8g",
        cfg.optimizer.lr, cfg.optimizer.lr * cfg.model.conditioner_lr_mult,
        cfg.optimizer.lr * cfg.model.ada_lr_mult, cfg.optimizer.lr * cfg.model.adapter_lr_mult,
    )
    resolved_fingerprint = _resolve_data_fingerprint(cfg.dataset)
    cfg.dataset.manifest_fingerprint = resolved_fingerprint
    valid_targets = None
    from ...data.spherical_mixture import TAIL_BLOCK_PROTOCOL, resolve_eos_id

    if cfg.dataset.tail_block_protocol != TAIL_BLOCK_PROTOCOL:
        raise ValueError("Unsupported tail block protocol")
    hf_config_path = Path(hf_path or "") / "config.json"
    if hf_config_path.is_file():
        from transformers import AutoTokenizer

        hf_config = json.loads(hf_config_path.read_text())
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, local_files_only=True)
        eos_id = resolve_eos_id(tokenizer, configured_id=hf_config.get("eos_token_id"))
        cfg.dataset.eos_token_id = resolve_eos_id(tokenizer, configured_id=cfg.dataset.eos_token_id)
        if cfg.dataset.eos_token_id != eos_id:
            raise ValueError("Dataset and model EOS IDs disagree")
    protocol = cfg.model.spherical_protocol_metadata
    from ...models.nemotron_labs_diffusion_spherical.decode_protocol import decode_protocol

    endpoint_protocol = decode_protocol(cfg.model)
    if endpoint_protocol["enabled"]:
        protocol["decode_endpoint"] = endpoint_protocol
    else:
        protocol.pop("decode_endpoint", None)
    protocol["tail_block"] = {
        "version": TAIL_BLOCK_PROTOCOL, "alignment": "target_relative",
        "block_size": cfg.dataset.block_size, "eos_token_id": cfg.dataset.eos_token_id,
        "original_target": "unchanged, including terminal EOS and newline",
        "fill": "EOS to each sample block ceiling; noisy, attended, positioned and supervised",
        "outer_padding": "not attended by valid queries; not supervised",
        "loss": "same-position CE; mean within diffusion length then mean over samples",
        "reference": "NVIDIA-NeMo/Automodel current SFT two-stage padding; adapted, not paper reproduction",
    }
    attention_backend = cfg.model.spherical_attention_backend
    native_bound = cfg.model.spherical_attention_native_score_bound
    fp64_bound = cfg.model.spherical_attention_fp64_score_bound
    if attention_backend not in {"native", "adaptive_math", "explicit_fp32", "explicit_fp64"}:
        raise ValueError("Unknown spherical_attention_backend")
    if (not math.isfinite(native_bound) or native_bound < 0 or not math.isfinite(fp64_bound)
            or fp64_bound <= 0 or native_bound > fp64_bound or cfg.model.spherical_attention_max_score_elements <= 0):
        raise ValueError("Invalid adaptive attention precision bounds or allocation budget")
    protocol["attention_numerics"] = {
        "backend": attention_backend, "version": "range_guard_full_math_v1",
        "native_score_bound": native_bound, "fp64_score_bound": fp64_bound,
        "max_score_elements": cfg.model.spherical_attention_max_score_elements,
        "score_bound": "per_head_float64_cauchy_schwarz_plus_finite_additive_bias",
        "bound_action": "arithmetic_selection_only_no_logit_clipping",
        "output_dtype": "model", "large_allocation": "query_chunks_with_wide_gradient_accumulation",
    }
    if attention_backend.startswith("explicit_"):
        if cfg.model.spherical_explicit_fp64_query_chunk_size < 0:
            raise ValueError("Explicit FP64 query chunk size must be nonnegative")
        protocol["attention_numerics"] = {
            "backend": attention_backend, "version": "explicit_matmul_softmax_v1",
            "compute_dtype": "float64" if attention_backend == "explicit_fp64" else "float32",
            "uses_sdpa": False, "dynamic_precision": False,
            "query_chunk_size": cfg.model.spherical_explicit_fp64_query_chunk_size if attention_backend == "explicit_fp64" else 0,
            "max_score_elements": cfg.model.spherical_attention_max_score_elements,
            "output_dtype": "model", "large_allocation": "query_chunks_with_wide_gradient_accumulation",
        }
    coordinate = getattr(cfg.model, "spherical_condition_coordinate", "t")
    if coordinate not in {"t", "gamma"}:
        raise ValueError("spherical_condition_coordinate must be 't' or 'gamma'")
    activation = getattr(cfg.model, "spherical_ada_activation", "identity")
    if activation not in {"identity", "silu"}:
        raise ValueError("spherical_ada_activation must be 'identity' or 'silu'")
    time_scale = float(getattr(cfg.model, "spherical_time_embedding_scale", 1.0))
    condition_dtype = getattr(cfg.model, "spherical_condition_compute_dtype", "model")
    condition_norm = getattr(cfg.model, "spherical_condition_norm", "none")
    if condition_norm not in {"none", "layernorm"}:
        raise ValueError("spherical_condition_norm must be 'none' or 'layernorm'")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("spherical_time_embedding_scale must be finite and positive")
    if condition_dtype not in {"model", "fp32"}:
        raise ValueError("spherical_condition_compute_dtype must be 'model' or 'fp32'")
    if "semantic_time_sampling" in protocol:
        protocol["semantic_time_sampling"]["model_input"] = "gamma" if coordinate == "gamma" else "semantic_time"
    probability = float(getattr(cfg.model, "spherical_gamma_uniform_probability", 0.0))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("spherical_gamma_uniform_probability must be finite and in [0, 1]")
    if "semantic_time_sampling" in protocol:
        sampling = protocol["semantic_time_sampling"]
        sampling.update(gamma_uniform_fraction=probability, t_uniform_fraction=1.0-probability,
                        branch_names=(["gamma_uniform", "t_uniform"] if probability else ["t_uniform"]))
        if probability:
            sampling["sampling_method"] = "per_block_Bernoulli_mixture_uniform_gamma_or_uniform_t_grid_shared_qlog_mapping"
            sampling["gamma_uniform_distribution"] = "continuous uniform [0,1)"
        else:
            sampling["sampling_method"] = "uniform_integer_index_then_divide_by_1000_then_shared_inverse"
            sampling.pop("gamma_uniform_distribution", None)
    if endpoint_protocol["enabled"]:
        sampling = protocol["semantic_time_sampling"]
        sampling.update(
            branch_names=["denoise", "decode"],
            t_uniform_fraction=1.0 - cfg.model.spherical_decode_probability,
            decode_fraction=cfg.model.spherical_decode_probability,
            gamma_uniform_fraction=0.0,
            t_range=[0.001, 1.0], grid_size=1000,
            endpoint_inclusive=False,
            sampling_method="block_decode_Bernoulli_else_uniform_positive_t_grid",
            decode_condition_t=0.0,
            decode_gamma_range=[cfg.model.spherical_decode_gamma_min, cfg.model.spherical_decode_gamma_max],
            statistics_scope="t_stage_and_sampling_t_gamma_are_denoise_only",
        )
    weight = cfg.model.spherical_condition_l2_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("spherical_condition_l2_weight must be finite and nonnegative")
    if weight and cfg.model.spherical_condition_rms_max is not None:
        raise ValueError("Conditioner L2 experiment requires the hard RMS bound disabled")
    protocol["conditioning"] = {
        "input_coordinate": coordinate,
        "ada_activation": activation,
        "time_embedding_scale": time_scale,
        "compute_dtype": condition_dtype,
        "output_norm": condition_norm,
        "output_rms_max": cfg.model.spherical_condition_rms_max,
        "output_l2_weight": weight,
        "trainable_component": cfg.model.spherical_trainable_component,
        "l2_reduction": "mean over all sequence positions and channels per valid sample",
    }
    protocol.update(
        seq_length=cfg.model.seq_length,
        target_max_tokens=cfg.dataset.target_max_tokens,
        training={
            "micro_batch_size": cfg.train.micro_batch_size,
            "global_batch_size": cfg.train.global_batch_size,
            "train_iters": cfg.train.train_iters,
            "epoch_contract": _epoch_contract(
                train_iters=cfg.train.train_iters,
                global_batch_size=cfg.train.global_batch_size,
                valid_targets=valid_targets,
            ),
            "base_lr": cfg.optimizer.lr,
            "ada_lr_mult": cfg.model.ada_lr_mult,
            "adapter_lr_mult": cfg.model.adapter_lr_mult,
            "conditioner_lr_mult": cfg.model.conditioner_lr_mult,
        },
        data_manifest_fingerprint=cfg.dataset.manifest_fingerprint,
    )
    if cfg.dataset.prepared_manifest:
        from ...data.spherical_mixture import load_manifest, mixture_plan
        import hashlib

        protocol["data_mixture"] = mixture_plan(cfg.dataset)
        manifest = load_manifest(cfg.dataset.prepared_manifest)
        for name, expected_hash in manifest["tokenizer"].items():
            path = Path(hf_path or "") / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Prepared-data tokenizer mismatch: {path}")
        protocol["training"]["epoch_contract"].update(
            termination="optimizer_step_budget", exhaustion="reshuffle_each_source_and_continue",
            requested_epochs=None, required_samples=None)


def _resolve_data_fingerprint(dataset: PreparedDatasetConfig) -> str:
    """Bind the prepared dataset and sampling plan into checkpoint metadata."""
    from ...data.spherical_mixture import mixture_plan

    return mixture_plan(dataset)["id"]


def validate_resume_run_config(cfg: ConfigContainer) -> None:
    """Validate load-critical structure, allowing data and training policy changes."""
    if not cfg.checkpoint.load:
        return
    load_root = Path(cfg.checkpoint.load)
    if not load_root.exists():
        return
    if is_checkpoint_iteration_directory(str(load_root)):
        iteration_dir = load_root
    else:
        iteration, release = _resolve_checkpoint_iteration(str(load_root), cfg.checkpoint.ckpt_step)
        if iteration < 0:
            return
        iteration_dir = load_root / ("release" if release else f"iter_{iteration:07d}")
    run_config_path = iteration_dir / "run_config.yaml"
    if not run_config_path.is_file():
        raise ValueError(f"spherical resume is missing {run_config_path}")
    train_state_path = iteration_dir / "train_state.pt"
    if train_state_path.is_file() and not cfg.checkpoint.finetune:
        import torch

        cursor = torch.load(train_state_path, map_location="cpu", weights_only=True)
        cfg.model.spherical_protocol_metadata["training"]["epoch_contract"] = _epoch_contract(
            train_iters=cfg.train.train_iters,
            global_batch_size=cfg.train.global_batch_size,
            valid_targets=None,
            resume_step=int(cursor["step"]),
            consumed_samples=int(cursor["consumed_train_samples"]),
        )
    from omegaconf import OmegaConf

    saved = OmegaConf.to_container(OmegaConf.load(run_config_path), resolve=True)
    saved_protocol = saved.get("model", {}).get("spherical_protocol_metadata")
    expected = cfg.model.spherical_protocol_metadata
    if not isinstance(saved_protocol, dict):
        raise ValueError("spherical resume is missing model.spherical_protocol_metadata")
    if cfg.dataset.prepared_manifest and saved_protocol.get("data_manifest_fingerprint") != cfg.dataset.manifest_fingerprint:
        from megatron.bridge.diffusion.data.spherical_mixture import PreparedMixtureDataset, validate_ordering_transition
        from types import SimpleNamespace

        if getattr(cfg.dataset, "length_ordering", "legacy") != "balanced_v2":
            raise ValueError("Prepared mixture plan changed; cannot reuse the checkpoint data cursor")
        saved_cursor = json.loads((iteration_dir / "data_cursor.json").read_text())["cursor"]
        if (saved_cursor["plan_id"] != saved_protocol.get("data_manifest_fingerprint")
                or saved_cursor["consumed_train_samples"] != int(cursor["consumed_train_samples"])):
            raise ValueError("Ordering migration checkpoint identity/cursor mismatch")
        audit = PreparedMixtureDataset(cfg.dataset, SimpleNamespace(pad_token_id=0, eos_token_id=cfg.dataset.eos_token_id), 1)
        validate_ordering_transition(saved_cursor, audit.cursor_state(saved_cursor["consumed_train_samples"]), audit.plan)
        logger.info("Validated explicit data-order migration at sample %d: %s -> %s",
                    saved_cursor["consumed_train_samples"], saved_cursor["plan_id"], audit.plan["id"])
    saved_version = saved_protocol.get("version")
    if saved_version not in {LEGACY_SPHERICAL_PROTOCOL_VERSION, expected["version"]}:
        raise ValueError(
            "spherical resume protocol version is not compatible with the current sampler: "
            f"saved={saved_version}, expected={expected['version']}"
        )
    old_tail = saved_protocol.get("tail_block", {}).get("version", "original_length_v0")
    new_tail = expected.get("tail_block", {}).get("version", "original_length_v0")
    if old_tail != new_tail:
        expected["tail_block_transition"] = {
            "from": old_tail, "to": new_tail, "checkpoint": str(iteration_dir),
            "completed_step": int(cursor["step"]),
            "consumed_train_samples": int(cursor["consumed_train_samples"]),
        }
        logger.warning("Tail block training protocol transition: %s", expected["tail_block_transition"])
    from ...models.nemotron_labs_diffusion_spherical.decode_protocol import decode_transition

    transition = decode_transition(
        saved.get("model", {}), cfg.model,
        allow=os.environ.get("SPHERICAL_ALLOW_DECODE_TRANSITION") == "1",
    )
    if transition is not None:
        expected["decode_transition"] = {
            **transition, "checkpoint": str(iteration_dir),
            "completed_step": int(cursor["step"]) if train_state_path.is_file() and not cfg.checkpoint.finetune else None,
            "optimizer": "preserved", "data_cursor": "preserved",
        }
        logger.warning("Explicit decode objective transition: %s", expected["decode_transition"])
    saved_structure = dict(saved_protocol)
    expected_structure = dict(expected)
    # Data identity, duration, LR and batch policy are provenance, not tensor
    # compatibility. Native loaders still validate model/optimizer state.
    saved_structure.pop("version", None)
    expected_structure.pop("version", None)
    saved_structure.pop("semantic_time_sampling", None)
    expected_structure.pop("semantic_time_sampling", None)
    allow_dp_resize = os.environ.get("SPHERICAL_ALLOW_DP_RESIZE") == "1"
    for key in ("architecture", "topology", "block_size"):
        if key == "topology" and allow_dp_resize:
            old_topology = dict(saved_structure.get(key, {}))
            new_topology = dict(expected_structure.get(key, {}))
            # Only data parallelism may change; TP/PP/CP architecture stays fixed.
            old_topology.pop("dp", None)
            new_topology.pop("dp", None)
            old_topology.pop("world_size", None)
            new_topology.pop("world_size", None)
            if old_topology != new_topology:
                raise ValueError("spherical resume model-parallel topology mismatch")
            continue
        if saved_structure.get(key) != expected_structure.get(key):
            raise ValueError(f"spherical resume {key} mismatch")
    if saved_structure != expected_structure:
        changed = sorted(
            key for key in saved_structure.keys() | expected_structure.keys()
            if saved_structure.get(key) != expected_structure.get(key)
        )
        logger.warning(
            "Continuing resume with changed run metadata: %s. Saved consumed-sample "
            "count is retained; changed data/batching is not an exact sample-order resume.", changed,
        )
    saved_model = saved.get("model", {})
    if saved_model.get("spherical_condition_coordinate", "t") != getattr(cfg.model, "spherical_condition_coordinate", "t"):
        raise ValueError("spherical resume conditioning coordinate mismatch")
    if saved_model.get("spherical_ada_activation", "identity") != getattr(cfg.model, "spherical_ada_activation", "identity"):
        raise ValueError("spherical resume Ada activation mismatch")
    for name, default in (("spherical_time_embedding_scale", 1.0), ("spherical_condition_compute_dtype", "model"),
                          ("spherical_condition_norm", "none")):
        if saved_model.get(name, default) != getattr(cfg.model, name, default):
            raise ValueError(f"spherical resume {name} mismatch")
    saved_condition_dim = int(saved_model.get("spherical_condition_dim", SPHERICAL_CONDITION_DIM))
    if saved_condition_dim not in SUPPORTED_SPHERICAL_CONDITION_DIMS:
        raise ValueError(
            "spherical resume condition dimension must be one of the supported native/HF widths "
            f"{SUPPORTED_SPHERICAL_CONDITION_DIMS}; "
            f"legacy or unsupported value={saved_condition_dim}"
        )
    if saved_condition_dim != int(cfg.model.spherical_condition_dim):
        raise ValueError(
            "spherical resume condition dimension mismatch: "
            f"saved={saved_condition_dim}, expected={cfg.model.spherical_condition_dim}"
        )
    metadata = load_spherical_metadata(iteration_dir)
    validate_resume_metadata(
        metadata,
        tp_size=int(os.environ.get("SPHERICAL_TP_SIZE", "4")),
        dp_size=int(os.environ.get("SPHERICAL_DP_SIZE", "2")),
        data_manifest_fingerprint=str(cfg.dataset.manifest_fingerprint),
        protocol_version=str(expected["version"]),
        spherical_condition_dim=int(cfg.model.spherical_condition_dim),
        spherical_condition_coordinate=getattr(cfg.model, "spherical_condition_coordinate", "t"),
        spherical_ada_activation=getattr(cfg.model, "spherical_ada_activation", "identity"),
        spherical_time_embedding_scale=getattr(cfg.model, "spherical_time_embedding_scale", 1.0),
        spherical_condition_compute_dtype=getattr(cfg.model, "spherical_condition_compute_dtype", "model"),
        spherical_condition_norm=getattr(cfg.model, "spherical_condition_norm", "none"),
        allow_dp_resize=allow_dp_resize,
    )
    architecture = expected["architecture"]
    metadata_architecture = {
        "hidden_size": metadata.hidden_size,
        "ffn_hidden_size": metadata.ffn_hidden_size,
        "num_layers": metadata.num_layers,
        "num_attention_heads": metadata.num_attention_heads,
        "num_query_groups": metadata.num_query_groups,
        "vocab_size": metadata.vocab_size,
    }
    if metadata_architecture != architecture or metadata.block_size != expected["block_size"]:
        raise ValueError("spherical resume architecture/block metadata mismatch")


__all__ = [
    "FORMAL_GLOBAL_BATCH_SIZE",
    "FORMAL_NUM_EPOCHS",
    "FORMAL_TRAIN_ITERS",
    "SUPPORTED_SPHERICAL_CONDITION_DIMS",
    "SphericalOptimizerOverrideProvider",
    "nemotron_labs_diffusion_spherical_pretrain_config",
    "bind_runtime_protocol",
    "validate_runtime_config",
    "validate_resume_run_config",
]
