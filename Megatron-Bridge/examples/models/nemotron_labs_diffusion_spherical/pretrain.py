#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Formal native Bridge entry point for Nemotron spherical training."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def _disable_unused_indexed_dataset_helper_compile() -> None:
    """Prepared token shards do not use the MCore indexed-dataset C++ helper."""
    initialize_module = importlib.import_module("megatron.bridge.training.initialize")
    initialize_module.compile_helpers = lambda: None


def _install_checkpoint_strategy_compatibility() -> None:
    """Bridge small MCore checkpoint API differences in the upgraded runtime."""
    import inspect

    import torch.distributed.checkpoint as torch_checkpoint
    from megatron.core.dist_checkpointing.strategies import torch as torch_strategy
    from megatron.core.utils import get_torch_version

    if not hasattr(torch_strategy, "get_torch_version"):
        torch_strategy.get_torch_version = get_torch_version
    if "no_dist" not in inspect.signature(torch_checkpoint.load).parameters:
        load_without_no_dist = torch_checkpoint.load

        def load_compat(
            state_dict,
            *args,
            checkpoint_id=None,
            storage_reader=None,
            planner=None,
            process_group=None,
            no_dist=None,
        ):
            # MCore's vendored loader still passes storage_reader positionally
            # and supplies the removed no_dist flag.
            if args:
                if storage_reader is not None or len(args) > 1:
                    raise TypeError("unsupported positional torch.distributed.checkpoint.load arguments")
                storage_reader = args[0]
            del no_dist
            return load_without_no_dist(
                state_dict,
                checkpoint_id=checkpoint_id,
                storage_reader=storage_reader,
                planner=planner,
                process_group=process_group,
            )

        torch_checkpoint.load = load_compat


def _write_run_manifest(
    cfg: ConfigContainer, output_dir: str, run_name: str, *, checkpoint_policy: dict | None = None
) -> None:
    """Create a run manifest without rewriting any historical run artifacts."""
    from megatron.bridge.diffusion.nemotron_labs_diffusion_spherical_protocol import SPHERICAL_CONDITION_DIM

    accumulation = cfg.train.global_batch_size // (int(os.environ.get("SPHERICAL_DP_SIZE", "2")) * cfg.train.micro_batch_size)
    manifest = {
        "protocol": cfg.model.spherical_protocol_metadata,
        "spherical_condition_dim": int(cfg.model.spherical_condition_dim),
        "semantic_time_sampling": cfg.model.spherical_protocol_metadata["semantic_time_sampling"],
        "topology": {"world_size": int(os.environ.get("WORLD_SIZE", "8")), "tp": int(os.environ.get("SPHERICAL_TP_SIZE", "4")), "dp": int(os.environ.get("SPHERICAL_DP_SIZE", "2")), "pp": int(os.environ.get("SPHERICAL_PP_SIZE", "1")), "cp": int(os.environ.get("SPHERICAL_CP_SIZE", "1"))},
        "micro_batch_size": cfg.train.micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": cfg.train.global_batch_size,
        "train_iters": cfg.train.train_iters,
        "epoch_contract": cfg.model.spherical_protocol_metadata["training"]["epoch_contract"],
        "seq_length": cfg.model.seq_length,
        "target_max_tokens": cfg.dataset.target_max_tokens,
        "checkpoint_format": cfg.checkpoint.ckpt_format,
    }
    if checkpoint_policy is not None:
        manifest["checkpoint_policy"] = checkpoint_policy
    if getattr(cfg.dataset, "prepared_manifest", None):
        import hashlib

        workspace = Path(__file__).resolve().parents[4]
        diffusion = workspace / "Megatron-Bridge/src/megatron/bridge/diffusion"
        sources = list((diffusion / "models/nemotron_labs_diffusion_spherical").glob("*.py"))
        sources += [diffusion / "data/spherical_mixture.py", diffusion / "data/nemotron_labs_diffusion_spherical.py",
                    diffusion / "recipes/nemotron_labs_diffusion_spherical/pretrain.py", Path(__file__).resolve(),
                    workspace / "llaco/train.py", workspace / "scripts/prepare_spherical_data.py"]
        manifest["source_sha256"] = {str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(sources)}
    run_dir = Path(output_dir) / run_name
    target = run_dir / "run_manifest.json"
    if target.is_file():
        existing = json.loads(target.read_text())
        comparable = dict(existing)
        # v3 manifests written before width metadata was added are unambiguous:
        # the native default was 512.  Compare that default in memory only so
        # the historical file remains byte-for-byte untouched.
        comparable.setdefault("spherical_condition_dim", SPHERICAL_CONDITION_DIM)
        if comparable != manifest:
            raise ValueError(f"refusing to rewrite historical run manifest: {target}")
        logger.info("existing compatible run manifest left unchanged: %s", target)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"refusing to write into historical run directory without a manifest: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized)
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ValueError(f"refusing to rewrite historical run manifest: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(
        "resolved topology micro_batch=%d accumulation=%d global_batch=%d "
        "train_iters=%d seq=%d target=%d",
        cfg.train.micro_batch_size,
        accumulation,
        cfg.train.global_batch_size,
        cfg.train.train_iters,
        cfg.model.seq_length,
        cfg.dataset.target_max_tokens,
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse native recipe arguments and Hydra-style overrides."""
    parser = argparse.ArgumentParser(description="Native Nemotron spherical Bridge pretraining")
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--model-config-checkpoint", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-dir", default="artifacts/nemotron_native_spherical_flow")
    parser.add_argument("--run-name", default="native_spherical_3epoch")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--acceptance-summary", default=None)
    parser.add_argument("--allow-smoke-config", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-only-interval", type=int, default=2000)
    parser.add_argument("--model-only-warmup-interval", type=int, default=0)
    parser.add_argument("--model-only-warmup-steps", type=int, default=0)
    parser.add_argument("--model-only-load", default=None)
    parser.add_argument("--wandb-step-offset", type=int, default=0)
    parser.add_argument("--panel-interval", type=int, default=100)
    parser.add_argument("--panel-gen-length", type=int, default=512)
    parser.add_argument("--panel-smoke", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_known_args()


def main() -> None:
    """Build ConfigContainer and invoke Bridge ``pretrain`` with spherical CE."""
    # Spawned checkpoint writers re-import this file as __mp_main__. They must
    # not initialize every Bridge model family or CUDA just to write tensors.
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.step import SphericalDGPTStep
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.validation import SphericalAcceptanceCallback
    from megatron.bridge.diffusion.recipes.nemotron_labs_diffusion_spherical.pretrain import (
        bind_runtime_protocol,
        nemotron_labs_diffusion_spherical_pretrain_config,
        validate_resume_run_config,
        validate_runtime_config,
    )
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.utils.omegaconf_utils import (
        apply_overrides,
        create_omegaconf_dict_config,
        parse_hydra_overrides,
    )
    from megatron.bridge.utils.common_utils import get_rank_safe

    args, overrides = parse_args()
    if args.wandb_step_offset < 0:
        raise ValueError("--wandb-step-offset must be non-negative")
    kwargs = {"hf_path": args.hf_path, "model_checkpoint": args.model_config_checkpoint, "output_dir": args.output_dir, "run_name": args.run_name}
    kwargs["prepared_manifest"] = args.prepared_manifest
    # Validate the actual requested batch against the selected DP topology at
    # construction time; legacy defaults (MB6/GB72) are not valid for DP8.
    for override in overrides:
        key, separator, value = override.partition("=")
        if separator and key in {"train.micro_batch_size", "train.global_batch_size"}:
            kwargs[key.split(".")[-1]] = int(value)
    cfg: ConfigContainer = nemotron_labs_diffusion_spherical_pretrain_config(**kwargs)
    merged, excluded = create_omegaconf_dict_config(cfg)
    if args.config_file:
        from omegaconf import OmegaConf

        merged = OmegaConf.merge(merged, OmegaConf.load(args.config_file))
    if overrides:
        merged = parse_hydra_overrides(merged, overrides)
    from omegaconf import OmegaConf

    apply_overrides(cfg, OmegaConf.to_container(merged, resolve=True), excluded)
    bind_runtime_protocol(cfg, hf_path=args.hf_path)
    validate_runtime_config(cfg, allow_smoke_config=args.allow_smoke_config)
    checkpoint_policy = {
        "resume_interval": cfg.checkpoint.save_interval,
        "resume_keep": cfg.checkpoint.most_recent_k,
        "model_only_interval": args.model_only_interval,
        "model_only_warmup_interval": args.model_only_warmup_interval,
        "model_only_warmup_steps": args.model_only_warmup_steps,
        "model_only_keep": "all",
        "fingerprint_validation": "warn",
    }
    if args.wandb_step_offset:
        checkpoint_policy["wandb_step_offset"] = args.wandb_step_offset
    if not args.allow_smoke_config:
        model_only_policy = (
            args.model_only_interval, args.model_only_warmup_interval, args.model_only_warmup_steps
        )
        if (
            cfg.checkpoint.save_interval != 500 or cfg.checkpoint.most_recent_k != 1
            or model_only_policy not in {(2000, 0, 0), (10000, 2000, 10000)}
        ):
            raise ValueError(
                "formal checkpoint policy requires resume every 500 (keep 1), model-only every 2000 "
                "or every 2000 through step 10000 then every 10000 (keep all)"
            )
        if cfg.checkpoint.async_save or not cfg.checkpoint.save_optim or not cfg.checkpoint.save_rng:
            raise ValueError("formal resume checkpoints require synchronous model/optimizer/RNG saves")
    validate_resume_run_config(cfg)
    _disable_unused_indexed_dataset_helper_compile()
    _install_checkpoint_strategy_compatibility()
    if get_rank_safe() == 0:
        _write_run_manifest(cfg, args.output_dir, args.run_name, checkpoint_policy=checkpoint_policy)
        logger.info("checkpoint policy: %s", checkpoint_policy)
        cfg.print_yaml()
        logger.info("native spherical entry: direct LLaCO checkpoint + pretrain")
    pretrain(
        config=cfg,
        forward_step_func=SphericalDGPTStep(seed=cfg.rng.seed),
        callbacks=[
            SphericalAcceptanceCallback(
                args.acceptance_summary,
                panel_interval=1 if args.panel_smoke else args.panel_interval,
                panel_prompts=("Explain why the sky appears blue.",) if args.panel_smoke else None,
                panel_gen_length=32 if args.panel_smoke else args.panel_gen_length,
                panel_steps=1 if args.panel_smoke else 32,
                model_only_interval=args.model_only_interval,
                model_only_warmup_interval=args.model_only_warmup_interval,
                model_only_warmup_steps=args.model_only_warmup_steps,
                model_only_load_path=args.model_only_load,
                wandb_step_offset=args.wandb_step_offset,
            )
        ],
    )


if __name__ == "__main__":
    main()
