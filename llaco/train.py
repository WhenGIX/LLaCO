"""BF16 LLaCO training entry wrapping the native Bridge recipe."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from . import TRAINING_BLOCK_SIZE

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "Megatron-Bridge" / "examples" / "models" / "nemotron_labs_diffusion_spherical" / "pretrain.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="LLaCO model-only checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer and HF configuration directory")
    parser.add_argument("--prepared-manifest", required=True, help="Manifest written by scripts/prepare_spherical_data.py")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "train"))
    parser.add_argument("--run-name", default="llaco_bf16_train")
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--global-batch", type=int)
    args, rest = parser.parse_known_args()
    os.environ.setdefault("SPHERICAL_TRAIN_BLOCK_SIZE", str(TRAINING_BLOCK_SIZE))
    os.environ.setdefault("SPHERICAL_TP_SIZE", "1")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp = int(os.environ["SPHERICAL_TP_SIZE"])
    if world % tp:
        parser.error("WORLD_SIZE must be divisible by SPHERICAL_TP_SIZE")
    os.environ.setdefault("SPHERICAL_DP_SIZE", str(world // tp))
    os.environ.setdefault("SPHERICAL_PP_SIZE", "1")
    os.environ.setdefault("SPHERICAL_CP_SIZE", "1")
    os.environ.setdefault("SPHERICAL_PRECISION_PROFILE", "h100")
    os.environ.setdefault("SPHERICAL_FULL_FP32", "0")
    os.environ.setdefault("SPHERICAL_FRESH_BASE", "1")
    os.environ.setdefault("SPHERICAL_RESUME_STEP", "0")
    dp = int(os.environ.get("SPHERICAL_DP_SIZE", "8"))
    global_batch = args.global_batch or (dp * args.micro_batch)
    overrides = [
        "model.block_size=8", "dataset.block_size=8",
        "model.spherical_compute_dtype=fp32",
        "mixed_precision.bf16=true", "mixed_precision.fp16=false",
        "mixed_precision.grad_reduce_in_fp32=true", "ddp.grad_reduce_in_fp32=true",
        f"train.micro_batch_size={args.micro_batch}",
        f"train.global_batch_size={global_batch}",
        f"train.train_iters={args.train_iters}",
        f"scheduler.lr_decay_iters={args.train_iters}",
    ]
    if os.environ.get("SPHERICAL_CAPACITY_STEPS"):
        overrides += [
            "checkpoint.save=null", "checkpoint.save_interval=0",
            "logger.wandb_project=null", "logger.wandb_exp_name=null",
            "logger.tensorboard_dir=null",
        ]
    metadata = json.loads((Path(args.checkpoint) / "spherical_metadata.json").read_text())
    for key in ("spherical_condition_dim", "spherical_condition_coordinate",
                "spherical_ada_activation", "spherical_time_embedding_scale",
                "spherical_condition_compute_dtype", "spherical_condition_norm"):
        if key in metadata:
            overrides.append(f"model.{key}={metadata[key]}")
    overrides += ["checkpoint.load=null", "checkpoint.pretrained_checkpoint=null"]
    from .runtime import configure_local_sources
    configure_local_sources()
    from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
    register_allowed_target_prefix("transformers_modules.")
    argv = [
        str(ENTRY), "--hf-path", args.tokenizer, "--model-config-checkpoint", args.checkpoint,
        "--prepared-manifest", args.prepared_manifest,
        "--output-dir", args.output_dir, "--run-name", args.run_name,
        "--allow-smoke-config", "--panel-interval", "0",
        "--model-only-interval", "0", "--model-only-warmup-interval", "0",
        "--model-only-warmup-steps", "0", "--model-only-load", args.checkpoint,
        *overrides, *rest,
    ]
    sys.argv = argv
    runpy.run_path(str(ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
