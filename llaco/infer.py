"""BF16 LLaCO inference CLI."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .runtime import configure_local_sources, load_bf16_model
from . import INFERENCE_BLOCK_SIZE, INFERENCE_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default='A bookstore sells notebooks for $4 each. Emma buys 3 notebooks and pays with a $20 bill. How much change should she receive? Show your reasoning and put the final answer in \\boxed{}.')
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=INFERENCE_BLOCK_SIZE)
    parser.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.gen_length % args.block_size:
        parser.error("gen-length must be a multiple of block-size")
    configure_local_sources()
    import torch
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.inference import spherical_block_sample

    core, tokenizer, cfg, dist = load_bf16_model(
        checkpoint=args.checkpoint, tokenizer_path=args.tokenizer, tp=args.tp, block_size=args.block_size
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )
    ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).cuda()
    started = time.monotonic()
    with torch.inference_mode():
        output, nfe = spherical_block_sample(
            core, core.spherical_geometry, core.spherical_adapter, ids,
            gen_length=args.gen_length, block_size=args.block_size, steps=args.steps,
            mask_token_id=int(cfg.mask_token_id), eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, seed=args.seed, deterministic=True,
        )
    text = tokenizer.decode(output[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
    result = {
        "precision": "bf16", "checkpoint": str(args.checkpoint), "prompt": args.prompt,
        "text": text, "nfe": nfe, "elapsed_seconds": time.monotonic() - started,
        "gen_length": args.gen_length, "block_size": args.block_size, "steps": args.steps,
    }
    if dist.get_rank() == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
