"""Shared BF16 runtime for LLaCO inference and training tools."""
from __future__ import annotations

import sys
import yaml
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / "Megatron-Bridge"
MODEL_CODE = BRIDGE_ROOT / "src/megatron/bridge/diffusion/models/nemotron_labs_diffusion_spherical"


def configure_local_sources() -> None:
    """Install the vendored Bridge/Megatron source trees before imports."""
    for path in (BRIDGE_ROOT / "src", BRIDGE_ROOT / "3rdparty" / "Megatron-LM"):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))


def register_checkpoint_config_alias(checkpoint: Path, hf_config) -> None:
    """Map the serialized HF dynamic-module name to the local config class."""
    run_config = yaml.safe_load((checkpoint / "run_config.yaml").read_text())
    target = run_config["model"]["hf_config"]["_target_"]
    class_path = target.removesuffix(".from_dict")
    serialized_module, _, serialized_class = class_path.rpartition(".")
    if serialized_class != type(hf_config).__name__:
        raise RuntimeError("checkpoint HF config does not match the local tokenizer config")
    current_module = sys.modules[type(hf_config).__module__]
    sys.modules[serialized_module] = current_module


def load_bf16_model(*, checkpoint: Path, tokenizer_path: Path, tp: int = 1, block_size: int = 4):
    """Load a model-only checkpoint through the original Bridge loader path.

    The project intentionally uses ``build_and_load_model`` here. The custom
    evaluator loader is not part of the public runtime.
    """
    configure_local_sources()
    import torch.distributed as dist
    from transformers import AutoConfig, AutoTokenizer
    from megatron.core.utils import unwrap_model
    from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config
    from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.decode_protocol import copy_decode_fields
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.provider import (
        initialize_spherical_components,
        install_spherical_transformer,
    )
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion_spherical.checkpoint import load_spherical_metadata

    if tp <= 0:
        raise ValueError("tp must be positive")
    register_allowed_target_prefix("transformers_modules.")
    hf_config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    register_checkpoint_config_alias(checkpoint, hf_config)
    cfg, _ = load_model_config(str(checkpoint))
    copy_decode_fields(cfg, load_spherical_metadata(checkpoint))
    cfg.tensor_model_parallel_size = tp
    cfg.pipeline_model_parallel_size = cfg.context_parallel_size = 1
    cfg.expert_model_parallel_size = cfg.expert_tensor_parallel_size = 1
    cfg.sequence_parallel = False
    cfg.bf16 = True
    cfg.fp16 = False
    cfg.params_dtype = torch.bfloat16
    cfg.pipeline_dtype = torch.bfloat16
    cfg.autocast_dtype = torch.bfloat16
    cfg.perform_initialization = False
    cfg.spherical_compute_dtype = "fp32"
    cfg.spherical_attention_backend = "explicit_fp32"
    cfg.gradient_accumulation_fusion = False
    cfg.cross_entropy_loss_fusion = False
    cfg.register_pre_wrap_hook(lambda models: initialize_spherical_components(models, block_size=block_size, compute_dtype="fp32"))
    cfg.register_pre_wrap_hook(install_spherical_transformer)
    cfg.initialize_model_parallel(seed=1234, create_gloo_process_groups=False)
    model = build_and_load_model(str(checkpoint), cfg, skip_temp_dist_context=True)
    core = unwrap_model(model[0] if isinstance(model, list) else model)
    core = getattr(core, "language_model", core).cuda().eval()
    core.spherical_geometry.refresh_codebook()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return core, tokenizer, cfg, dist
