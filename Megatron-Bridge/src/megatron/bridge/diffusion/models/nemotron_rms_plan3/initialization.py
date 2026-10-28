"""Immutable base-model and Plan 3 initialization locks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def _canonical_embedding_sha256(weight: torch.Tensor) -> str:
    """Hash the BF16 embedding representation consumed by the trainer."""
    digest = hashlib.sha256()
    for start in range(0, weight.shape[0], 1024):
        value = weight[start : start + 1024].detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _payload_sha256(payload: dict) -> str:
    value = {key: item for key, item in payload.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_base_model_lock(base_dir: str | Path, output: str | Path) -> dict:
    """Create a lock covering every loadable base artifact and BF16 embedding."""
    base = Path(base_dir).resolve()
    # Include remote-code Python modules as well as weights/config/templates;
    # Transformers imports these files when trust_remote_code is enabled.
    files = sorted(path for path in base.iterdir() if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors", ".py"})
    model_file = base / "model.safetensors"
    if not model_file.is_file():
        raise ValueError("Plan 3 base lock requires model.safetensors")
    from safetensors import safe_open

    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        key = "encoder.embed_tokens.weight"
        if key not in handle.keys():
            raise ValueError("Plan 3 base lock cannot find encoder.embed_tokens.weight")
        embedding = handle.get_tensor(key)
    payload = {
        "namespace": "nemotron_rms_plan3",
        "algorithm_version": "plan3-base-lock-v2-bf16-row1024",
        "base_dir": str(base),
        "files": {path.name: file_sha256(path) for path in files},
        "embedding_key": key,
        "embedding_shape": list(embedding.shape),
        "embedding_dtype": "bfloat16_row_chunk_1024",
        "embedding_sha256": _canonical_embedding_sha256(embedding),
    }
    payload["sha256"] = _payload_sha256(payload)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload

def validate_base_model_lock_files(base_dir: str | Path, lock_path: str | Path) -> dict:
    """Verify the lock payload, canonical base path, and every locked file."""
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("namespace") != "nemotron_rms_plan3" or lock.get("algorithm_version") != "plan3-base-lock-v2-bf16-row1024":
        raise ValueError("base-model lock is not the current Plan 3 lock format")
    if lock.get("sha256") != _payload_sha256(lock):
        raise ValueError("base-model lock self hash mismatch")
    base = Path(base_dir).resolve()
    if Path(lock.get("base_dir", "")).resolve() != base:
        raise ValueError("base-model lock points to a different base directory")
    lock_files = set(lock.get("files", {}))
    current_files = {
        path.name
        for path in base.iterdir()
        if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors", ".py"}
    }
    if current_files != lock_files:
        missing = sorted(lock_files - current_files)
        extra = sorted(current_files - lock_files)
        raise ValueError(f"base-model artifact set changed (missing={missing}, extra={extra})")
    for name, expected in lock.get("files", {}).items():
        path = base / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"base-model lock mismatch for {name}")
    required_remote_code = {"configuration_nemotron_labs_diffusion.py", "modeling_nemotron_labs_diffusion.py"}
    missing_remote_code = sorted(name for name in required_remote_code if name not in lock.get("files", {}))
    if missing_remote_code:
        raise ValueError(f"base-model lock omits required remote-code files: {', '.join(missing_remote_code)}")
    return lock


def validate_base_model_lock(base_dir: str | Path, lock_path: str | Path, embedding_weight: torch.Tensor) -> dict:
    """Verify the lock, source files, and the embedding loaded by Transformers."""
    lock = validate_base_model_lock_files(base_dir, lock_path)
    if list(embedding_weight.shape) != lock.get("embedding_shape"):
        raise ValueError("loaded input embedding shape differs from base lock")
    if _canonical_embedding_sha256(embedding_weight) != lock.get("embedding_sha256"):
        raise ValueError("loaded input embedding differs from base lock")
    return lock


def initialization_sha256(module: torch.nn.Module, base_lock_sha256: str) -> str:
    """Hash Plan 3-only trainable initialization while the base lock covers the base."""
    digest = hashlib.sha256()
    digest.update(base_lock_sha256.encode("ascii"))
    for name, value in sorted(module.named_parameters()):
        if value.requires_grad and not name.startswith("base."):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_initialization_lock_binding(lock_path: str | Path, base_lock_sha256: str) -> dict:
    """Verify that an initialization lock is current and bound to the base lock."""
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("namespace") != "nemotron_rms_plan3" or lock.get("algorithm_version") != "plan3-adapter-initialization-v1":
        raise ValueError("initialization lock is not the current Plan 3 lock format")
    if lock.get("base_model_lock_sha256") != base_lock_sha256:
        raise ValueError("initialization lock points to a different base-model lock")
    digest = lock.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("initialization lock has an invalid initialization hash")
    return lock


def validate_or_create_initialization_lock(
    module: torch.nn.Module, base_lock_sha256: str, lock_path: str | Path
) -> dict:
    """Create once or verify the immutable step-zero Plan 3 adapter initialization."""
    path = Path(lock_path)
    actual = initialization_sha256(module, base_lock_sha256)
    if path.exists():
        lock = validate_initialization_lock_binding(path, base_lock_sha256)
        if lock.get("sha256") != actual:
            raise ValueError("Plan 3 initialization lock mismatch")
        return lock
    payload = {
        "namespace": "nemotron_rms_plan3",
        "algorithm_version": "plan3-adapter-initialization-v1",
        "base_model_lock_sha256": base_lock_sha256,
        "sha256": actual,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
