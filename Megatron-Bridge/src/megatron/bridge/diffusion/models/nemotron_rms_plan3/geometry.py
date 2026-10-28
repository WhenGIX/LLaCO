"""Frozen RMS clean-to-mask geometry for Plan 3."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import torch

EXPECTED_R = 0.005589511466413362
EXPECTED_MASK_ID = 100
EXPECTED_VOCAB_SIZE = 131072
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_STAGE1_MANIFEST_SHA256 = "1367e86ea07af36016eb24db131e219a1f843e098a500dd6dcc2b93eb74efd02"

def ids_sha256(ids):
    return hashlib.sha256(torch.as_tensor(ids, dtype=torch.int64).cpu().numpy().tobytes()).hexdigest()


def embedding_sha256(weight: torch.Tensor) -> str:
    """Hash the exact BF16 rows used by Plan 3 geometry and decoding."""
    digest = hashlib.sha256()
    value = weight.detach()
    for start in range(0, value.shape[0], 1024):
        rows = value[start : start + 1024].to(torch.bfloat16).contiguous().view(torch.uint16).cpu()
        digest.update(rows.numpy().tobytes())
    return digest.hexdigest()

def compute_v_rms(embedding_weight, mask_token_id=EXPECTED_MASK_ID):
    w = embedding_weight.detach().float()
    finite = torch.isfinite(w).all(dim=1)
    norms = w.norm(dim=1)
    ids = torch.arange(w.shape[0], device=w.device)[finite & (norms > 0)]
    ids = ids[ids != int(mask_token_id)]
    if ids.numel() == 0:
        raise ValueError("V_rms is empty")
    return ids

@dataclass(frozen=True)
class Plan3Geometry:
    rms: torch.Tensor
    mask_state: torch.Tensor
    mask_token_id: int
    vocab_size: int
    hidden_size: int
    v_rms: torch.Tensor
    v_pred: torch.Tensor
    embedding_sha256: str

    @classmethod
    def from_embedding(cls, embedding_weight, mask_token_id=EXPECTED_MASK_ID, *, expected_rms=None, v_rms=None):
        if embedding_weight.ndim != 2:
            raise ValueError("embedding table must be rank 2")
        vocab, hidden = embedding_weight.shape
        computed_ids = compute_v_rms(embedding_weight, mask_token_id)
        ids = computed_ids if v_rms is None else torch.as_tensor(v_rms, dtype=torch.long, device=embedding_weight.device)
        if int(mask_token_id) in set(ids.tolist()) or ids.numel() != vocab - 1:
            raise ValueError("V_rms must contain every finite non-zero row except mask")
        if not torch.equal(ids, computed_ids):
            raise ValueError("V_rms must use the deterministic token-id ordering of the embedding")
        w = embedding_weight.detach().float().index_select(0, ids)
        rms = w.square().mean().sqrt()
        if expected_rms is not None and not torch.isclose(rms, torch.as_tensor(expected_rms), rtol=1e-6, atol=1e-8):
            raise ValueError(f"RMS mismatch: computed {float(rms)} expected {expected_rms}")
        mask = embedding_weight[int(mask_token_id)].detach().float()
        if not torch.isfinite(mask).all():
            raise ValueError("mask embedding must be finite")
        return cls(
            rms,
            mask / rms,
            int(mask_token_id),
            int(vocab),
            int(hidden),
            ids.detach().cpu(),
            ids.detach().cpu(),
            embedding_sha256(embedding_weight),
        )

    def clean(self, embedding_weight, token_ids):
        return embedding_weight[token_ids] / self.rms.to(embedding_weight.device, embedding_weight.dtype)

    def state(self, embedding_weight, token_ids, gamma):
        clean = self.clean(embedding_weight, token_ids)
        g = torch.as_tensor(gamma, device=clean.device, dtype=clean.dtype)
        while g.ndim < clean.ndim: g = g.unsqueeze(-1)
        return (1 - g) * clean + g * self.mask_state.to(clean.device, clean.dtype)

    def to_model(self, state):
        return state * self.rms.to(state.device, state.dtype)

    def metadata(self):
        return {
            "type": "rms_clean_to_mask_deterministic",
            "input_embedding_frozen": True,
            "embedding_sha256": self.embedding_sha256,
            "R": float(self.rms),
            "mask_token_id": self.mask_token_id,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "V_rms": {"definition": "finite_nonzero_rows_minus_mask", "ids_sha256": ids_sha256(self.v_rms), "count": int(self.v_rms.numel())},
            "V_pred": {"definition": "equal_to_V_rms_ordered_by_token_id", "ids_sha256": ids_sha256(self.v_pred), "count": int(self.v_pred.numel())},
        }


def validate_geometry_archive(geometry: Plan3Geometry, manifest_path: str | Path) -> str:
    """Validate the frozen stage-1 geometry archive against a loaded embedding."""
    path = Path(manifest_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_STAGE1_MANIFEST_SHA256:
        raise ValueError("Plan 3 stage-1 geometry manifest SHA-256 mismatch")
    value = json.loads(path.read_text())
    if (
        float(value.get("global_rms_R", "nan")) != EXPECTED_R
        or int(value.get("mask_id", -1)) != geometry.mask_token_id
        or int(value.get("vocab_size", -1)) != geometry.vocab_size
        or int(value.get("hidden_size", -1)) != geometry.hidden_size
        or int(value.get("rms_row_count_V", -1)) != geometry.v_rms.numel()
    ):
        raise ValueError("loaded geometry does not match the Plan 3 stage-1 archive")
    if not torch.isclose(geometry.rms.float().cpu(), torch.tensor(EXPECTED_R), rtol=1e-6, atol=1e-8):
        raise ValueError("loaded embedding RMS does not match the Plan 3 archive")
    return digest
