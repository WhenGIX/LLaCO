# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading and validation for offline RMS identity statistics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .geometry import compute_global_rms
from .scheduler import IdentityScheduler


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StatisticsManifest:
    """Validated offline geometry statistics and scheduler fit metadata."""

    terminal_mode: str
    statistics_path: Path
    fit_path: Path
    statistics_sha256: str
    fit_sha256: str
    global_rms: float
    r_mask: float
    b_noise: float
    mask_token_id: int
    vocab_size: int
    source_token_count: int
    rms_excluded_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    scheduler: IdentityScheduler
    raw_statistics_metadata: dict
    raw_fits: dict

    @classmethod
    def load(cls, statistics_path: str | Path, fit_path: str | Path, terminal_mode: str):
        statistics_path = Path(statistics_path)
        fit_path = Path(fit_path)
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        fits = json.loads(fit_path.read_text(encoding="utf-8"))
        metadata = statistics["metadata"]
        gamma_grid = metadata.get("gamma_grid")
        if gamma_grid is not None and (
            int(gamma_grid.get("points", 0)) != 1002 or not bool(gamma_grid.get("includes_endpoints"))
        ):
            raise ValueError("statistics gamma grid must contain 1002 points including endpoints")
        sampling = metadata.get("sampling")
        if sampling is not None and (
            int(sampling.get("gammas_per_token", 0)) != 3
            or not bool(sampling.get("distinct_per_token"))
            or not bool(sampling.get("independent_noise_per_pair"))
        ):
            raise ValueError("statistics sampling does not match the documented RMS protocol")
        expected_name = {
            "scale-matched": "bmask_scale_matched",
            "unit-noise": "bmask_1",
            "equal-mask-rms": "bmask_equal_mask_rms",
        }.get(terminal_mode)
        if expected_name is None:
            raise ValueError(f"unsupported RMS terminal mode {terminal_mode!r}")
        if metadata.get("bmask_name") != expected_name:
            raise ValueError(f"statistics terminal {metadata.get('bmask_name')!r} does not match {terminal_mode!r}")
        if terminal_mode == "scale-matched" and float(metadata["r_mask"]) > 1.0:
            raise ValueError("scale-matched statistics are invalid because r_mask > 1")
        r_mask = float(metadata["r_mask"])
        b_noise = float(metadata["bmask"])
        if not math.isfinite(r_mask) or r_mask < 0:
            raise ValueError("statistics r_mask must be finite and non-negative")
        expected_b = (
            math.sqrt(1.0 - r_mask * r_mask)
            if terminal_mode == "scale-matched"
            else r_mask
            if terminal_mode == "equal-mask-rms"
            else 1.0
        )
        if not math.isfinite(b_noise) or not math.isclose(b_noise, expected_b, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("statistics b_noise is inconsistent with r_mask and terminal mode")
        p_fit = fits.get("P_kumaraswamy_survival")
        if not isinstance(p_fit, dict) or not isinstance(p_fit.get("parameters"), dict):
            raise ValueError("statistics fit must include P_kumaraswamy_survival parameters")
        p_parameters = p_fit["parameters"]
        if any(
            not math.isfinite(float(p_parameters.get(name, float("nan")))) or float(p_parameters[name]) <= 0
            for name in ("a", "b")
        ):
            raise ValueError("P_kumaraswamy_survival parameters must be finite and positive")
        parameters = fits["A_kumaraswamy_survival"]["parameters"]
        scheduler = IdentityScheduler(terminal_mode, float(parameters["a"]), float(parameters["b"]))
        registered = IdentityScheduler.for_terminal(terminal_mode)
        if not math.isclose(scheduler.a, registered.a, rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
            scheduler.b, registered.b, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("statistics fit does not match the registered terminal scheduler")
        return cls(
            terminal_mode=terminal_mode,
            statistics_path=statistics_path.resolve(),
            fit_path=fit_path.resolve(),
            statistics_sha256=file_sha256(statistics_path),
            fit_sha256=file_sha256(fit_path),
            global_rms=float(metadata["global_rms"]),
            r_mask=r_mask,
            b_noise=b_noise,
            mask_token_id=int(metadata["mask_id"]),
            vocab_size=int(metadata["vocab_size"]),
            source_token_count=int(metadata["source_token_count"]),
            rms_excluded_ids=tuple(map(int, metadata.get("rms_excluded_ids", metadata.get("excluded_ids", [])))),
            excluded_ids=tuple(map(int, metadata.get("identity_excluded_ids", metadata.get("zero_norm_excluded_ids", [])))),
            scheduler=scheduler,
            raw_statistics_metadata=metadata,
            raw_fits=fits,
        )

    @classmethod
    def from_root(cls, root: str | Path, terminal_mode: str):
        directory_name = {
            "scale-matched": "bmask_scale_matched",
            "unit-noise": "bmask_1",
            "equal-mask-rms": "bmask_equal_mask_rms",
        }.get(terminal_mode)
        if directory_name is None:
            raise ValueError(f"unsupported RMS terminal mode {terminal_mode!r}")
        directory = Path(root) / directory_name
        fit_name = "fits_kumaraswamy.json" if terminal_mode == "equal-mask-rms" else "fits_theoretical.json"
        return cls.load(directory / "statistics.json", directory / fit_name, terminal_mode)

    def validate_embedding(self, embedding_weight: torch.Tensor, mask_token_id: int, atol=1e-7):
        if embedding_weight.shape[0] != self.vocab_size:
            raise ValueError("statistics vocabulary size does not match the embedding table")
        if int(mask_token_id) != self.mask_token_id:
            raise ValueError("statistics mask token does not match the model")
        rms = float(compute_global_rms(embedding_weight, self.rms_ids(device=embedding_weight.device)))
        mask_state = embedding_weight[mask_token_id].detach().float() / rms
        r_mask = float(mask_state.square().mean().sqrt())
        if not math.isclose(rms, self.global_rms, rel_tol=1e-6, abs_tol=atol):
            raise ValueError(f"statistics R={self.global_rms:.10g} does not match embedding R={rms:.10g}")
        if not math.isclose(r_mask, self.r_mask, rel_tol=1e-6, abs_tol=atol):
            raise ValueError("statistics r_mask does not match the embedding table")
        if self.terminal_mode == "scale-matched" and r_mask > 1.0:
            raise ValueError("scale-matched statistics are invalid because embedding r_mask > 1")
        if not math.isfinite(self.global_rms) or self.global_rms <= 0:
            raise ValueError("statistics global_rms must be finite and positive")
        expected_b = (
            math.sqrt(1.0 - r_mask * r_mask)
            if self.terminal_mode == "scale-matched"
            else r_mask
            if self.terminal_mode == "equal-mask-rms"
            else 1.0
        )
        if not math.isclose(expected_b, self.b_noise, rel_tol=1e-6, abs_tol=atol):
            raise ValueError("statistics b_noise is inconsistent with r_mask and terminal mode")

    def prediction_ids(self, device=None) -> torch.Tensor:
        excluded = {self.mask_token_id, *self.excluded_ids}
        ids = [token_id for token_id in range(self.vocab_size) if token_id not in excluded]
        if len(ids) != self.source_token_count:
            raise ValueError("statistics source-token count is inconsistent with its exclusions")
        return torch.tensor(ids, dtype=torch.long, device=device)

    def rms_ids(self, device=None) -> torch.Tensor:
        """Return the exact rows used by the manifest's global RMS definition."""
        excluded = set(self.rms_excluded_ids)
        ids = [token_id for token_id in range(self.vocab_size) if token_id not in excluded]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def metadata(self) -> dict:
        return {
            "statistics_manifest": str(self.statistics_path),
            "statistics_manifest_sha256": self.statistics_sha256,
            "statistics_fit": str(self.fit_path),
            "statistics_fit_sha256": self.fit_sha256,
            "statistics_terminal_mode": self.terminal_mode,
            "statistics_sampling": self.raw_statistics_metadata.get("sampling"),
            "statistics_gamma_grid": self.raw_statistics_metadata.get("gamma_grid"),
            "statistics_seed": self.raw_statistics_metadata.get("seed"),
            "excluded_token_ids": list(self.excluded_ids),
            "rms_excluded_token_ids": list(self.rms_excluded_ids),
            "fit_targets": {
                key: value.get("parameters")
                for key, value in self.raw_fits.items()
                if isinstance(value, dict) and "parameters" in value
            },
        }


__all__ = ["StatisticsManifest", "file_sha256"]
