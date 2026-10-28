"""Strict loader for the immutable Plan 3 canonical schedule."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

EXPECTED_CSV_SHA256 = "06bc39c79df164be63588bac8424a706dde6e58ab8ced7e1646a42796eeaa700"
EXPECTED_JSON_SHA256 = "536566314e982500205cbd724546ec6bd91b052b097ee74f01892d8da30a41a6"
SCHEDULER_A = 6.082463451085497
SCHEDULER_B = 1.4278641019192462
EXPECTED_ROWS = 1000
_COLUMNS = ("index", "t", "gamma", "sigma_onset_10pct_Dmax", "role")


def _exact_gamma_mask(total: int, fraction: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Choose a deterministic nearest-integer number of gamma-branch items."""
    count = int(math.floor(total * fraction + 0.5))
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    if count:
        mask[torch.randperm(total, generator=generator, device=device)[:count]] = True
    return mask


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScheduleRow:
    index: int
    t: float
    gamma: float
    sigma_10: float
    role: str


@dataclass(frozen=True)
class Plan3ScheduleManifest:
    csv_path: Path
    json_path: Path
    rows: tuple[ScheduleRow, ...]
    csv_sha256: str
    json_sha256: str
    scheduler_a: float = SCHEDULER_A
    scheduler_b: float = SCHEDULER_B

    @classmethod
    def load(cls, csv_path: str | Path, json_path: str | Path, *, verify_archive_hash: bool = True):
        csv_path, json_path = Path(csv_path), Path(json_path)
        if not csv_path.is_file() or not json_path.is_file():
            raise FileNotFoundError("Plan 3 schedule requires both canonical CSV and JSON")
        csv_hash, json_hash = sha256_file(csv_path), sha256_file(json_path)
        if verify_archive_hash and (csv_hash != EXPECTED_CSV_SHA256 or json_hash != EXPECTED_JSON_SHA256):
            raise ValueError("canonical Plan 3 schedule SHA-256 mismatch")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _COLUMNS:
                raise ValueError(f"schedule columns must be {_COLUMNS}")
            csv_rows = [
                ScheduleRow(int(r["index"]), float(r["t"]), float(r["gamma"]), float(r["sigma_onset_10pct_Dmax"]), r["role"])
                for r in reader
            ]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows_payload = payload.get("rows")
        if not isinstance(rows_payload, list):
            raise ValueError("schedule JSON must contain rows")
        params = payload.get("gamma_scheduler", {}).get("parameters", [])
        if len(params) != 2 or not math.isclose(float(params[0]), SCHEDULER_A, rel_tol=0, abs_tol=1e-14) or not math.isclose(float(params[1]), SCHEDULER_B, rel_tol=0, abs_tol=1e-14):
            raise ValueError("unexpected Plan 3 semantic scheduler parameters")
        if len(csv_rows) != EXPECTED_ROWS or len(rows_payload) != EXPECTED_ROWS:
            raise ValueError("Plan 3 canonical schedule must have exactly 1000 rows")
        for row, raw in zip(csv_rows, rows_payload):
            expected = (row.index, row.t, row.gamma, row.sigma_10, row.role)
            actual = (int(raw["index"]), float(raw["t"]), float(raw["gamma"]), float(raw["sigma_onset_10pct_Dmax"]), raw["role"])
            if expected[0] != actual[0] or expected[4] != actual[4] or any(a != b for a, b in zip(expected[1:4], actual[1:4])):
                raise ValueError(f"CSV/JSON schedule row mismatch at index {row.index}")
            if row.index != int(round(row.t * 999)):
                raise ValueError("schedule index does not equal floor(t*999) grid")
            if not all(math.isfinite(x) for x in (row.t, row.gamma, row.sigma_10)):
                raise ValueError("schedule contains non-finite value")
            expected_gamma = 1.0 - (1.0 - row.gamma ** SCHEDULER_A) ** SCHEDULER_B if 0 < row.gamma < 1 else row.t
            if not math.isclose(expected_gamma, row.t, rel_tol=0, abs_tol=2e-12):
                raise ValueError("gamma column does not match frozen semantic scheduler")
        if csv_rows[0] != ScheduleRow(0, 0.0, 0.0, 4.716463788366068, csv_rows[0].role):
            raise ValueError("invalid t=0 schedule endpoint")
        if csv_rows[-1].index != 999 or csv_rows[-1].t != 1.0 or csv_rows[-1].gamma != 1.0 or csv_rows[-1].sigma_10 != 0.0:
            raise ValueError("invalid t=1 schedule endpoint")
        if any(a.t >= b.t or a.gamma >= b.gamma or a.sigma_10 < b.sigma_10 for a, b in zip(csv_rows, csv_rows[1:])):
            raise ValueError("schedule must increase in t/gamma and decrease in sigma")
        return cls(csv_path, json_path, tuple(csv_rows), csv_hash, json_hash)

    def lookup_indices(self, indices: Iterable[int] | torch.Tensor):
        idx = torch.as_tensor(indices, dtype=torch.long)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= len(self.rows)):
            raise IndexError("canonical schedule index out of range")
        selected = [self.rows[int(i)] for i in idx.reshape(-1).tolist()]
        return {"index": idx.clone(), "t": torch.tensor([r.t for r in selected], dtype=torch.float64).reshape(idx.shape), "gamma": torch.tensor([r.gamma for r in selected], dtype=torch.float64).reshape(idx.shape), "sigma_10": torch.tensor([r.sigma_10 for r in selected], dtype=torch.float64).reshape(idx.shape), "role": tuple(r.role for r in selected)}

    def sample_rows(self, generator: torch.Generator, shape=(), *, uniform_gamma_fraction: float = 0.30):
        """Sample discrete-uniform canonical ``t`` or continuous gamma rows.

        The t branch is discrete uniform over all canonical rows. The gamma
        branch draws continuous ``U[0, 1]`` gamma and maps to nearest gamma.
        """
        if not 0.0 <= uniform_gamma_fraction <= 1.0:
            raise ValueError("uniform_gamma_fraction must be in [0, 1]")
        shape = tuple(shape)
        device = generator.device
        total = math.prod(shape) if shape else 1
        # Allocate the branch count exactly (to the nearest integer) for each
        # sampled batch, rather than relying on a Bernoulli draw whose realized
        # fraction can vary substantially on short batches.
        use_gamma = _exact_gamma_mask(total, uniform_gamma_fraction, generator, device)
        sampled = torch.empty(total, dtype=torch.long, device=device)

        def nearest_coordinate(mask, grid):
            count = int(mask.sum())
            if count == 0:
                return
            value = torch.rand(count, generator=generator, device=device)
            upper = torch.searchsorted(grid, value).clamp_max(len(self.rows) - 1)
            lower = (upper - 1).clamp_min(0)
            choose_upper = (value - grid[lower]) >= (grid[upper] - value)
            sampled[mask] = torch.where(choose_upper, upper, lower)

        gamma_grid = torch.tensor([row.gamma for row in self.rows], dtype=torch.float32, device=device)
        t_mask = ~use_gamma
        t_count = int(t_mask.sum())
        if t_count:
            sampled[t_mask] = torch.randint(
                len(self.rows), (t_count,), generator=generator, device=device, dtype=torch.long
            )
        nearest_coordinate(use_gamma, gamma_grid)
        return sampled.reshape(shape)

    def metadata(self):
        return {"schema_version": "plan3-canonical-schedule-v1", "csv_path": str(self.csv_path), "json_path": str(self.json_path), "csv_sha256": self.csv_sha256, "json_sha256": self.json_sha256, "rows": len(self.rows), "index": "t", "scheduler": {"family": "kumaraswamy_survival", "a": self.scheduler_a, "b": self.scheduler_b}}
