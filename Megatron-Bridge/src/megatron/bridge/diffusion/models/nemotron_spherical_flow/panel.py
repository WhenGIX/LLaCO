"""Shared fixed-panel loading and provenance helpers for spherical inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..nemotron_rms_plan3.prompting import render_user_prompt

PANEL_CATEGORIES = ("choice", "math", "code", "general")


def _canonical_payload(panel: dict) -> bytes:
    value = {"panel_id": panel["panel_id"], "source": panel.get("source", {}), "rows": panel["rows"]}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def panel_sha256(panel: dict) -> str:
    return hashlib.sha256(_canonical_payload(panel)).hexdigest()


def load_fixed_panel(path: str | Path) -> dict:
    """Load a four-row panel and reject ambiguous or malformed inputs."""
    file = Path(path)
    panel = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(panel, dict) or not isinstance(panel.get("rows"), list):
        raise ValueError(f"fixed panel must contain a rows list: {file}")
    rows = panel["rows"]
    if len(rows) != len(PANEL_CATEGORIES):
        raise ValueError("fixed panel must contain exactly four rows")
    if [row.get("category") if isinstance(row, dict) else None for row in rows] != list(PANEL_CATEGORIES):
        raise ValueError("fixed panel categories must be choice, math, code, general in that order")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("example_id"), str) or not row["example_id"]:
            raise ValueError("fixed panel rows require non-empty example_id values")
        if not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError("fixed panel rows require non-empty question values")
        ids.append(row["example_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("fixed panel example_id values must be unique")
    if not isinstance(panel.get("panel_id"), str) or not panel["panel_id"]:
        raise ValueError("fixed panel requires a panel_id")
    digest = panel_sha256(panel)
    recorded = panel.get("manifest_sha256")
    if recorded is not None and recorded != digest:
        raise ValueError("fixed panel manifest_sha256 does not match its contents")
    return {"panel_id": panel["panel_id"], "source": panel.get("source", {}), "rows": rows, "manifest_sha256": digest}


def derive_block_seed(
    base_seed: int, example_id: str, block_index: int, panel_id: str | None = None
) -> int:
    payload = f"spherical-panel-v1|{int(base_seed)}|{panel_id or ''}|{example_id}|{int(block_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def render_panel_prompt(tokenizer, question: str, *, device=None):
    return render_user_prompt(tokenizer, question, device=device)


__all__ = ["PANEL_CATEGORIES", "derive_block_seed", "load_fixed_panel", "panel_sha256", "render_panel_prompt"]
