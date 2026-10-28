"""Strict Plan 3 metadata and checkpoint compatibility helpers."""
from __future__ import annotations
import hashlib
import json
from .inference import NODE_SELECTOR_VERSION

NAMESPACE = "nemotron_rms_plan3"
FORMAT_VERSION = 1
_SOLVER_CONTRACT = {"base_dllm_steps_per_block": 32, "plan3_update_count": 32, "plan3_nfe": 33, "plan3_final_t": 0.01, "node_selector_version": NODE_SELECTOR_VERSION}
_NOISE_FORMULAS = {
    "A_direct_onset": "x_t=s_t+sigma_10(t)*epsilon",
    "B_tapered_onset": "x_t=s_t+t*sigma_10(t)*epsilon",
}
_SCHEDULER = ("kumaraswamy_survival", 6.082463451085497, 1.4278641019192462)

def stable_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def build_metadata(*, geometry, schedule, mode, data_manifest_sha256, base_model_lock_sha256, comparison_group_id, initialization_sha256, training_protocol_sha256, solver_source_sha256, posterior_mode="full", posterior_k=None):
    if mode not in {"A_direct_onset", "B_tapered_onset"}: raise ValueError("invalid Plan 3 noise mode")
    formula = "x_t=s_t+sigma_10(t)*epsilon" if mode == "A_direct_onset" else "x_t=s_t+t*sigma_10(t)*epsilon"
    return {"namespace": NAMESPACE, "format_version": FORMAT_VERSION, "geometry": geometry.metadata(), "semantic_scheduler": {"conditioning": "t", "family": "kumaraswamy_survival", "a": schedule.scheduler_a, "b": schedule.scheduler_b}, "canonical_schedule": schedule.metadata(), "observation_noise": {"training_mode": mode, "formula": formula}, "comparison": {"comparison_group_id": comparison_group_id, "initialization_sha256": initialization_sha256}, "base_model_lock": {"sha256": base_model_lock_sha256}, "training_protocol_sha256": training_protocol_sha256, "data_manifest_sha256": data_manifest_sha256, "solver": {"source_sha256": solver_source_sha256, **_SOLVER_CONTRACT}, "infer_final_t": 0.01, "posterior": {"mode": posterior_mode, "k": posterior_k}}

def validate_metadata(saved, expected, *, inference_only=False):
    if saved.get("namespace") != NAMESPACE or saved.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint is not a Plan 3 checkpoint")
    noise = saved.get("observation_noise")
    if not isinstance(noise, dict) or noise.get("training_mode") not in _NOISE_FORMULAS:
        raise ValueError("Plan 3 metadata has an invalid observation-noise mode")
    if noise.get("formula") != _NOISE_FORMULAS[noise["training_mode"]]:
        raise ValueError("Plan 3 metadata observation-noise formula does not match its mode")
    scheduler = saved.get("semantic_scheduler")
    try:
        scheduler_valid = (
            isinstance(scheduler, dict)
            and scheduler.get("family") == _SCHEDULER[0]
            and float(scheduler.get("a", "nan")) == _SCHEDULER[1]
            and float(scheduler.get("b", "nan")) == _SCHEDULER[2]
            and scheduler.get("conditioning") == "t"
        )
    except (TypeError, ValueError):
        scheduler_valid = False
    if not scheduler_valid:
        raise ValueError("Plan 3 metadata semantic scheduler contract mismatch")
    canonical = saved.get("canonical_schedule")
    try:
        canonical_valid = (
            isinstance(canonical, dict)
            and canonical.get("schema_version") == "plan3-canonical-schedule-v1"
            and canonical.get("index") == "t"
            and int(canonical.get("rows", -1)) == 1000
            and bool(canonical.get("csv_sha256"))
            and bool(canonical.get("json_sha256"))
        )
    except (TypeError, ValueError):
        canonical_valid = False
    if not canonical_valid:
        raise ValueError("Plan 3 metadata canonical schedule contract mismatch")
    solver = saved.get("solver")
    if not isinstance(solver, dict) or not isinstance(solver.get("source_sha256"), str) or len(solver["source_sha256"]) != 64 or any(solver.get(key) != value for key, value in _SOLVER_CONTRACT.items()):
        raise ValueError("Plan 3 metadata solver contract mismatch")
    posterior = saved.get("posterior")
    if not isinstance(posterior, dict) or posterior.get("mode") not in {"full", "topk"}:
        raise ValueError("Plan 3 metadata posterior mode is invalid")
    if posterior["mode"] == "full" and posterior.get("k") is not None:
        raise ValueError("full posterior must not declare k")
    if posterior["mode"] == "topk" and (not isinstance(posterior.get("k"), int) or posterior["k"] <= 0):
        raise ValueError("topk posterior requires a positive integer k")
    keys = ("geometry", "semantic_scheduler", "canonical_schedule", "observation_noise", "base_model_lock", "solver", "posterior")
    if not inference_only: keys += ("comparison", "training_protocol_sha256", "data_manifest_sha256")
    for key in keys:
        if saved.get(key) != expected.get(key): raise ValueError(f"Plan 3 metadata mismatch for {key}")

def load_checkpoint(path, *, map_location="cpu"):
    import torch
    value = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(value, dict) or value.get("metadata", {}).get("namespace") != NAMESPACE:
        raise ValueError("checkpoint is not a Plan 3 checkpoint")
    return value
