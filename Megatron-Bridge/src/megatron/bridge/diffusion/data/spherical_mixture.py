# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Deterministic sample-weighted mixture with source-local reshuffling and resume.

The checkpoint's consumed_train_samples is the only mutable cursor. Every source
epoch, permutation, and mixture window is a pure function of the immutable plan,
seed, and logical sample index; DataLoader prefetch cannot advance saved state.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

RECORD = np.dtype([("offset", "<u8"), ("context", "<u4"), ("target", "<u4"),
                   ("question", "V32"), ("group", "V32"), ("row", "<u8"), ("turn", "<u4")])


TAIL_BLOCK_PROTOCOL = "target_relative_eos_fill_v1"


def resolve_eos_id(tokenizer, *, configured_id: int | None = None) -> int:
    """Resolve a real EOS ID, checking tokenizer/config agreement and vocabulary."""
    outer = tokenizer
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    candidates = [configured_id]
    for obj in (tokenizer, outer):
        for name in ("eos_token_id", "eod"):
            value = getattr(obj, name, None)
            if value is not None:
                candidates.append(value)
    ids = []
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("Tail EOS fill requires one unambiguous EOS token ID")
            value = value[0]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("EOS token ID must be a nonnegative integer")
        ids.append(int(value))
    if not ids or len(set(ids)) != 1:
        raise ValueError("Missing EOS token ID or tokenizer/config EOS mismatch")
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and ids[0] >= vocab_size:
        raise ValueError("EOS token ID is outside tokenizer vocabulary")
    return ids[0]


def fill_target(target: torch.Tensor, *, width: int, block_size: int, eos_id: int, pad_id: int) -> dict:
    """Keep raw tokens, supervise EOS to this sample's block end, then outer-pad."""
    original = len(target)
    if block_size <= 0 or width % block_size or original <= 0:
        raise ValueError("Nonempty targets and block-aligned positive widths are required")
    diffusion = ((original + block_size - 1) // block_size) * block_size
    if diffusion > width:
        raise ValueError("Target exceeds configured diffusion width")
    ids = torch.full((width,), pad_id, dtype=torch.long)
    ids[:original] = target
    ids[original:diffusion] = eos_id
    offsets = torch.arange(width)
    return {
        "clean_ids": ids,
        "loss_mask": (offsets < diffusion).float(),
        "original_target_lengths": torch.tensor(original),
        "diffusion_target_lengths": torch.tensor(diffusion),
        # Compatibility alias: ALL downstream layout consumers use diffusion length.
        "target_lengths": torch.tensor(diffusion),
        "eos_token_id": torch.tensor(eos_id),
    }


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def load_manifest(path: str) -> dict:
    """Read a completely published manifest and verify its content identity."""
    manifest = json.loads(Path(path).read_text())
    identity = manifest.pop("id")
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["id"] = identity
    if actual != identity or manifest["schema_version"] != 1:
        raise ValueError("Prepared dataset manifest identity/version mismatch")
    if any(s["counts"]["train_samples"] <= 0 for s in manifest["sources"]):
        raise ValueError("All participating sources must have nonempty training data")
    return manifest


def mixture_plan(config) -> dict:
    """Allocate exact category quotas and explicit within-category source weights."""
    manifest = load_manifest(config.prepared_manifest)
    sources = manifest["sources"]
    cycle = config.mixture_cycle_size
    weights = manifest["selection"]["sample_mixture"]
    quotas = [0] * len(sources)
    for domain, weight in weights.items():
        ids = [i for i, source in enumerate(sources) if source["domain"] == domain]
        if not ids:
            raise ValueError(f"Empty mixture domain: {domain}")
        total = cycle * weight
        if abs(total - round(total)) > 1e-8:
            raise ValueError("Mixture cycle must exactly represent category weights")
        total = round(total)
        mass = np.array([sources[i]["counts"]["train_samples"] for i in ids], dtype=np.float64)
        if domain == "chat":
            on = [j for j, i in enumerate(ids) if sources[i]["id"].endswith("reasoning_on")]
            off = [j for j, i in enumerate(ids) if sources[i]["id"].endswith("reasoning_off")]
            if len(on) != 1 or len(off) != 1:
                raise ValueError("Expected exactly one reasoning_on/off chat source")
            half = (mass[on[0]] + mass[off[0]]) / 2
            mass[on[0]] = mass[off[0]] = half
        desired = total * mass / mass.sum()
        counts = np.floor(desired).astype(int)
        order = np.argsort(-(desired - counts), kind="stable")
        for j in order[:total - counts.sum()]:
            counts[j] += 1
        if np.any(counts <= 0):
            raise ValueError("Mixture cycle too small to represent every source; increase cycle size")
        for j, i in enumerate(ids):
            quotas[i] = int(counts[j])
    if sum(quotas) != cycle:
        raise ValueError("Mixture quotas do not sum to cycle size")
    plan = {"version": 1, "dataset_id": manifest["id"], "seed": config.mixture_seed,
            "cycle_size": cycle, "source_ids": [s["id"] for s in sources], "quotas": quotas,
            "source_sizes": [s["counts"]["train_samples"] for s in sources],
            "length_bucket_group": config.length_bucket_group,
            "within_category": "proportional to filtered unique training samples; Chat-v2 on/off equal mass",
            "source_epochs": "PCG64 permutation seeded by base seed, source ID, epoch",
            "resume": "derive all source epochs/permutations from plan_id and consumed_train_samples",
            "dynamic_padding": config.dynamic_padding}
    plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    ordering = getattr(config, "length_ordering", "legacy")
    if ordering != "legacy":
        if ordering != "balanced_v2":
            raise ValueError(f"Unknown length ordering: {ordering}")
        start, width = config.length_ordering_start, config.length_ordering_window
        micro, global_batch = config.length_ordering_microbatch, config.length_ordering_global_batch
        if (start < 0 or min(width, micro, global_batch) <= 0 or global_batch % micro
                or width % global_batch or start % global_batch or width % cycle):
            raise ValueError("Balanced ordering requires complete, aligned batches and mixture windows")
        parent = plan.pop("id")
        plan["version"] = 2
        plan["ordering"] = {"name": ordering, "parent_plan_id": parent, "start_sample": start,
                            "window": width, "microbatch": micro, "global_batch": global_batch,
                            "context_bin": 128, "target_alignment": config.block_size,
                            "policy": "cluster context/target within microbatches; balance physical lengths across ranks; shuffle whole global batches"}
        plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def validate_ordering_transition(saved: dict, expected: dict, plan: dict) -> None:
    """Accept only a recorded legacy-to-balanced migration at its exact cursor."""
    ordering = plan.get("ordering", {})
    if (saved.get("plan_id") != ordering.get("parent_plan_id")
            or saved.get("consumed_train_samples") != ordering.get("start_sample")):
        raise ValueError("Prepared mixture plan changed; cannot reuse the checkpoint data cursor")
    translated = dict(saved, plan_id=plan["id"])
    if translated != expected:
        raise ValueError("Ordering migration changed consumed samples, source epochs, or dataset identity")


class PreparedMixtureDataset(Dataset):
    """Random-access infinite logical mixture bounded only by the training budget."""

    def __init__(self, config, tokenizer, total_samples: int):
        self.config = config
        self.manifest = load_manifest(config.prepared_manifest)
        self.plan = mixture_plan(config)
        self.root = Path(config.prepared_manifest).parent
        self.total_samples = int(total_samples)
        if self.total_samples <= 0:
            raise ValueError("Training sample budget must be positive")
        self.refs = [np.load(self.root / s["train"]["path"], mmap_mode="r") for s in self.manifest["sources"]]
        for refs, expected in zip(self.refs, self.plan["source_sizes"], strict=True):
            if len(refs) != expected:
                raise ValueError("Prepared source index size mismatch")
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "pad", None)
            if callable(self.pad_id):  # HF exposes pad() as an operation, not a token ID.
                self.pad_id = None
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eod", None)
        if self.pad_id is None:
            raise ValueError("Tokenizer must expose a pad or EOS token ID")
        self.pad_id = int(self.pad_id)
        self.eos_id = resolve_eos_id(tokenizer, configured_id=getattr(config, "eos_token_id", None))
        self._windows = OrderedDict()
        self._permutations = OrderedDict()
        self._shards = OrderedDict()
        self._balanced_windows = OrderedDict()

    def __len__(self) -> int:
        return self.total_samples

    def _permutation(self, source: int, epoch: int) -> np.ndarray:
        key = source, epoch
        if key not in self._permutations:
            rng = np.random.default_rng(stable_seed(self.plan["seed"], self.plan["source_ids"][source], epoch))
            self._permutations[key] = rng.permutation(self.plan["source_sizes"][source])
            if len(self._permutations) > 2 * len(self.refs):
                self._permutations.popitem(last=False)
        self._permutations.move_to_end(key)
        return self._permutations[key]

    def window(self, cycle: int) -> np.ndarray:
        """Return [source, source-index, source-ordinal] for one balanced window."""
        if cycle in self._windows:
            self._windows.move_to_end(cycle)
            return self._windows[cycle]
        rows = []
        lengths = []
        for source, quota in enumerate(self.plan["quotas"]):
            size = self.plan["source_sizes"][source]
            for ordinal in range(cycle * quota, (cycle + 1) * quota):
                epoch, index = divmod(ordinal, size)
                selected = int(self._permutation(source, epoch)[index])
                ref = self.refs[source][selected]
                rows.append((source, selected, ordinal))
                lengths.append(((int(ref["target"]) + 255) // 256, (int(ref["context"]) + 255) // 256))
        rng = np.random.default_rng(stable_seed(self.plan["seed"], "mixture", cycle))
        order = rng.permutation(len(rows)).tolist()
        order.sort(key=lambda i: lengths[i])
        width = self.plan["length_bucket_group"]
        groups = [order[i:i + width] for i in range(0, len(order), width)]
        rng.shuffle(groups)
        order = [i for group in groups for i in group]
        result = np.array(rows, dtype=np.int64)[order]
        # Bucketing may reorder a source inside an epoch, but must not deliver
        # the next epoch before exhausting the previous one. Preserve source
        # slots (and hence quotas) while repairing only epoch-boundary crossings.
        for source, size in enumerate(self.plan["source_sizes"]):
            positions = np.flatnonzero(result[:, 0] == source)
            selected = result[positions]
            result[positions] = selected[np.argsort(selected[:, 2] // size, kind="stable")]
        self._windows[cycle] = result
        if len(self._windows) > 3:
            self._windows.popitem(last=False)
        return result

    def _balanced_window(self, block: int) -> np.ndarray:
        """Reorder the same unseen sample set, preserving source epoch ordering."""
        if block in self._balanced_windows:
            self._balanced_windows.move_to_end(block)
            return self._balanced_windows[block]
        cfg = self.plan["ordering"]
        start = cfg["start_sample"] + block * cfg["window"]
        cycle = self.plan["cycle_size"]
        parts = []
        end = start + cfg["window"]
        for i in range(start // cycle, (end - 1) // cycle + 1):
            parts.append(self.window(i)[max(0, start - i * cycle):min(cycle, end - i * cycle)])
        rows = np.concatenate(parts)
        contexts = np.array([int(self.refs[s][r]["context"]) for s, r, _ in rows])
        targets = np.array([int(self.refs[s][r]["target"]) for s, r, _ in rows])
        rng = np.random.default_rng(stable_seed(self.plan["seed"], "balanced_v2", start))
        order = rng.permutation(len(rows)).tolist()
        order.sort(key=lambda i: (contexts[i] // cfg["context_bin"], targets[i], contexts[i]))
        micro = cfg["microbatch"]
        groups = np.array(order).reshape(-1, micro)
        alignment = cfg["target_alignment"]
        physical = contexts[groups].max(axis=1) + 2 * ((targets[groups].max(axis=1) + alignment - 1) // alignment) * alignment
        groups = groups[np.argsort(physical, kind="stable")]
        batches = groups.reshape(-1, cfg["global_batch"] // micro, micro)
        rng.shuffle(batches, axis=0)
        for batch in batches:
            rng.shuffle(batch, axis=0)
        result = rows[batches.reshape(-1)]
        # Source exhaustion must still precede the next epoch, including when
        # the migration starts midway through a legacy mixture window.
        for source, size in enumerate(self.plan["source_sizes"]):
            positions = np.flatnonzero(result[:, 0] == source)
            selected = result[positions]
            result[positions] = selected[np.argsort(selected[:, 2] // size, kind="stable")]
        self._balanced_windows[block] = result
        if len(self._balanced_windows) > 2:
            self._balanced_windows.popitem(last=False)
        return result

    def sample_reference(self, index: int) -> np.ndarray:
        """Return provenance without reading tokens, for loading and cost audits."""
        ordering = self.plan.get("ordering")
        if ordering and index >= ordering["start_sample"]:
            block, within = divmod(index - ordering["start_sample"], ordering["window"])
            return self._balanced_window(block)[within]
        cycle, within = divmod(index, self.plan["cycle_size"])
        return self.window(cycle)[within]

    def cursor_state(self, consumed: int) -> dict:
        """Audit the actual consumed prefix after a recorded ordering migration."""
        ordering = self.plan.get("ordering")
        if not ordering or consumed <= ordering["start_sample"]:
            return self._legacy_cursor_state(consumed)
        block, within = divmod(consumed - ordering["start_sample"], ordering["window"])
        boundary = consumed - within
        state = self._legacy_cursor_state(boundary)
        if not within:
            return state
        partial = self._balanced_window(block)[:within]
        state.update(consumed_train_samples=consumed,
                     cycle=consumed // self.plan["cycle_size"], within_cycle=consumed % self.plan["cycle_size"])
        for source, source_state in enumerate(state["sources"]):
            visited = partial[partial[:, 0] == source, 2]
            size = self.plan["source_sizes"][source]
            count = source_state["samples"] + len(visited)
            source_state.update(samples=count, completed_equivalent_epochs=count // size,
                                samples_beyond_first_pass=max(0, count - size),
                                repeated_samples=source_state["repeated_samples"] + int(np.count_nonzero(visited >= size)),
                                latest_visited_epoch=max(source_state["latest_visited_epoch"], int(visited.max()) // size if len(visited) else -1))
            source_state["ordinal_range_in_window"] = [
                boundary // self.plan["cycle_size"] * self.plan["quotas"][source],
                ((boundary + ordering["window"] + self.plan["cycle_size"] - 1) // self.plan["cycle_size"]) * self.plan["quotas"][source]]
        return state

    def _legacy_cursor_state(self, consumed: int) -> dict:
        """Serializable audit state; deterministic RNG state is encoded by seeds/epochs."""
        if consumed < 0:
            raise ValueError("Negative mixture cursor")
        cycle, within = divmod(consumed, self.plan["cycle_size"])
        partial = self.window(cycle)[:within] if within else np.empty((0, 3), dtype=np.int64)
        source_states = []
        for source, quota in enumerate(self.plan["quotas"]):
            count = cycle * quota + int(np.count_nonzero(partial[:, 0] == source))
            size = self.plan["source_sizes"][source]
            visited = partial[partial[:, 0] == source, 2]
            repeated = max(0, cycle * quota - size) + int(np.count_nonzero(visited >= size))
            # Length sorting can visit positions out of order inside one epoch.
            source_states.append({"source": self.plan["source_ids"][source], "samples": count,
                                  "completed_equivalent_epochs": count // size,
                                  "samples_beyond_first_pass": max(0, count - size),
                                  "repeated_samples": repeated,
                                  "latest_visited_epoch": max((cycle * quota - 1) // size, int(visited.max()) // size if len(visited) else -1),
                                  "ordinal_range_in_window": [cycle * quota, (cycle + 1) * quota]})
        return {"plan_id": self.plan["id"], "dataset_id": self.manifest["id"], "seed": self.plan["seed"],
                "consumed_train_samples": consumed, "cycle": cycle, "within_cycle": within,
                "sources": source_states, "rng_state": "stateless; restored from seed/source/epoch and logical cursor"}

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.total_samples:
            raise IndexError(index)
        source, selected, ordinal = map(int, self.sample_reference(index))
        ref = self.refs[source][selected]
        shard = int(ref["shard"])
        if shard not in self._shards:
            info = self.manifest["shards"][shard]
            self._shards[shard] = (np.memmap(self.root / info["records"], dtype=RECORD, mode="r"),
                                   np.memmap(self.root / info["tokens"], dtype="<u4", mode="r"))
            if len(self._shards) > 64:
                self._shards.popitem(last=False)
        self._shards.move_to_end(shard)
        records, tokens = self._shards[shard]
        record = records[int(ref["record"])]
        context, target, offset = int(record["context"]), int(record["target"]), int(record["offset"])
        if context != int(ref["context"]) or target != int(ref["target"]):
            raise ValueError("Prepared record/index length mismatch")
        ids = np.array(tokens[offset:offset + context + target], dtype=np.int64)
        if len(ids) != context + target:
            raise ValueError("Truncated prepared token shard")
        return {"context": torch.from_numpy(ids[:context]), "target": torch.from_numpy(ids[context:]),
                "sample_id": index, "source_id": source, "source_epoch": ordinal // self.plan["source_sizes"][source],
                "row_id": int(record["row"]), "turn_id": int(record["turn"])}

    def collate_fn(self, samples: list[dict]) -> dict:
        """EOS-fill each target block, then pad the batch; preserve all original tokens."""
        block = self.config.block_size
        context = max(len(s["context"]) for s in samples)
        target = ((max(len(s["target"]) for s in samples) + block - 1) // block) * block
        if not self.config.dynamic_padding:
            context = self.config.seq_length - 2 * self.config.target_max_tokens
            target = self.config.target_max_tokens
        if target > self.config.target_max_tokens or context > self.config.seq_length - 2 * self.config.target_max_tokens:
            raise ValueError("Prepared sample exceeds configured complete-context/target limits")
        b = len(samples)
        filled = [fill_target(s["target"], width=target, block_size=block,
                              eos_id=self.eos_id, pad_id=self.pad_id) for s in samples]
        result = {key: torch.stack([sample[key] for sample in filled]) for key in filled[0]}
        result["context_ids"] = torch.full((b, context), self.pad_id, dtype=torch.long)
        result["context_lengths"] = torch.tensor([len(s["context"]) for s in samples])
        for i, sample in enumerate(samples):
            result["context_ids"][i, :len(sample["context"])] = sample["context"]
        for key in ("sample_id", "source_id", "source_epoch", "row_id", "turn_id"):
            result[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
        result["conversation_id"] = [str(s["sample_id"]) for s in samples]
        result["source_shard"] = [self.plan["source_ids"][s["source_id"]] for s in samples]
        return result
