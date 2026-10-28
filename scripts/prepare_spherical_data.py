#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Prepare immutable, variable-length spherical training data from a selection plan.

Workers write token shards independently. Finalization deduplicates normalized
questions, unions related conversation questions before splitting, and publishes
the manifest only after every shard completes. No source file is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import re
import time
import unicodedata

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
SCHEMA = 1
RECORD = np.dtype([("offset", "<u8"), ("context", "<u4"), ("target", "<u4"),
                   ("question", "V32"), ("group", "V32"), ("row", "<u8"), ("turn", "<u4")])
REF = np.dtype([("shard", "<u4"), ("record", "<u8"), ("context", "<u4"), ("target", "<u4")])


def canonical(text: str, *, casefold: bool = True) -> str:
    value = unicodedata.normalize("NFKC", text)
    if casefold:
        value = value.casefold()
    # Preserve punctuation/operators: 2+3 and 2-3 are different math problems.
    return " ".join(re.findall(r"\w+|[^\w\s]", value))


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def token_ids(messages: list[dict], *, generation_prompt: bool, tools: list) -> list[int]:
    controls = {"enable_thinking": False, "truncate_history_thinking": True}
    template = TOKENIZER.get_chat_template()
    if "preserve_thinking" in template and "truncate_history_thinking" not in template:
        controls.pop("truncate_history_thinking")
        controls["preserve_thinking"] = False
    value = TOKENIZER.apply_chat_template(messages, add_generation_prompt=generation_prompt,
        tokenize=True, tools=tools, **controls)
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def json_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def build_benchmark_index(root: Path) -> dict:
    """Exclude verbatim normalized evaluation questions, including wrapped copies.

    This is exact text decontamination, not a claim to detect paraphrases.
    All local test/validation/dev/prompt sets and GPQA CSV files are included.
    """
    texts = set()
    files = []
    for path in sorted(root.rglob("*")):
        if path.suffix == ".parquet" and any(part in path.name for part in ("test", "validation", "dev", "prompt")):
            pf = pq.ParquetFile(path)
            fields = [key for key in ("question", "problem", "input", "prompt", "text") if key in pf.schema_arrow.names]
            if not fields:
                raise ValueError(f"No known evaluation question field: {path}")
            rows = (row for batch in pf.iter_batches(batch_size=1024, columns=fields) for row in batch.to_pylist())
        elif path.suffix == ".csv" and "gpqa" in str(path):
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            fields = ["Question"]
        else:
            continue
        count = 0
        for row in rows:
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and canonical(value):
                    texts.add(canonical(value))
                    count += 1
        files.append({"path": str(path), "sha256": file_hash(path), "question_fields": fields, "count": count})
    if not texts:
        raise ValueError("No benchmark questions found")
    return {"files": files, "questions": sorted(texts), "method": "NFKC/casefold/token-spacing exact substring, preserving all punctuation/operators; no semantic paraphrase detection"}


def worker_init(tokenizer_path: str, benchmark_path: str) -> None:
    global TOKENIZER, BENCH, BENCH_SHORT
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    TOKENIZER.model_max_length = 10**9
    BENCH, BENCH_SHORT = defaultdict(list), set()
    for question in json.loads(Path(benchmark_path).read_text())["questions"]:
        words = question.split()
        if len(words) >= 4:
            BENCH[" ".join(words[:4])].append(question)
        else:
            BENCH_SHORT.add(question)


def benchmark_overlap(text: str) -> bool:
    value = canonical(text)
    if value in BENCH_SHORT:
        return True
    words = value.split()
    padded = " " + value + " "
    return any(" " + question + " " in padded
               for i in range(max(0, len(words) - 3))
               for question in BENCH.get(" ".join(words[i:i + 4]), ()))


def source_rows(task: dict):
    path = Path(task["path"])
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        columns = [k for k in ("messages", "tools", "subset") if k in pf.schema_arrow.names]
        base = sum(pf.metadata.row_group(i).num_rows for i in range(task["row_group"]))
        for batch in pf.iter_batches(row_groups=[task["row_group"]], batch_size=128, columns=columns):
            for row in batch.to_pylist():
                yield base, row
                base += 1
    else:
        with path.open("rb") as stream:
            start = task["start"]
            if start:
                stream.seek(start - 1)
                if stream.read(1) != b"\n":
                    stream.readline()
            while stream.tell() < task["end"]:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    yield offset, json.loads(line)


def process_task(task: dict) -> dict:
    directory = Path(task["output"])
    name = f"shard_{task['id']:05d}"
    result_path = directory / (name + ".json")
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["task"] != task:
            raise ValueError(f"Changed preprocessing task: {name}")
        for kind in ("tokens", "records"):
            if file_hash(directory / saved[kind]) != saved[kind + "_sha256"]:
                raise ValueError(f"Corrupt prepared shard: {name}/{kind}")
        return saved
    counts = Counter()
    lengths = Counter()
    offset = 0
    token_path, record_path = directory / (name + ".tokens"), directory / (name + ".records")
    started = time.time()
    with token_path.open("wb") as token_file, record_path.open("wb") as record_file:
        for row_id, row in source_rows(task):
            counts["rows"] += 1
            if any(row.get(key) != value for key, value in task.get("row_filter", {}).items()):
                counts["row_filter"] += 1
                continue
            raw = row.get("messages")
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"Invalid messages at {task['path']}:{row_id}")
            messages = [{k: v for k, v in m.items() if v is not None} for m in raw]
            if any(m.get("role") in ("tool", "function") or m.get("tool_calls") or m.get("function_call") or m.get("tool_call_id") for m in messages):
                counts["tool_conversation"] += 1
                continue
            tools = row.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools) if tools.strip() else []
            if any(not isinstance(m.get("content", ""), str) for m in messages):
                raise ValueError(f"Non-text content at {task['path']}:{row_id}")
            turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
            if task["assistant_targets"] == "last":
                turns = turns[-1:]
            users = [m.get("content", "") for m in messages if m.get("role") == "user"]
            if not users:
                counts["missing_question"] += 1
                continue
            if any(benchmark_overlap(text) for text in users):
                counts["benchmark_conversation"] += 1
                continue
            group_text = users[-1] if task["assistant_targets"] == "last" else users[0]
            group = digest(canonical(group_text, casefold=False))
            for turn in turns:
                counts["assistant_targets"] += 1
                prior_users = [m.get("content", "") for m in messages[:turn] if m.get("role") == "user"]
                if not prior_users or not canonical(prior_users[-1]):
                    counts["missing_question"] += 1
                    continue
                prefix = token_ids(messages[:turn], generation_prompt=True, tools=tools or [])
                full = token_ids(messages[:turn + 1], generation_prompt=False, tools=tools or [])
                common = next((i for i, (a, b) in enumerate(zip(prefix, full)) if a != b), min(len(prefix), len(full)))
                if common < max(0, len(prefix) - 16):
                    raise ValueError(f"Template prefix mismatch at {task['path']}:{row_id}:{turn}")
                context, target = common, len(full) - common
                lengths[f"C{((context + 511)//512)*512}_T{((target + 511)//512)*512}"] += 1
                if not context or not target:
                    counts["empty_tokens"] += 1
                    continue
                if context > task["context_max"] or target > task["target_max"]:
                    counts["length_filtered"] += 1
                    continue
                np.asarray(full, dtype="<u4").tofile(token_file)
                question = digest(canonical(prior_users[-1], casefold=False))
                np.array([(offset, context, target, question, group, row_id, turn)], dtype=RECORD).tofile(record_file)
                offset += len(full)
                counts["accepted"] += 1
                counts["target_tokens"] += target
                counts["context_tokens"] += context
    result = {"task": task, "counts": dict(counts), "length_buckets_before_filter": dict(lengths),
              "tokens": token_path.name, "records": record_path.name,
              "tokens_sha256": file_hash(token_path), "records_sha256": file_hash(record_path),
              "seconds": time.time() - started}
    json_write(result_path, result)
    return result


def finalize(output: Path, preparation: dict, results: list[dict], seed: int) -> None:
    """Deduplicate globally and prevent connected questions crossing splits."""
    validation_fraction = preparation["validation_fraction"]
    parent: dict[bytes, bytes] = {}
    def root(key):
        trail = []
        while key in parent and parent[key] != key:
            trail.append(key)
            key = parent[key]
        for item in trail:
            parent[item] = key
        return key
    for result in results:
        n = result["counts"].get("accepted", 0)
        if not n:
            continue
        records = np.memmap(output / result["records"], dtype=RECORD, mode="r", shape=(n,))
        for record in records:
            a, b = root(bytes(record["question"])), root(bytes(record["group"]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    LOG.info("Question groups linked; selecting unique targets")
    seen = set()
    source_refs, stats = defaultdict(lambda: {"train": [], "validation": []}), defaultdict(Counter)
    for result in results:
        task = result["task"]
        source = task["source"]
        stats[source].update({"raw_" + k: v for k, v in result["counts"].items()})
        n = result["counts"].get("accepted", 0)
        if not n:
            continue
        records = np.memmap(output / result["records"], dtype=RECORD, mode="r", shape=(n,))
        for i, record in enumerate(records):
            question = bytes(record["question"])
            if question in seen:
                stats[source]["duplicates"] += 1
                continue
            seen.add(question)
            group = root(question)
            split = "validation" if int.from_bytes(digest(str(seed) + group.hex())[:8], "little") / 2**64 < validation_fraction else "train"
            source_refs[source][split].append((task["id"], i, int(record["context"]), int(record["target"])))
            stats[source][split + "_samples"] += 1
            stats[source][split + "_target_tokens"] += int(record["target"])
            stats[source][split + "_context_tokens"] += int(record["context"])
    sources = []
    domains = {task["source"]: task["domain"] for task in preparation["tasks"]}
    for source in sorted(domains):
        if not stats[source]["train_samples"]:
            raise ValueError(f"Source empty after filtering/deduplication: {source}")
        paths = {}
        for split in ("train", "validation"):
            path = output / f"{source}.{split}.npy"
            np.save(path, np.array(source_refs[source][split], dtype=REF))
            paths[split] = {"path": path.name, "sha256": file_hash(path)}
        sources.append({"id": source, "domain": domains[source], "counts": dict(stats[source]), **paths})
    manifest = {"schema_version": SCHEMA, "preparation_id": preparation["id"],
                "selection": preparation["selection"], "tokenizer": preparation["tokenizer"],
                "seed": seed, "validation_fraction": validation_fraction,
                "deduplication": "one target per NFKC/token-spacing-normalized latest user question, preserving case/punctuation/operators; stable source/file/row order",
                "validation_grouping": "connected components of first and target user questions; last-only datasets use final question to avoid shared few-shot examples",
                "benchmark_index_sha256": file_hash(output / "benchmarks.json"),
                "sources": sources,
                "shards": [{k: r[k] for k in ("task", "tokens", "records", "tokens_sha256", "records_sha256", "counts")} for r in results]}
    manifest["id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    json_write(output / "manifest.json", manifest)
    LOG.info("Preparation complete: %s", {s["id"]: s["counts"]["train_samples"] for s in sources})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, help="Optional local benchmark directory for exact-text decontamination")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    args = parser.parse_args()
    if args.workers <= 0 or not 0 <= args.validation_fraction < 1:
        parser.error("workers must be positive; validation-fraction must be in [0, 1)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    selection = json.loads(args.selection.read_text())
    limits = selection["filters"]
    if limits["context_max_tokens"] <= 0 or limits["target_max_tokens"] <= 0 or limits["target_max_tokens"] % 8:
        parser.error("context limit must be positive; target limit must be a positive multiple of training block size 8")
    domains = {rule["domain"] for rule in selection["sources"]}
    weights = selection["sample_mixture"]
    if set(weights) != domains or any(w <= 0 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-8:
        parser.error("sample_mixture must give each source domain a positive weight and sum to 1")
    if any(rule["assistant_targets"] not in {"all", "last"} for rule in selection["sources"]):
        parser.error("assistant_targets must be all or last")
    if args.benchmarks is not None and not args.benchmarks.is_dir():
        parser.error("benchmarks must be an existing directory")
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark_path = args.output / "benchmarks.json"
    benchmark_index = build_benchmark_index(args.benchmarks) if args.benchmarks else {"questions": [], "files": []}
    if benchmark_path.exists() and json.loads(benchmark_path.read_text()) != benchmark_index:
        raise ValueError("Benchmark exclusion set changed; choose a new output directory")
    if not benchmark_path.exists():
        json_write(benchmark_path, benchmark_index)
    tasks = []
    for rule_index, rule in enumerate(selection["sources"]):
        paths = sorted(Path(selection["data_root"]).glob(rule["glob"]))
        if not paths:
            raise ValueError(f"No files for {rule}")
        for path in paths:
            if not path.is_file() or path.suffix not in {".jsonl", ".parquet"}:
                raise ValueError(f"Only JSONL/parquet input files are supported: {path}")
            stratum = path.stem if path.suffix == ".jsonl" else "parquet"
            source = f"{rule['domain']}_{rule_index}_{stratum}"
            base = dict(path=str(path), source=source, domain=rule["domain"],
                        assistant_targets=rule["assistant_targets"], row_filter=rule.get("row_filter", {}),
                        size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                        context_max=selection["filters"]["context_max_tokens"],
                        target_max=selection["filters"]["target_max_tokens"], output=str(args.output.resolve()))
            if path.suffix == ".parquet":
                pieces = [{"row_group": i} for i in range(pq.ParquetFile(path).num_row_groups)]
            else:
                pieces = [{"start": i, "end": min(i + (128 << 20), base["size"])} for i in range(0, base["size"], 128 << 20)]
            for piece in pieces:
                tasks.append({**base, **piece, "id": len(tasks)})
    tokenizer = {p.name: file_hash(p) for p in sorted(Path(selection["tokenizer_path"]).iterdir())
                 if p.is_file() and (p.name.startswith("tokenizer") or p.suffix == ".py" or p.name in ("config.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"))}
    if not tokenizer:
        raise ValueError("No tokenizer files found in tokenizer_path")
    tokenizer_root = Path(selection["tokenizer_path"])
    for path in sorted((tokenizer_root / "chat_templates").glob("*.jinja")):
        tokenizer[str(path.relative_to(tokenizer_root))] = file_hash(path)
    preparation = {"schema_version": SCHEMA, "selection": selection, "tokenizer": tokenizer, "tasks": tasks,
                   "benchmark_sha256": file_hash(benchmark_path), "seed": args.seed,
                   "validation_fraction": args.validation_fraction,
                   "preprocessor_sha256": file_hash(Path(__file__))}
    preparation["id"] = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    path = args.output / "preparation.json"
    if path.exists() and json.loads(path.read_text()) != preparation:
        raise ValueError("Preparation config/source/code changed; choose a new output directory")
    json_write(path, preparation)
    results = []
    LOG.info("Starting %d shards with %d workers", len(tasks), args.workers)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(selection["tokenizer_path"], str(benchmark_path))) as pool:
        futures = [pool.submit(process_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            LOG.info("Completed %d/%d: %s %s", len(results), len(tasks), result["task"]["source"], result["counts"])
            json_write(args.output / "progress.json", {"completed": len(results), "total": len(tasks),
                       "counts": dict(sum((Counter(r["counts"]) for r in results), Counter()))})
    finalize(args.output, preparation, sorted(results, key=lambda r: r["task"]["id"]), args.seed)


if __name__ == "__main__":
    main()
