#!/usr/bin/env python3
"""Validate and publish one permanent-cache index from distributed shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge(input_path: Path, features_dir: Path, num_shards: int) -> dict:
    if num_shards < 2:
        raise ValueError("Distributed merge requires at least two shards")
    records = json.loads(input_path.read_text(encoding="utf-8"))
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Input JSON contains duplicate sample names")

    rows = []
    stats = []
    for shard in range(num_shards):
        suffix = f".shard_{shard:02d}_of_{num_shards:02d}"
        index_path = features_dir / f"index{suffix}.json"
        stats_path = features_dir / f"extraction_stats{suffix}.json"
        shard_rows = json.loads(index_path.read_text(encoding="utf-8"))
        shard_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        expected_indices = list(range(shard, len(records), num_shards))
        actual_indices = sorted(int(row["input_index"]) for row in shard_rows)
        if actual_indices != expected_indices:
            raise ValueError(f"Shard {shard} has missing or duplicate input indices")
        if (shard_stats.get("samples_total") != len(shard_rows)
                or shard_stats.get("num_shards") != num_shards
                or shard_stats.get("shard_index") != shard
                or shard_stats.get("cycles") != 4
                or shard_stats.get("use_msa") is not False):
            raise ValueError(f"Shard {shard} has incompatible extraction stats")
        for row in shard_rows:
            index = int(row["input_index"])
            sample_id = names[index]
            if row["sample_id"] != sample_id or row["file"] != sample_id + ".pt":
                raise ValueError(f"Shard {shard} metadata mismatch at input {index}")
            # Legacy files predate per-record kernel provenance.
            row.setdefault("kernel_backend", "torch/torch")
            if row["kernel_backend"] not in {"torch/torch", "cuequivariance/cuequivariance"}:
                raise ValueError(f"Unsupported kernel backend for {sample_id}")
        rows.extend(shard_rows)
        stats.append(shard_stats)

    rows.sort(key=lambda row: int(row["input_index"]))
    if len(rows) != len(records):
        raise ValueError("Merged index does not cover all input records")
    cached_files = {entry.name for entry in os.scandir(features_dir) if entry.name.endswith(".pt")}
    missing = [row["file"] for row in rows if row["file"] not in cached_files]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature files; first: {missing[0]}")

    total_bytes = sum(int(row["file_bytes"]) for row in rows)
    generated = sum(int(item["samples_generated"]) for item in stats)
    reused = sum(int(item["samples_reused"]) for item in stats)
    result = {
        "samples_total": len(rows),
        "samples_generated": generated,
        "samples_reused": reused,
        "mean_n_token": sum(int(row["n_token"]) for row in rows) / len(rows),
        "total_cache_bytes": total_bytes,
        "mean_cache_bytes": total_bytes / len(rows),
        "generated_seconds": sum(float(item["generated_seconds"]) for item in stats),
        "gpu_peak_memory_bytes": max(int(item.get("gpu_peak_memory_bytes") or 0) for item in stats),
        "use_msa": False,
        "cycles": 4,
        "num_shards": num_shards,
        "parallel_workers": num_shards,
        "distributed": True,
        "kernel_backend_counts": dict(Counter(row["kernel_backend"] for row in rows)),
    }
    atomic_json(features_dir / "index.json", rows)
    atomic_json(features_dir / "extraction_stats.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input_json, args.features_dir, args.num_shards), indent=2))
