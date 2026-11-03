#!/usr/bin/env python3
"""Publish the distributed permanent cache, then start epoch-1 negative caching."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.num_shards < 2 or args.poll_seconds < 1:
        parser.error("Expected at least two shards and a positive poll interval")

    root = Path(__file__).resolve().parents[2]
    merge_path = args.job_dir / "bin/merge_shards.py"
    spec = importlib.util.spec_from_file_location("hitph_merge_shards", merge_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {merge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    while True:
        missing = [index for index in range(args.num_shards)
                   if not (args.features_dir / f"index.shard_{index:02d}_of_{args.num_shards:02d}.json").is_file()
                   or not (args.features_dir / f"extraction_stats.shard_{index:02d}_of_{args.num_shards:02d}.json").is_file()]
        if not missing:
            break
        print(f"Waiting for {len(missing)} of {args.num_shards} permanent shards; first missing: {missing[0]}", flush=True)
        time.sleep(args.poll_seconds)

    result = module.merge(args.job_dir / "input/permanent.json", args.features_dir, args.num_shards)
    print("Published permanent index:", json.dumps(result), flush=True)
    command = [
        json.loads(args.config.read_text(encoding="utf-8"))["python"],
        str(root / "scripts/run_hitph_cycle4_epochs.py"),
        "--cache-epoch1-after-permanent",
        "--config", str(args.config),
        "--work-dir", str(args.work_dir),
    ]
    print("Starting epoch-1 negative cache:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
