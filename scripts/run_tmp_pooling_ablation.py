#!/usr/bin/env python3
"""Run matched A/B/C pooling trials on the fixed Stage 1 small-set manifest.

Supports sequential trials or concurrent single-GPU trials selected by available
memory, regardless of GPU utilization. This script never generates a new split,
negatives or feature cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def audit_manifest(manifest: Path, feature_dir: Path, expected_counts: list[int]) -> dict:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "original_id", "pmhc_id", "split", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        rows = list(reader)
    ids = [row["sample_id"] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest sample_id must be non-empty and unique")
    split_counts = Counter(row["split"] for row in rows)
    expected = dict(zip(("train", "validation", "test"), expected_counts))
    if dict(split_counts) != expected:
        raise ValueError(f"Expected split counts {expected}, found {dict(split_counts)}")
    class_counts = {}
    for split in expected:
        labels = [row["label"] for row in rows if row["split"] == split]
        if set(labels) != {"0", "1"}:
            raise ValueError(f"{split} needs both binary labels encoded as 0/1")
        class_counts[split] = dict(Counter(labels))
    missing = [sample_id for sample_id in ids if not (feature_dir / f"{sample_id}.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} feature files; first: {missing[0]}")
    return {
        "split_counts": dict(split_counts),
        "class_counts": class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--modes", choices=("A", "B", "C"), nargs="+", default=["A", "B", "C"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--expected-counts", type=int, nargs=3, default=[388, 48, 48])
    parser.add_argument("--pairformer-blocks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Complete epochs per trial; overrides --max-steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=None,
                        help="Deprecated; validation runs once per epoch")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early_stop", "--early-stop", action="store_true")
    parser.add_argument("--early_stop_patience", "--early-stop-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--parallel-runs", type=int, default=1,
                        help="Number of independent single-GPU trials launched together")
    parser.add_argument("--gpu-ids", nargs="+", default=None,
                        help="GPU IDs, or 'auto' to select by remaining GPU memory")
    parser.add_argument("--gpu-memory-per-run-mib", type=int, default=9216,
                        help="Conservative memory budget for each concurrent trial")
    parser.add_argument("--gpu-reserve-mib", type=int, default=4096,
                        help="Memory left unallocated on every selected GPU")
    parser.add_argument("--wait-for-gpus", action="store_true",
                        help="Wait for enough free GPU memory before launching trials")
    parser.add_argument("--dry-run", action="store_true", help="Audit paths/counts and print commands without training or writing files")
    args = parser.parse_args()
    if args.parallel_runs < 1 or args.gpu_memory_per_run_mib < 1 or args.gpu_reserve_mib < 0:
        parser.error("Run count and memory budget must be positive; reserve must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Seeds and modes must be unique")
    if args.pairformer_blocks < 0 or min(args.expected_counts) < 1:
        parser.error("Pairformer blocks must be nonnegative and expected counts positive")
    manifest = args.manifest.resolve()
    feature_dir = args.features_dir.resolve()
    output_root = args.output_root.resolve()
    audit = audit_manifest(manifest, feature_dir, args.expected_counts)
    trials = [(mode, seed, output_root / f"{mode}_seed{seed}") for mode in args.modes for seed in args.seeds]
    if not args.dry_run and output_root.exists():
        raise FileExistsError(f"Use a fresh output root to avoid overwriting results: {output_root}")
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(audit, indent=2), flush=True)
    commands = []
    for mode, seed, out in trials:
        command = [
            sys.executable, str(repo / "scripts/train_tmp_specificity.py"),
            "--manifest", str(manifest), "--features-dir", str(feature_dir),
            "--output-dir", str(out), "--pooling-mode", mode, "--seed", str(seed),
        ]
        for name in (
            "pairformer_blocks", "max_steps", "learning_rate", "weight_decay",
            "gradient_accumulation", "hidden_dim", "dropout",
            "early_stop_patience", "lr_scheduler", "lr_scheduler_factor",
            "lr_scheduler_patience", "min_lr", "precision",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.early_stop:
            command.append("--early_stop")
        commands.append(command)
    if not args.dry_run:
        output_root.mkdir(parents=True)
        config = {
            **vars(args), "manifest": str(manifest), "features_dir": str(feature_dir),
            "output_root": str(output_root), "audit": audit, "commands": commands,
        }
        (output_root / "ablation_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    if args.dry_run:
        for command in commands:
            print(shlex.join(command), flush=True)
        return
    if args.parallel_runs == 1 and args.gpu_ids is None:
        for command in commands:
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=repo, check=True)
    else:
        run_parallel(commands, repo, args)


def gpu_memory_slots(args: argparse.Namespace) -> list[str]:
    """Allocate up to three trials per GPU by free memory, ignoring utilization."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True,
    )
    allowed = None if args.gpu_ids in (None, ["auto"]) else set(args.gpu_ids)
    available = {}
    for line in output.splitlines():
        gpu_id, used, total = [part.strip() for part in line.split(",")]
        if allowed is None or gpu_id in allowed:
            available[gpu_id] = int(total) - int(used) - args.gpu_reserve_mib
    slots, counts = [], Counter()
    while available:
        gpu_id = max(available, key=available.get)
        if available[gpu_id] < args.gpu_memory_per_run_mib:
            break
        slots.append(gpu_id)
        counts[gpu_id] += 1
        available[gpu_id] -= args.gpu_memory_per_run_mib
        if counts[gpu_id] == 3:
            del available[gpu_id]
    return slots


def run_parallel(commands: list[list[str]], repo: Path, args: argparse.Namespace) -> None:
    """Run trials concurrently, assigning single GPUs within memory budgets."""
    width = min(args.parallel_runs, len(commands))
    while True:
        slots = gpu_memory_slots(args)
        if len(slots) >= width:
            break
        if not args.wait_for_gpus:
            raise RuntimeError(f"Need {width} concurrent trial slots; memory permits {len(slots)}")
        print(f"Waiting for GPU memory: {len(slots)} slots available; need {width}", flush=True)
        time.sleep(30)
    assignments = [slots[index % width] for index in range(len(commands))]
    (args.output_root / "gpu_assignments.json").write_text(
        json.dumps(dict(zip([f"{m}_seed{s}" for m in args.modes for s in args.seeds], assignments)), indent=2),
        encoding="utf-8",
    )
    for start in range(0, len(commands), width):
        workers, handles = [], []
        try:
            for index in range(start, min(start + width, len(commands))):
                command = commands[index]
                out = Path(command[command.index("--output-dir") + 1])
                handle = (args.output_root / f"{out.name}.console.log").open("w", encoding="utf-8")
                handles.append(handle)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = assignments[index]
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                print(f"GPU {assignments[index]}: {shlex.join(command)}", flush=True)
                workers.append(subprocess.Popen(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT))
            while any(worker.poll() is None for worker in workers):
                failures = [worker.returncode for worker in workers if worker.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f"Pooling trial failed: exit codes {failures}; inspect console logs")
                time.sleep(2)
            if any(worker.returncode != 0 for worker in workers):
                raise RuntimeError("Pooling trial failed; inspect console logs")
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                worker.wait()
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
