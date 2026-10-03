#!/usr/bin/env python3
"""Run matched A/B/C pooling trials on the fixed Stage 1 small-set manifest.

Runs sequentially on the CUDA device selected by CUDA_VISIBLE_DEVICES. This
script never generates a new split, negatives, or frozen feature cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
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
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Audit paths/counts and print commands without training or writing files")
    args = parser.parse_args()
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
            "gradient_accumulation", "validate_every", "hidden_dim", "dropout",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        commands.append(command)
    if not args.dry_run:
        output_root.mkdir(parents=True)
        config = {
            **vars(args), "manifest": str(manifest), "features_dir": str(feature_dir),
            "output_root": str(output_root), "audit": audit, "commands": commands,
        }
        (output_root / "ablation_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo, check=True)
    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(repo / "scripts/summarize_tmp_pooling_ablation.py"),
             "--runs-root", str(output_root)], cwd=repo, check=True,
        )


if __name__ == "__main__":
    main()
