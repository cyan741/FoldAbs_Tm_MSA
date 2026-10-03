#!/usr/bin/env python3
"""Summarize completed pooling runs and paired differences; never select on test."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("bce", "auroc", "auprc", "auc_0_1")


def stats(values: list[float]) -> dict:
    finite = [v for v in values if math.isfinite(v)]
    return {
        "n": len(finite),
        "mean": statistics.mean(finite) if finite else None,
        "std": statistics.stdev(finite) if len(finite) > 1 else None,
    }


def summarize(root: Path) -> dict:
    rows, configs = [], []
    for path in sorted(root.glob("*_seed*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        configs.append(config)
        mode, seed = config["pooling_mode"], config["seed"]
        if mode not in ("A", "B", "C"):
            raise ValueError(f"Invalid pooling mode in {path}")
        for split in ("train", "validation", "test"):
            rows.append({
                "mode": mode, "seed": seed, "split": split,
                "best_step": metrics["best_step"],
                "pooling_parameters": config["pooling_parameters"],
                **{name: float(metrics[split][name]) for name in METRICS},
            })
    if not rows:
        raise ValueError("No completed pooling runs found")
    # Fail rather than silently averaging runs with different splits/configs.
    controls = (
        "manifest_sha256", "features_dir", "git_sha", "feature_schema",
        "selection_metric", "pairformer_blocks", "max_steps", "learning_rate",
        "weight_decay", "gradient_accumulation", "validate_every", "hidden_dim", "dropout",
        "auto_pos_weight",
    )
    for key in controls:
        if any(c[key] != configs[0][key] for c in configs):
            raise ValueError(f"Mismatched comparison setting: {key}")
    common_model = {k: v for k, v in configs[0]["model_config"].items() if k != "pooling_mode"}
    for config in configs:
        if {k: v for k, v in config["model_config"].items() if k != "pooling_mode"} != common_model:
            raise ValueError("Mismatched model config beyond pooling_mode")
    by_key = {(r["mode"], r["seed"], r["split"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate mode/seed/split results")
    modes = sorted({r["mode"] for r in rows})
    aggregates, paired = {}, {}
    for mode in modes:
        aggregates[mode] = {
            split: {name: stats([r[name] for r in rows if r["mode"] == mode and r["split"] == split]) for name in METRICS}
            for split in ("train", "validation", "test")
        }
    for mode in ("B", "C"):
        if mode not in modes or "A" not in modes:
            continue
        paired[mode + "-A"] = {}
        for split in ("train", "validation", "test"):
            seeds = sorted({seed for m, seed, s in by_key if m == mode and s == split}
                           & {seed for m, seed, s in by_key if m == "A" and s == split})
            paired[mode + "-A"][split] = {
                "seeds": seeds,
                **{name: {
                    **stats([by_key[mode, seed, split][name] - by_key["A", seed, split][name] for seed in seeds]),
                    "per_seed": {
                        str(seed): delta if math.isfinite(delta) else None
                        for seed in seeds
                        for delta in [by_key[mode, seed, split][name] - by_key["A", seed, split][name]]
                    },
                } for name in METRICS},
            }
    with (root / "pooling_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "comparison_controls": {key: configs[0][key] for key in controls},
        "aggregates": aggregates, "paired_deltas": paired,
        "note": "For deltas, lower BCE is better; higher AUROC/AUPRC/auc_0_1 is better. Test is report-only; no winner is selected.",
    }
    # No single-class NaN metrics are serialized as misleading finite scores.
    (root / "pooling_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs_root.resolve())
    print(f"Wrote pooling_runs.csv and pooling_summary.json in {args.runs_root}")


if __name__ == "__main__":
    main()
