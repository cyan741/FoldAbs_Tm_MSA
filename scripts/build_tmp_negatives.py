#!/usr/bin/env python3
"""Build split-local, positive-excluded pseudo-negatives for canonical TMP data.

This utility is kept for future datasets that require pseudo-negative sampling.
The current HITPH Stage-1 dataset already contains label=0 records and should
NOT be passed through this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from foldabs_repro.tmp_prediction.negative_sampling import (
    CANONICAL_COLUMNS,
    build_pseudo_negatives,
)


def load_canonical_positive_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical TMP columns {missing}; found {list(frame.columns)}")
    frame = frame.loc[:, CANONICAL_COLUMNS].copy()
    if not frame["label"].eq(1).all():
        raise ValueError(
            "build_tmp_negatives.py expects a positive-only canonical TMP CSV. "
            "HITPH already contains negatives in Stage 1, so use it directly instead."
        )
    if frame["id"].duplicated().any():
        raise ValueError("id must be unique")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-epitope-per-positive", type=int, default=1)
    parser.add_argument("--tcr-swap-per-positive", type=int, default=1)
    args = parser.parse_args()

    positives = load_canonical_positive_csv(args.positives_csv)
    sampled = build_pseudo_negatives(
        positives,
        seed=args.seed,
        cross_epitope_per_positive=args.cross_epitope_per_positive,
        tcr_swap_per_positive=args.tcr_swap_per_positive,
    )

    keep = [*CANONICAL_COLUMNS, "negative_type"]
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    sampled.loc[:, keep].to_csv(path, index=False)

    print(f"Wrote {len(sampled)} positives+pseudo-negatives to {path}")
    print(sampled["negative_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
