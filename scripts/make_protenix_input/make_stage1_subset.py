#!/usr/bin/env python3
"""Select deterministic, label-balanced rows from each existing TMP split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .data import load_tmp_csv
except ImportError:  # Support direct execution as a file.
    from data import load_tmp_csv


def balanced_sample(part: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Sample an equal number of positive and negative rows from one split."""
    if size < 2 or size % 2:
        raise ValueError("Each requested split size must be a positive even number")
    per_label = size // 2
    sampled = []
    for label in (0, 1):
        candidates = part.loc[part["label"].eq(label)]
        if len(candidates) < per_label:
            raise ValueError(
                f"Need {per_label} rows with label={label}, found {len(candidates)}"
            )
        sampled.append(candidates.sample(n=per_label, random_state=seed + label))
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    """Load a canonical CSV and write one balanced Stage-1 subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_tmp_csv(args.input_csv)
    sizes = {
        "train": args.train_size,
        "validation": args.validation_size,
        "test": args.test_size,
    }
    output = pd.concat(
        [
            balanced_sample(
                source.loc[source["split"].eq(split)],
                size,
                args.seed + index * 10,
            )
            for index, (split, size) in enumerate(sizes.items())
        ],
        ignore_index=True,
    )
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(pd.crosstab(output["split"], output["label"]).to_string())
    print(f"Wrote {len(output)} rows to {path}")


if __name__ == "__main__":
    main()
