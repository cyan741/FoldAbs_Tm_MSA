#!/usr/bin/env python3
"""Build split-local, positive-excluded pseudo-negatives for TMP training."""

from __future__ import annotations

import argparse
from pathlib import Path

from foldabs_repro.tmp_prediction.data import load_tmp_csv
from foldabs_repro.tmp_prediction.negative_sampling import build_pseudo_negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross-epitope-per-positive", type=int, default=1)
    parser.add_argument("--tcr-swap-per-positive", type=int, default=1)
    args = parser.parse_args()
    positives = load_tmp_csv(args.positives_csv, require_split=True)
    sampled = build_pseudo_negatives(
        positives,
        seed=args.seed,
        cross_epitope_per_positive=args.cross_epitope_per_positive,
        tcr_swap_per_positive=args.tcr_swap_per_positive,
    )
    output = sampled.rename(
        columns={
            "sample_id": "ID",
            "tcra": "TCRa",
            "tcrb": "TCRb",
            "peptide": "Peptide",
            "hla": "HLA",
        }
    )
    keep = ["ID", "TCRa", "TCRb", "Peptide", "HLA", "label", "split", "negative_type"]
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output[keep].to_csv(path, index=False)
    print(f"Wrote {len(output)} positives+pseudo-negatives to {path}")
    print(output["negative_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
