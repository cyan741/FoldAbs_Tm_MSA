#!/usr/bin/env python3
"""Build the training manifest from a canonical TMP CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .data import TMPDatasetSchema, load_tmp_csv
except ImportError:
    from data import TMPDatasetSchema, load_tmp_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    schema = TMPDatasetSchema()
    frame = load_tmp_csv(args.input_csv, schema)
    manifest = pd.DataFrame(
        {
            "sample_id": frame[schema.id_column].astype(str),
            "original_id": frame[schema.id_column].astype(str),
            "pmhc_id": (
                frame[schema.peptide_column].astype(str)
                + "|"
                + frame[schema.hla_column].astype(str)
            ),
            "label": frame[schema.label_column].astype(int),
            "split": frame[schema.split_column].astype(str),
        }
    )
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(f"Wrote {len(manifest)} manifest rows to {path}")
    print(manifest["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
