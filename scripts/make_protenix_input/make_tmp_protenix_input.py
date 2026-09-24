#!/usr/bin/env python3
"""Convert a fixed-format TCR-pMHC CSV file to Protenix complex JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .data import TMPDatasetSchema, load_tmp_csv
except ImportError:  # Support direct execution as a file.
    from data import TMPDatasetSchema, load_tmp_csv


ENTITY_SEQUENCE_COLUMNS = ("tcra", "tcrb", "peptide", "hla_sequence")


def protein_chain(sequence: str) -> dict[str, dict[str, object]]:
    """Wrap one amino-acid sequence in Protenix's proteinChain schema."""
    return {
        "proteinChain": {
            "count": 1,
            "sequence": sequence,
            "modifications": [],
        }
    }


def build_protenix_record(row: object, schema: TMPDatasetSchema) -> dict[str, object]:
    """Build one four-chain complex in TCRa/TCRb/peptide/MHC order."""
    sequences = [
        protein_chain(str(getattr(row, column)))
        for column in ENTITY_SEQUENCE_COLUMNS
    ]
    return {
        "name": str(getattr(row, schema.id_column)),
        "covalent_bonds": [],
        "sequences": sequences,
    }


def build_protenix_inputs(
    frame: pd.DataFrame,
    schema: TMPDatasetSchema = TMPDatasetSchema(),
) -> list[dict]:
    """Convert all validated CSV rows into Protenix input records."""
    return [
        build_protenix_record(row, schema)
        for row in frame.itertuples(index=False)
    ]


def write_protenix_json(records: list[dict], output_path: str | Path) -> Path:
    """Serialize Protenix records as a UTF-8 JSON array and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    """Parse the minimal CSV-to-JSON command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Fixed-format input CSV.")
    parser.add_argument("--output-json", required=True, help="Protenix JSON to create.")
    return parser.parse_args()


def main() -> None:
    """Validate the source CSV and write the corresponding Protenix JSON."""
    args = parse_args()
    schema = TMPDatasetSchema()
    frame = load_tmp_csv(args.input_csv, schema)
    records = build_protenix_inputs(frame, schema)
    output_path = write_protenix_json(records, args.output_json)
    print(f"Wrote {len(records)} TMP complexes to {output_path}")


if __name__ == "__main__":
    main()
