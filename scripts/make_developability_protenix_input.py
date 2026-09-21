#!/usr/bin/env python3
"""Convert a developability CSV into antibody-only Protenix inference JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foldabs_repro.data import DatasetSchema, load_fitness_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Protenix inputs for a developability dataset.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--heavy-column", default="heavy")
    parser.add_argument("--light-column", default="light")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    frame = load_fitness_csv(
        args.input_csv,
        schema=DatasetSchema(
            heavy_column=args.heavy_column,
            light_column=args.light_column,
            label_column=args.label_column,
        ),
    )
    if args.limit:
        frame = frame.iloc[: args.limit]

    records = []
    for row in frame.itertuples(index=False):
        chains = [
            {"proteinChain": {"count": 1, "sequence": row.heavy, "modifications": []}}
        ]
        if row.light:
            chains.append(
                {"proteinChain": {"count": 1, "sequence": row.light, "modifications": []}}
            )
        records.append({"name": row.sample_id, "covalent_bonds": [], "sequences": chains})

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} antibody inputs to {output_path}")


if __name__ == "__main__":
    main()
