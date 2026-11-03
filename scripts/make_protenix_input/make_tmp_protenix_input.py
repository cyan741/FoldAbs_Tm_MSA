#!/usr/bin/env python3
"""Convert a fixed-format TCR-pMHC CSV file to Protenix complex JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(import_root))

import pandas as pd

from foldabs_repro.tmp_prediction.canonical import TMPDatasetSchema, load_tmp_csv

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", help="Fixed-format input CSV.")
    source.add_argument("--hitph-splits-dir", help="Delivered dynamic-train / fixed-evaluation split directory")
    parser.add_argument("--no-msa", action="store_true", help="Compatibility flag; inputs are always generated without MSA")
    parser.add_argument("--train-negative-epoch", type=int,
                        help="Prepare sampled train negatives for one epoch instead of the permanent baseline")
    parser.add_argument("--output-json", required=True, help="Protenix JSON to create.")
    parser.add_argument("--shard-size", type=int, default=0, help="Write stable ID-sorted JSON shards")
    return parser.parse_args()


def prepare_hitph_inputs(splits_dir, output_json, *, epoch=None, shard_size=0):
    """Export standard no-MSA complexes with training IDs and fixed evaluation negatives."""
    from types import SimpleNamespace
    from scripts.train_tmp_specificity import PeptideMismatchSampling

    if shard_size < 0 or (epoch is not None and epoch < 1):
        raise ValueError("Invalid shard size or epoch")
    protocol = PeptideMismatchSampling(SimpleNamespace(
        splits_dir=str(splits_dir), negatives_per_positive=2, negative_sampling_seed=42))
    if epoch is None:
        manifest = pd.concat([protocol.manifest(protocol.frames[split], split)
                              for split in ("train", "val", "test")], ignore_index=True)
    else:
        manifest, _ = protocol.epoch(epoch)
        manifest = manifest[manifest.label.eq(0)].reset_index(drop=True)
    if manifest.sample_id.duplicated().any():
        raise ValueError("Duplicate cache sample IDs")
    records = build_protenix_inputs(manifest.rename(columns={'sample_id': 'id'}))
    records.sort(key=lambda record: record['name'])
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shard_size:
        for index, start in enumerate(range(0, len(records), shard_size)):
            shard = output.with_name(f"{output.stem}_{index:05d}{output.suffix}")
            part = records[start:start+shard_size]
            write_protenix_json(part, shard)
    else:
        write_protenix_json(records, output)
    manifest.to_csv(output.with_suffix('.manifest.csv'), index=False)
    return {'records': len(records), 'positive': int(manifest.label.eq(1).sum()),
            'negative': int(manifest.label.eq(0).sum()), 'epoch': epoch,
            'shard_size': shard_size, 'use_msa': False, 'output_json': str(output)}


def main() -> None:
    """Validate the source CSV and write the corresponding Protenix JSON."""
    args = parse_args()
    if args.hitph_splits_dir:
        print(json.dumps(prepare_hitph_inputs(args.hitph_splits_dir,
                                             args.output_json, epoch=args.train_negative_epoch,
                                             shard_size=args.shard_size), indent=2))
        return
    if args.train_negative_epoch is not None:
        raise ValueError("--train-negative-epoch requires --hitph-splits-dir")
    schema = TMPDatasetSchema()
    frame = load_tmp_csv(args.input_csv, schema)
    records = build_protenix_inputs(frame, schema)
    if args.shard_size < 0:
        raise ValueError("--shard-size must be nonnegative")
    output_path = Path(args.output_json)
    if args.shard_size:
        records.sort(key=lambda record: record["name"])
        for index, start in enumerate(range(0, len(records), args.shard_size)):
            shard = output_path.with_name(f"{output_path.stem}_{index:05d}{output_path.suffix}")
            write_protenix_json(records[start:start + args.shard_size], shard)
    else:
        write_protenix_json(records, output_path)
    print(f"Wrote {len(records)} TMP complexes to {output_path}")


if __name__ == "__main__":
    main()
