"""Validate identities and list just the chain cache files needed by a subset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--input-prefix")
    args = parser.parse_args()
    input_paths = ([args.splits_dir / f"{args.input_prefix}_{split}.csv" for split in ("train", "val", "test")]
                   if args.input_prefix else [args.splits_dir / "positive_splits.csv"])
    sequences = set()
    for input_path in input_paths:
        with input_path.open() as handle:
            for row in csv.DictReader(handle):
                sequences.update(row[column] for column in ("tcra_v", "tcrb_v", "peptide", "hla_clip"))
    sequences = sorted(sequences)
    spec = json.loads((args.cache_dir / "run_spec.json").read_text())
    paths = ["run_spec.json"]
    total_bytes = 0
    for index, sequence in enumerate(sequences):
        key = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        metadata_path = Path("metadata") / f"{key}.json"
        tensor_path = Path("embeddings/chains") / f"{key}.pt"
        metadata = json.loads((args.cache_dir / metadata_path).read_text())
        if metadata != {"sequence": sequence, "spec": spec}:
            raise ValueError(f"ESM identity mismatch: {key}")
        total_bytes += (args.cache_dir / tensor_path).stat().st_size
        paths.extend([str(metadata_path), str(tensor_path)])
        if (index + 1) % 1000 == 0:
            print(json.dumps({"validated_chains": index + 1}), flush=True)
    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text("\n".join(paths) + "\n")
    print(json.dumps({"chains": len(sequences), "tensor_bytes": total_bytes,
                      "output_list": str(args.output_list)}), flush=True)


if __name__ == "__main__":
    main()
