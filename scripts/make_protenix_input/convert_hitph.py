#!/usr/bin/env python3
"""Convert Hi-TpH/HITPH CSV rows into the canonical TMP input CSV.

The source pseudo-sequence column ("hla") is intentionally ignored. HLA heavy
chain input is populated from Hi-TpH-hla_allele2seq.json["clip sequence"].

If one or more alleles cannot be mapped, all successfully mapped records are
still written to --output-csv and unmatched source rows are written beside it;
the command then exits with an error so missing HLA mappings are never silent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data import TMPDatasetSchema, canonical_hla, clean_sequence, validate_sequence
except ImportError:  # Support direct execution as a file.
    from data import TMPDatasetSchema, canonical_hla, clean_sequence, validate_sequence


SOURCE_COLUMNS = {"pep", "hla", "hla.allele", "ab", "label", "beta"}


def load_hla_clip_map(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for allele, record in payload.items():
        try:
            canonical = canonical_hla(allele)
        except ValueError:
            continue
        clip = clean_sequence(record.get("clip sequence", ""))
        if clip:
            mapping[canonical] = clip
    if not mapping:
        raise ValueError(f"No usable HLA clip sequences found in {path}")
    return mapping


def stratified_split(
    labels: pd.Series,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> pd.Series:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("train/validation/test fractions must all be > 0")

    rng = np.random.default_rng(seed)
    output = pd.Series(index=labels.index, dtype="object")
    for label_value, idx in labels.groupby(labels).groups.items():
        indices = np.asarray(list(idx), dtype=int)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_fraction))
        n_validation = int(round(n * validation_fraction))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_validation = min(max(n_validation, 1), n - n_train - 1)
        else:
            n_train = max(1, min(n_train, n))
            n_validation = max(0, min(n_validation, n - n_train))
        output.loc[indices[:n_train]] = "train"
        output.loc[indices[n_train:n_train + n_validation]] = "validation"
        output.loc[indices[n_train + n_validation:]] = "test"

    if output.isna().any():
        raise AssertionError("Split assignment left unassigned rows")
    return output


def convert(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    missing = sorted(SOURCE_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Missing HITPH columns {missing}; found {list(source.columns)}")

    label = pd.to_numeric(source["label"], errors="raise")
    if not label.isin([0, 1]).all():
        raise ValueError("HITPH label must contain only 0 or 1")

    alpha: list[str] = []
    beta_from_ab: list[str] = []
    for row_index, value in source["ab"].items():
        parts = [clean_sequence(piece) for piece in str(value).split("/")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Row {row_index}: ab must contain exactly 'TCRa/TCRb', got {value!r}")
        alpha.append(parts[0])
        beta_from_ab.append(parts[1])

    beta_column = source["beta"].map(clean_sequence).tolist()
    mismatch = [i for i, (a, b) in enumerate(zip(beta_from_ab, beta_column)) if a != b]
    if mismatch:
        preview = mismatch[:10]
        raise ValueError(f"beta column disagrees with ab beta-chain for rows {preview}")

    canonical_alleles = source["hla.allele"].map(canonical_hla)
    hla_map = load_hla_clip_map(args.hla_map_json)
    mapped_mask = canonical_alleles.isin(hla_map)
    unmatched = source.loc[~mapped_mask].copy()
    if not unmatched.empty:
        unmatched.insert(0, "source_row", unmatched.index)
        unmatched.insert(1, "canonical_hla", canonical_alleles.loc[~mapped_mask].values)

    keep = source.loc[mapped_mask].copy().reset_index(drop=False).rename(columns={"index": "source_row"})
    alleles = canonical_alleles.loc[mapped_mask].reset_index(drop=True)
    alpha_keep = pd.Series(alpha, index=source.index).loc[mapped_mask].reset_index(drop=True)
    beta_keep = pd.Series(beta_from_ab, index=source.index).loc[mapped_mask].reset_index(drop=True)

    output = pd.DataFrame(
        {
            "id": [f"hitph_{i:08d}" for i in range(len(keep))],
            "tcra": alpha_keep,
            "tcrb": beta_keep,
            "peptide": keep["pep"].map(clean_sequence),
            "hla": alleles,
            "hla_sequence": alleles.map(hla_map),
            "label": pd.to_numeric(keep["label"], errors="raise").astype(int),
        }
    )

    schema = TMPDatasetSchema()
    for row in output.itertuples(index=False):
        for field in ("tcra", "tcrb", "peptide", "hla_sequence"):
            validate_sequence(str(getattr(row, field)), field, str(row.id))

    output["split"] = stratified_split(
        output["label"],
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    return output.loc[:, schema.columns()], unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--hla-map-json", required=True, help="Hi-TpH data/Hi-TpH-hla_allele2seq.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    output, unmatched = convert(args)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    report = {
        "input_rows": int(len(pd.read_csv(args.input_csv))),
        "mapped_rows": int(len(output)),
        "unmatched_rows": int(len(unmatched)),
        "label_counts": {str(k): int(v) for k, v in output["label"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in output["split"].value_counts().items()},
        "hla_source": str(args.hla_map_json),
        "hla_sequence_field": "clip sequence",
        "source_pseudo_sequence_used": False,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote mapped TMP CSV: {output_path}")
    if not unmatched.empty:
        unmatched_path = output_path.with_suffix(output_path.suffix + ".unmatched.csv")
        unmatched.to_csv(unmatched_path, index=False)
        raise RuntimeError(
            f"{len(unmatched)} HLA records could not be mapped. "
            f"Mapped records were saved to {output_path}; unmatched rows: {unmatched_path}"
        )


if __name__ == "__main__":
    main()
