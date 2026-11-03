"""Validation and loading for the fixed TCR-pMHC input CSV format."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd


ACCEPTED_AA = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUOJ")
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class TMPDatasetSchema:
    """Column names required in every normalized TCR-pMHC CSV file."""

    id_column: str = "id"
    tcra_column: str = "tcra"
    tcrb_column: str = "tcrb"
    peptide_column: str = "peptide"
    hla_column: str = "hla"
    hla_sequence_column: str = "hla_sequence"
    label_column: str = "label"
    split_column: str = "split"

    def columns(self) -> tuple[str, ...]:
        """Return schema columns in their canonical order."""
        return tuple(getattr(self, field.name) for field in fields(self))


def clean_sequence(value: object) -> str:
    """Remove whitespace and normalize an amino-acid sequence to uppercase."""
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def canonical_hla(value: object) -> str:
    """Normalize HLA spelling while preserving valid short or annotated alleles."""
    allele = str(value).strip().upper()
    if not allele:
        raise ValueError("HLA allele cannot be empty")
    if allele.startswith("H-2-"):
        return allele
    allele = allele.removeprefix("HLA-")
    two_field = re.fullmatch(r"([A-Z]+)\*?(\d{2}):?(\d{2})", allele)
    if two_field:
        locus, field1, field2 = two_field.groups()
        return f"HLA-{locus}*{field1}:{field2}"
    if not re.fullmatch(r"[A-Z]+\*\d{2}(?::\d+)*(?: [A-Z0-9]+)?", allele):
        raise ValueError(f"Unsupported HLA allele notation: {value!r}")
    return f"HLA-{allele}"


def validate_sequence(sequence: str, field: str, sample_id: str) -> None:
    """Reject empty sequences and residues outside the accepted alphabet."""
    if not sequence:
        raise ValueError(f"{field} is empty for {sample_id}")
    invalid = sorted(set(sequence) - ACCEPTED_AA)
    if invalid:
        raise ValueError(f"Invalid residues {invalid} in {field} for {sample_id}")


def load_tmp_csv(
    path: str | Path,
    schema: TMPDatasetSchema = TMPDatasetSchema(),
) -> pd.DataFrame:
    """Load, normalize, and validate one fixed-format TCR-pMHC CSV file."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(schema.columns()) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(frame.columns)}")

    output = frame.loc[:, schema.columns()].copy()
    if output[schema.id_column].isna().any():
        raise ValueError("id cannot be empty")
    output[schema.id_column] = output[schema.id_column].astype(str).str.strip()
    if output[schema.id_column].eq("").any():
        raise ValueError("id cannot be empty")
    if output[schema.id_column].duplicated().any():
        duplicates = output.loc[
            output[schema.id_column].duplicated(), schema.id_column
        ].tolist()[:5]
        raise ValueError(f"id must be unique; duplicates include {duplicates}")

    sequence_columns = (
        schema.tcra_column,
        schema.tcrb_column,
        schema.peptide_column,
        schema.hla_sequence_column,
    )
    for column in sequence_columns:
        output[column] = output[column].map(clean_sequence)
    for row in output.itertuples(index=False):
        sample_id = str(getattr(row, schema.id_column))
        for column in sequence_columns:
            validate_sequence(str(getattr(row, column)), column, sample_id)

    output[schema.hla_column] = output[schema.hla_column].map(canonical_hla)
    labels = pd.to_numeric(output[schema.label_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("label must contain only 0 or 1")
    output[schema.label_column] = labels.astype(int)

    splits = output[schema.split_column].astype(str).str.strip().str.lower()
    invalid_splits = sorted(set(splits) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unknown split values: {invalid_splits}")
    output[schema.split_column] = splits
    return output
