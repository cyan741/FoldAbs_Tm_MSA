from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ACCEPTED_AA = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUOJ")


@dataclass(frozen=True)
class TMPDatasetSchema:
    id_column: str = "ID"
    tcra_column: str = "TCRa"
    tcrb_column: str = "TCRb"
    peptide_column: str = "Peptide"
    hla_column: str = "HLA"
    label_column: str = "label"
    split_column: str = "split"


def clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def canonical_hla(value: object) -> str:
    allele = str(value).strip().upper().replace("HLA-", "")
    match = re.fullmatch(r"([A-Z]+)\*?(\d{2}):?(\d{2})", allele)
    if not match:
        raise ValueError(f"Unsupported HLA allele notation: {value!r}")
    locus, field1, field2 = match.groups()
    return f"HLA-{locus}*{field1}:{field2}"


def _validate_sequence(sequence: str, field: str, sample_id: str) -> None:
    if not sequence:
        raise ValueError(f"{field} is empty for {sample_id}")
    invalid = sorted(set(sequence) - ACCEPTED_AA)
    if invalid:
        raise ValueError(f"Invalid residues {invalid} in {field} for {sample_id}")


def load_tmp_csv(
    path: str | Path,
    schema: TMPDatasetSchema = TMPDatasetSchema(),
    *,
    require_label: bool = False,
    require_split: bool = False,
) -> pd.DataFrame:
    """Load paired-alpha/beta MHC-I records without inventing missing chains."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        schema.tcra_column,
        schema.tcrb_column,
        schema.peptide_column,
        schema.hla_column,
    }
    if require_label:
        required.add(schema.label_column)
    if require_split:
        required.add(schema.split_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing TMP columns {missing}; found {list(frame.columns)}")

    ids = (
        frame[schema.id_column].astype(str)
        if schema.id_column in frame.columns
        else pd.Series([f"tmp_{index:07d}" for index in range(len(frame))])
    )
    output = pd.DataFrame(
        {
            "sample_id": ids,
            "original_id": ids,
            "tcra": frame[schema.tcra_column].map(clean_sequence),
            "tcrb": frame[schema.tcrb_column].map(clean_sequence),
            "peptide": frame[schema.peptide_column].map(clean_sequence),
            "hla": frame[schema.hla_column].map(canonical_hla),
        }
    )
    if output["sample_id"].duplicated().any():
        duplicates = output.loc[output["sample_id"].duplicated(), "sample_id"].tolist()[:5]
        raise ValueError(f"sample_id must be unique; duplicates include {duplicates}")
    for row in output.itertuples(index=False):
        _validate_sequence(row.tcra, "TCRa", row.sample_id)
        _validate_sequence(row.tcrb, "TCRb", row.sample_id)
        _validate_sequence(row.peptide, "Peptide", row.sample_id)

    if schema.label_column in frame.columns:
        labels = pd.to_numeric(frame[schema.label_column], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("TMP labels must be binary 0/1")
        output["label"] = labels.astype(int)
    if schema.split_column in frame.columns:
        split = frame[schema.split_column].astype(str).str.lower()
        allowed = {"train", "validation", "test"}
        invalid = sorted(set(split) - allowed)
        if invalid:
            raise ValueError(f"Unknown split values: {invalid}")
        output["split"] = split
    output["pmhc_id"] = output["peptide"] + "|" + output["hla"]
    return output


def load_hla_sequences(path: str | Path) -> dict[str, str]:
    """Read allele->full extracellular heavy-chain sequence JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("HLA sequence file must be a non-empty JSON object")
    result: dict[str, str] = {}
    for allele, sequence in payload.items():
        key = canonical_hla(allele)
        value = clean_sequence(sequence)
        _validate_sequence(value, "HLA heavy sequence", key)
        result[key] = value
    return result
