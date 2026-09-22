from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd


def _protein_chain(sequence: str) -> dict:
    return {"proteinChain": {"count": 1, "sequence": sequence, "modifications": []}}


def build_protenix_inputs(
    frame: pd.DataFrame,
    hla_sequences: dict[str, str],
    *,
    beta2m_sequence: str = "",
) -> tuple[list[dict], pd.DataFrame]:
    """Build ordered TCRa/TCRb/peptide/HLA/(beta2m) Protenix records.

    The role order is persisted in the returned manifest. Downstream code maps
    sorted Protenix asym_id values back to these roles and refuses a mismatch.
    """
    records: list[dict] = []
    manifest_rows: list[dict] = []
    for row in frame.itertuples(index=False):
        if row.hla not in hla_sequences:
            raise KeyError(f"No full HLA heavy-chain sequence for {row.hla}")
        role_sequences = [
            ("tcra", row.tcra),
            ("tcrb", row.tcrb),
            ("peptide", row.peptide),
            ("mhc", hla_sequences[row.hla]),
        ]
        if beta2m_sequence:
            role_sequences.append(("beta2m", beta2m_sequence))
        roles = [role for role, _ in role_sequences]
        records.append(
            {
                "name": row.sample_id,
                "covalent_bonds": [],
                "sequences": [_protein_chain(sequence) for _, sequence in role_sequences],
            }
        )
        item = {
            "sample_id": row.sample_id,
            "original_id": row.original_id,
            "tcra": row.tcra,
            "tcrb": row.tcrb,
            "peptide": row.peptide,
            "hla": row.hla,
            "pmhc_id": row.pmhc_id,
            "entity_roles": json.dumps(roles),
            "include_beta2m": bool(beta2m_sequence),
        }
        for optional in ("label", "split"):
            if hasattr(row, optional):
                item[optional] = getattr(row, optional)
        manifest_rows.append(item)
    return records, pd.DataFrame(manifest_rows)


def parse_roles(value: str | Iterable[str]) -> list[str]:
    roles = json.loads(value) if isinstance(value, str) else list(value)
    if len(roles) != len(set(roles)):
        raise ValueError(f"Entity roles must be unique: {roles}")
    required = {"tcra", "tcrb", "peptide", "mhc"}
    if not required.issubset(roles):
        raise ValueError(f"Entity roles are missing {sorted(required - set(roles))}")
    return roles
