from __future__ import annotations

import random

import pandas as pd


CANONICAL_COLUMNS = (
    "id", "tcra", "tcrb", "peptide", "hla", "hla_sequence", "label", "split"
)


def _tcr_key(row) -> tuple[str, str]:
    return str(row.tcra), str(row.tcrb)


def build_pseudo_negatives(
    positives: pd.DataFrame,
    *,
    seed: int = 42,
    cross_epitope_per_positive: int = 1,
    tcr_swap_per_positive: int = 1,
) -> pd.DataFrame:
    """Generate split-local pseudo-negatives from canonical positive TMP rows.

    The input must use the canonical TMP schema and contain positive rows only.
    Known positive TCR+peptide pairs are excluded globally. HLA clip sequences
    are carried through from the canonical input and are never regenerated here.

    These generated labels are pseudo-negatives, not experimental non-binding
    observations.
    """
    required = set(CANONICAL_COLUMNS)
    missing = required - set(positives.columns)
    if missing:
        raise ValueError(f"Positive table is missing {sorted(missing)}")
    if cross_epitope_per_positive < 0 or tcr_swap_per_positive < 0:
        raise ValueError("Negative counts cannot be negative")
    if not positives["label"].eq(1).all():
        raise ValueError("build_pseudo_negatives expects positive rows only")

    rng = random.Random(seed)
    known_positive = {
        (str(row.tcra), str(row.tcrb), str(row.peptide))
        for row in positives.itertuples(index=False)
    }

    output = positives.loc[:, CANONICAL_COLUMNS].copy()
    output["negative_type"] = "observed_positive"
    negatives: list[dict] = []

    for split, part in positives.groupby("split", sort=False):
        rows = list(part.itertuples(index=False))

        peptides_by_hla: dict[str, list[str]] = {}
        hla_sequence_by_hla: dict[str, str] = {}
        for row in rows:
            peptides_by_hla.setdefault(str(row.hla), []).append(str(row.peptide))
            previous = hla_sequence_by_hla.setdefault(str(row.hla), str(row.hla_sequence))
            if previous != str(row.hla_sequence):
                raise ValueError(f"Inconsistent hla_sequence for {row.hla}")
        peptides_by_hla = {
            allele: sorted(set(peptides)) for allele, peptides in peptides_by_hla.items()
        }

        for row in rows:
            other_peptides = [p for p in peptides_by_hla[str(row.hla)] if p != row.peptide]
            rng.shuffle(other_peptides)
            made = 0
            for peptide in other_peptides:
                if made >= cross_epitope_per_positive:
                    break
                if (row.tcra, row.tcrb, peptide) in known_positive:
                    continue
                item = {column: getattr(row, column) for column in CANONICAL_COLUMNS}
                item.update(
                    id=f"{row.id}__neg_cross_{made}",
                    peptide=peptide,
                    hla_sequence=hla_sequence_by_hla[str(row.hla)],
                    label=0,
                    negative_type="same_hla_cross_epitope",
                )
                negatives.append(item)
                made += 1

            donors = [
                donor for donor in rows
                if str(donor.peptide) != str(row.peptide)
                and _tcr_key(donor) != _tcr_key(row)
            ]
            rng.shuffle(donors)
            made = 0
            for donor in donors:
                if made >= tcr_swap_per_positive:
                    break
                if (donor.tcra, donor.tcrb, row.peptide) in known_positive:
                    continue
                item = {column: getattr(row, column) for column in CANONICAL_COLUMNS}
                item.update(
                    id=f"{row.id}__neg_tcr_{made}",
                    tcra=donor.tcra,
                    tcrb=donor.tcrb,
                    label=0,
                    negative_type="tcr_swap",
                )
                negatives.append(item)
                made += 1

    if negatives:
        output = pd.concat([output, pd.DataFrame(negatives)], ignore_index=True, sort=False)
    if output["id"].duplicated().any():
        raise AssertionError("Negative generation produced duplicate IDs")
    return output
