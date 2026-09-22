from __future__ import annotations

import random

import pandas as pd


def _tcr_key(row) -> tuple[str, str]:
    return str(row.tcra), str(row.tcrb)


def build_pseudo_negatives(
    positives: pd.DataFrame,
    *,
    seed: int = 42,
    cross_epitope_per_positive: int = 1,
    tcr_swap_per_positive: int = 1,
) -> pd.DataFrame:
    """Generate partition-local negatives after the split has been fixed.

    All known positive TCR+peptide pairs are excluded across HLA alleles. This
    is intentionally conservative: pseudo-negatives are training labels, not
    experimental non-binding observations.
    """
    required = {"sample_id", "tcra", "tcrb", "peptide", "hla", "pmhc_id", "split"}
    missing = required - set(positives.columns)
    if missing:
        raise ValueError(f"Positive table is missing {sorted(missing)}")
    if cross_epitope_per_positive < 0 or tcr_swap_per_positive < 0:
        raise ValueError("Negative counts cannot be negative")
    if "label" in positives and not positives["label"].eq(1).all():
        raise ValueError("build_pseudo_negatives expects positive rows only")

    rng = random.Random(seed)
    known_positive = {
        (str(row.tcra), str(row.tcrb), str(row.peptide))
        for row in positives.itertuples(index=False)
    }
    output = positives.copy()
    output["label"] = 1
    output["negative_type"] = "observed_positive"
    negatives: list[dict] = []

    for split, part in positives.groupby("split", sort=False):
        rows = list(part.itertuples(index=False))
        peptides_by_hla: dict[str, list[str]] = {}
        for row in rows:
            peptides_by_hla.setdefault(str(row.hla), []).append(str(row.peptide))
        peptides_by_hla = {
            allele: sorted(set(peptides)) for allele, peptides in peptides_by_hla.items()
        }
        tcr_donors = rows[:]

        for row in rows:
            other_peptides = [p for p in peptides_by_hla[str(row.hla)] if p != row.peptide]
            rng.shuffle(other_peptides)
            made = 0
            for peptide in other_peptides:
                if made >= cross_epitope_per_positive:
                    break
                if (row.tcra, row.tcrb, peptide) in known_positive:
                    continue
                item = row._asdict()
                item.update(
                    sample_id=f"{row.sample_id}__neg_cross_{made}",
                    original_id=f"{row.original_id}__neg_cross_{made}",
                    peptide=peptide,
                    pmhc_id=f"{peptide}|{row.hla}",
                    label=0,
                    negative_type="same_hla_cross_epitope",
                )
                negatives.append(item)
                made += 1

            donors = [
                donor for donor in tcr_donors
                if donor.pmhc_id != row.pmhc_id and _tcr_key(donor) != _tcr_key(row)
            ]
            rng.shuffle(donors)
            made = 0
            for donor in donors:
                if made >= tcr_swap_per_positive:
                    break
                if (donor.tcra, donor.tcrb, row.peptide) in known_positive:
                    continue
                item = row._asdict()
                item.update(
                    sample_id=f"{row.sample_id}__neg_tcr_{made}",
                    original_id=f"{row.original_id}__neg_tcr_{made}",
                    tcra=donor.tcra,
                    tcrb=donor.tcrb,
                    label=0,
                    negative_type="tcr_swap",
                )
                negatives.append(item)
                made += 1

    if negatives:
        output = pd.concat([output, pd.DataFrame(negatives)], ignore_index=True, sort=False)
    if output["sample_id"].duplicated().any():
        raise AssertionError("Negative generation produced duplicate sample IDs")
    return output
