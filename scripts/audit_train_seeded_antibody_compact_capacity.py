#!/usr/bin/env python3
"""Audit a broad antibody compact-MSA proxy built from *training* seeds only.

The older compact proxy selected 200 nanobody heavy chains, which makes it a
reasonable nanobody experiment but a biased parent library for the mixed
heavy/light antibody thermostability benchmark.  This audit instead selects
three sequence strata from the fixed training split only:

* nanobody heavy chains;
* conventional-antibody heavy chains; and
* conventional-antibody light chains.

No held-out test or validation sequence can be selected.  The A3M hits are
read from the already available ColabFold cache, so this audit makes no network
request and does not use labels, GPU features, losses, or predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_compact_nb_msa_database import (
    STANDARD_AA,
    _clean_sequence,
    _key,
    _kmers,
    _read_fasta,
    _select_diverse,
    _ungapped_sequence,
)


def _parse_caps(value: str) -> list[int]:
    try:
        caps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--caps must be comma-separated positive integers") from exc
    if not caps or caps[0] < 1:
        raise argparse.ArgumentTypeError("--caps must contain positive integers")
    return caps


def _digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _unique_candidate_rows(frame: pd.DataFrame, column: str) -> list[dict]:
    """Turn one chain column into the format used by the established selector."""
    sequences = sorted({sequence for sequence in frame[column].tolist() if sequence and set(sequence).issubset(STANDARD_AA)})
    return [{"heavy": sequence, "kmers": _kmers(sequence)} for sequence in sequences]


def _select_stratum(frame: pd.DataFrame, column: str, count: int, label: str) -> list[dict[str, str | int]]:
    candidates = _unique_candidate_rows(frame, column)
    if len(candidates) < count:
        raise ValueError(f"Need {count} valid {label} seeds but training split contains only {len(candidates)}")
    selected = _select_diverse(candidates, count)
    return [
        {"stratum": label, "sequence": str(row["heavy"]), "length": len(str(row["heavy"]))}
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--msa-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--selected-seeds-csv", type=Path, required=True)
    parser.add_argument("--caps", type=_parse_caps, default=_parse_caps("1000,2500,5000,10000"))
    parser.add_argument("--nb-heavy-seeds", type=int, default=80)
    parser.add_argument("--paired-heavy-seeds", type=int, default=60)
    parser.add_argument("--paired-light-seeds", type=int, default=60)
    args = parser.parse_args()
    if min(args.nb_heavy_seeds, args.paired_heavy_seeds, args.paired_light_seeds) < 1:
        raise ValueError("Every seed-stratum count must be positive")

    manifest = pd.read_csv(args.split_manifest.resolve(), encoding="utf-8-sig")
    missing = sorted({"split", "heavy", "light", "source"} - set(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest lacks required columns: {missing}")
    manifest["heavy"] = manifest["heavy"].map(_clean_sequence)
    manifest["light"] = manifest["light"].map(_clean_sequence)
    train = manifest.loc[manifest["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("Split manifest contains no train rows")

    # The architecture labels use sequence availability only.  They do not use
    # source names, Tm labels, or any model output.
    nb_train = train.loc[train["light"].eq("") & train["heavy"].ne("")]
    paired_train = train.loc[train["light"].ne("") & train["heavy"].ne("")]
    seed_rows = (
        _select_stratum(nb_train, "heavy", args.nb_heavy_seeds, "nanobody_heavy")
        + _select_stratum(paired_train, "heavy", args.paired_heavy_seeds, "paired_heavy")
        + _select_stratum(paired_train, "light", args.paired_light_seeds, "paired_light")
    )
    for seed_index, row in enumerate(seed_rows):
        row["seed_index"] = seed_index

    cache_dir = args.msa_cache_dir.resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing cached normal-MSA manifest: {manifest_path}")
    msa_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    digest_sets = {cap: set() for cap in args.caps}
    residues = {cap: 0 for cap in args.caps}
    raw_rows = {cap: 0 for cap in args.caps}
    per_stratum = {row["stratum"]: {cap: set() for cap in args.caps} for row in seed_rows}

    for number, seed in enumerate(seed_rows, start=1):
        sequence = str(seed["sequence"])
        entry = msa_manifest.get(_key(sequence), {})
        a3m_path = Path(entry.get("unpairedMsaPath", ""))
        if not a3m_path.is_file():
            raise FileNotFoundError(
                f"Training seed {number} ({seed['stratum']}) lacks cached normal MSA: {sequence[:20]}..."
            )
        rows_available = 0
        for hit_index, (_name, aligned) in enumerate(_read_fasta(a3m_path)):
            rows_available += 1
            hit = _ungapped_sequence(aligned)
            if not hit:
                continue
            fingerprint = _digest(hit)
            for cap in args.caps:
                if hit_index >= cap:
                    continue
                raw_rows[cap] += 1
                per_stratum[str(seed["stratum"])][cap].add(fingerprint)
                if fingerprint not in digest_sets[cap]:
                    digest_sets[cap].add(fingerprint)
                    residues[cap] += len(hit)
        seed["a3m_path"] = str(a3m_path)
        seed["a3m_rows_available"] = rows_available
        print(f"Audited training seed {number}/{len(seed_rows)} ({seed['stratum']}): {rows_available} rows", flush=True)

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    selected_csv = args.selected_seeds_csv.resolve()
    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    capacities = [
        {
            "max_hits_per_seed": cap,
            "raw_valid_rows_before_global_deduplication": raw_rows[cap],
            "unique_database_sequences": len(digest_sets[cap]),
            "database_residues": residues[cap],
            "unique_sequences_seen_by_stratum_before_cross_stratum_deduplication": {
                stratum: len(values[cap]) for stratum, values in per_stratum.items()
            },
        }
        for cap in args.caps
    ]
    report = {
        "scope": "Offline capacity audit for a training-seeded, broad-antibody ColabFold compact-MSA proxy",
        "network_or_gpu_used": False,
        "test_or_validation_sequences_used_as_seeds": False,
        "source_cache": str(cache_dir),
        "seed_counts": {
            "nanobody_heavy": args.nb_heavy_seeds,
            "paired_heavy": args.paired_heavy_seeds,
            "paired_light": args.paired_light_seeds,
            "total": len(seed_rows),
        },
        "selection_rule": "training split only; length plus 3-mer diversity independently within each chain stratum",
        "interpretation_boundary": (
            "Parent hits come from the existing ColabFold normal-MSA cache, not direct UniRef retrieval. "
            "The broad seed design removes the prior Nb-only seed bias but does not make this a UniRef experiment."
        ),
        "capacities": capacities,
        "selected_seeds_csv": str(selected_csv),
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Training-seeded broad antibody capacity audit: {output_json}")


if __name__ == "__main__":
    main()
