#!/usr/bin/env python3
"""Build a compact nanobody sequence database from cached full-database MSAs.

This implements the public idea in AF3-TurboAb: take MSAs for a representative
set of nanobodies, aggregate their homologous sequences, and deduplicate them
into a much smaller search database.  The exact 200 sequences, CDR annotation,
germline calls, and source database used by that paper are not public here.
Consequently seed diversity is a documented proxy: heavy-chain length plus
3-mer sequence diversity, never thermostability labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd


STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def _read_fasta(path: Path):
    name: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def _ungapped_sequence(a3m_sequence: str) -> str:
    sequence = a3m_sequence.replace("-", "").upper()
    if not sequence or not set(sequence).issubset(STANDARD_AA):
        return ""
    return sequence


def _kmers(sequence: str, k: int = 3) -> frozenset[str]:
    if len(sequence) < k:
        return frozenset({sequence})
    return frozenset(sequence[index : index + k] for index in range(len(sequence) - k + 1))


def _distance(left: dict, right: dict, length_span: int) -> float:
    union = left["kmers"] | right["kmers"]
    jaccard_distance = 0.0 if not union else 1.0 - len(left["kmers"] & right["kmers"]) / len(union)
    length_distance = abs(len(left["heavy"]) - len(right["heavy"])) / max(length_span, 1)
    return 0.8 * jaccard_distance + 0.2 * length_distance


def _select_diverse(rows: list[dict], count: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Need at least {count} nanobody candidates; found {len(rows)}")
    rows = sorted(rows, key=lambda row: row["heavy"])
    length_span = max(map(lambda row: len(row["heavy"]), rows)) - min(map(lambda row: len(row["heavy"]), rows))
    selected: list[dict] = []
    # Starting at both length extremes prevents the greedy phase from silently
    # collapsing to only the most common nanobody length.
    for start in (min(rows, key=lambda row: (len(row["heavy"]), row["heavy"])), max(rows, key=lambda row: (len(row["heavy"]), row["heavy"]))):
        if start not in selected:
            selected.append(start)
    while len(selected) < count:
        candidates = [row for row in rows if row not in selected]
        next_row = max(
            candidates,
            key=lambda row: (
                min(_distance(row, previous, length_span) for previous in selected),
                row["heavy"],
            ),
        )
        selected.append(next_row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact nanobody MSA search database.")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--msa-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--source-contains", default="nbthermo_dsf_sypro")
    parser.add_argument(
        "--max-hits-per-seed",
        type=int,
        default=1000,
        help="Keep this many A3M rows per seed before global de-duplication; 0 keeps all rows.",
    )
    args = parser.parse_args()
    if args.seed_count < 1 or args.max_hits_per_seed < 0:
        raise ValueError("--seed-count must be positive and --max-hits-per-seed cannot be negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = json.loads((Path(args.msa_cache_dir).resolve() / "manifest.json").read_text(encoding="utf-8"))
    data = pd.read_csv(Path(args.data_csv).resolve(), encoding="utf-8-sig")
    required = {"heavy", "light", "source"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset lacks columns needed for compact-Nb selection: {missing}")
    data["heavy"] = data["heavy"].map(_clean_sequence)
    data["light"] = data["light"].map(_clean_sequence)
    selected_source = data["source"].astype(str).str.contains(args.source_contains, regex=False, na=False)
    candidates = data[selected_source & data["light"].eq("") & data["heavy"].ne("")].drop_duplicates("heavy")
    rows = [
        {"heavy": sequence, "kmers": _kmers(sequence)}
        for sequence in sorted(candidates["heavy"].tolist())
        if set(sequence).issubset(STANDARD_AA)
    ]
    seeds = _select_diverse(rows, args.seed_count)

    database: OrderedDict[str, str] = OrderedDict()
    seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds):
        sequence = str(seed["heavy"])
        entry = cache_manifest.get(_key(sequence), {})
        a3m_path = Path(entry.get("unpairedMsaPath", ""))
        if not a3m_path.is_file():
            raise FileNotFoundError(
                f"Selected seed {seed_index} lacks a cached unpaired MSA: {sequence[:20]}..."
            )
        rows_seen = 0
        rows_kept = 0
        for hit_index, (_name, aligned) in enumerate(_read_fasta(a3m_path)):
            if args.max_hits_per_seed and hit_index >= args.max_hits_per_seed:
                break
            rows_seen += 1
            hit = _ungapped_sequence(aligned)
            if not hit:
                continue
            if hit not in database:
                database[hit] = f"nb_seed_{seed_index:03d}_hit_{hit_index:05d}"
                rows_kept += 1
        seed_rows.append(
            {
                "seed_index": seed_index,
                "heavy_sequence": sequence,
                "length": len(sequence),
                "a3m_path": str(a3m_path),
                "a3m_rows_examined": rows_seen,
                "new_unique_database_sequences": rows_kept,
            }
        )

    fasta_path = output_dir / "nb_compact_database.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence, identifier in database.items():
            handle.write(f">{identifier}\n{sequence}\n")
    seeds_path = output_dir / "selected_seeds.csv"
    with seeds_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    report = {
        "scope": "Paper-inspired compact nanobody sequence database for local MSA retrieval",
        "paper_alignment": "Aggregate and de-duplicate MSA hits from 200 representative nanobodies.",
        "known_approximation": (
            "The source paper's exact seed list, CDR annotations, germline usage, templates, and database "
            "are unavailable. Seeds here use only public NbThermo sequence identity, length, and 3-mer diversity; "
            "no thermostability label is used."
        ),
        "source_filter": args.source_contains,
        "candidate_nanobody_sequences": len(rows),
        "seed_count": len(seeds),
        "max_hits_per_seed": args.max_hits_per_seed,
        "unique_database_sequences": len(database),
        "database_residues": sum(map(len, database)),
        "database_fasta": str(fasta_path),
        "selected_seeds_csv": str(seeds_path),
    }
    report_path = output_dir / "compact_database_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Compact database: {fasta_path}")


if __name__ == "__main__":
    main()
