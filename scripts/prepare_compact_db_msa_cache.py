#!/usr/bin/env python3
"""Create Protenix-compatible A3M files by searching a local FASTA database.

Unlike ``prepare_protenix_msa_cache.py``, this script never calls the public
ColabFold/MMseqs service.  It runs jackhmmer against a fixed local database and
writes an ordinary ``unpairedMsaPath`` manifest, so downstream Protenix feature
extraction is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path


def _key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _valid_entry(entry: dict) -> bool:
    path = entry.get("unpairedMsaPath")
    return bool(path and Path(path).is_file() and Path(path).stat().st_size > 0)


def _read_stockholm(path: Path) -> OrderedDict[str, str]:
    """Read the aligned sequence blocks written by HMMER without extra deps."""
    chunks: OrderedDict[str, list[str]] = OrderedDict()
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == "//" or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            chunks.setdefault(fields[0], []).append(fields[1])
    return OrderedDict((name, "".join(parts)) for name, parts in chunks.items())


def _stockholm_to_a3m(stockholm_path: Path, output_path: Path, max_hits: int) -> int:
    alignment = _read_stockholm(stockholm_path)
    if not alignment:
        raise RuntimeError(f"jackhmmer produced no Stockholm alignment: {stockholm_path}")
    query_name = next((name for name in alignment if name == "query"), next(iter(alignment)))
    query = alignment[query_name]
    match_columns = [character != "-" for character in query]
    if not any(match_columns):
        raise RuntimeError(f"Query has no aligned match columns: {stockholm_path}")

    ordered_names = [query_name] + [name for name in alignment if name != query_name]
    if max_hits:
        ordered_names = ordered_names[: max_hits + 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for number, name in enumerate(ordered_names):
            sequence = alignment[name]
            if len(sequence) != len(query):
                raise RuntimeError(f"Inconsistent Stockholm row length for {name}")
            converted: list[str] = []
            for is_match, query_char, char in zip(match_columns, query, sequence):
                if is_match:
                    converted.append(char.upper() if char != "-" else "-")
                elif char != "-":
                    # A3M uses lower case for an insertion relative to query.
                    converted.append(char.lower())
            handle.write(f">query" if number == 0 else f">hit_{number:05d}")
            handle.write("\n")
            handle.write("".join(converted))
            handle.write("\n")
    return len(ordered_names)


def _write_report(
    path: Path,
    *,
    manifest: dict,
    elapsed: float,
    completed: int,
    total: int,
    database: Path,
    scope: str,
    source_tag: str,
    iterations: int,
    e_value: float,
    max_hits: int,
) -> None:
    depths = []
    for entry in manifest.values():
        a3m_path = Path(entry.get("unpairedMsaPath", ""))
        if a3m_path.is_file():
            with a3m_path.open(encoding="utf-8") as handle:
                depths.append(sum(1 for line in handle if line.startswith(">")))
    payload = {
        "scope": scope,
        "source_tag": source_tag,
        "database_fasta": str(database),
        "jackhmmer_iterations": iterations,
        "jackhmmer_e_value": e_value,
        "max_hits": max_hits,
        "valid_cache_entries": sum(_valid_entry(entry) for entry in manifest.values()),
        "completed_this_run": completed,
        "total_unique_sequences": total,
        "elapsed_seconds": round(elapsed, 3),
        "unpaired_depth": {
            "min": min(depths) if depths else 0,
            "mean": sum(depths) / len(depths) if depths else 0.0,
            "max": max(depths) if depths else 0,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_jackhmmer(*, executable: str, query: str, database: Path, destination: Path, cpu: int, iterations: int, e_value: float) -> None:
    query_path = destination / "query.fasta"
    query_path.write_text(f">query\n{query}\n", encoding="utf-8")
    stockholm_path = destination / "alignment.sto"
    command = [
        executable,
        "--cpu",
        str(cpu),
        "-N",
        str(iterations),
        "-E",
        str(e_value),
        "--incE",
        str(e_value),
        # The Stockholm file below is the machine-readable alignment we need.
        # Do not also dump every domain alignment as human-readable text into
        # jackhmmer.log: broad antibody databases can otherwise create a
        # multi-GB log before the useful A3M cache is even written.
        "--noali",
        "-A",
        str(stockholm_path),
        str(query_path),
        str(database),
    ]
    with (destination / "jackhmmer.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"jackhmmer failed for {query[:20]}... (exit {completed.returncode}); inspect {destination / 'jackhmmer.log'}"
        )
    if not stockholm_path.is_file():
        raise RuntimeError(f"jackhmmer did not create {stockholm_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable local-jackhmmer MSA cache.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--database-fasta", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--jackhmmer", default="jackhmmer")
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--e-value", type=float, default=1e-3)
    parser.add_argument(
        "--max-hits",
        type=int,
        default=0,
        help="Maximum non-query rows retained in each final A3M; 0 keeps every row.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-query Stockholm/log intermediates after a successful A3M conversion.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 processes every input record; a positive value is a pilot.")
    parser.add_argument(
        "--scope",
        default="Local jackhmmer MSA cache over the compact nanobody database",
    )
    parser.add_argument("--source-tag", default="local_compact_nb_jackhmmer")
    args = parser.parse_args()
    if args.cpu < 1 or args.iterations < 1 or args.e_value <= 0 or args.max_hits < 0 or args.limit < 0:
        raise ValueError("Invalid jackhmmer cache arguments")
    executable = shutil.which(args.jackhmmer)
    if not executable:
        raise FileNotFoundError(
            "jackhmmer was not found. Install HMMER in the foldabs environment, then rerun. "
            "For conda: conda install -n foldabs -c bioconda hmmer"
        )
    database = Path(args.database_fasta).resolve()
    if not database.is_file() or database.stat().st_size == 0:
        raise FileNotFoundError(f"Missing compact FASTA database: {database}")
    records = json.loads(Path(args.input_json).resolve().read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    sequences = list(
        dict.fromkeys(
            item["proteinChain"]["sequence"]
            for record in records
            for item in record.get("sequences", [])
            if item.get("proteinChain")
        )
    )
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    report_path = cache_dir / "cache_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    pending = [sequence for sequence in sequences if not _valid_entry(manifest.get(_key(sequence), {}))]
    started = time.perf_counter()
    print(f"Local jackhmmer MSA: unique sequences={len(sequences)} cached={len(sequences) - len(pending)} pending={len(pending)}")

    for completed, sequence in enumerate(pending, start=1):
        request_id = _key(sequence)[:16]
        request_dir = cache_dir / "requests" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        _run_jackhmmer(
            executable=executable,
            query=sequence,
            database=database,
            destination=request_dir,
            cpu=args.cpu,
            iterations=args.iterations,
            e_value=args.e_value,
        )
        a3m_path = request_dir / "non_pairing.a3m"
        depth = _stockholm_to_a3m(request_dir / "alignment.sto", a3m_path, args.max_hits)
        # The final A3M is the reproducible cache artifact.  The Stockholm
        # intermediate is much larger when a query has tens of thousands of
        # homologs, so remove it after a successful conversion by default.
        if not args.keep_intermediates:
            for temporary_name in ("alignment.sto", "jackhmmer.log"):
                temporary_path = request_dir / temporary_name
                if temporary_path.exists():
                    temporary_path.unlink()
        manifest[_key(sequence)] = {
            "sequence": sequence,
            "unpairedMsaPath": str(a3m_path),
            "source": args.source_tag,
            "a3m_depth": depth,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(
            report_path,
            manifest=manifest,
            elapsed=time.perf_counter() - started,
            completed=completed,
            total=len(sequences),
            database=database,
            scope=args.scope,
            source_tag=args.source_tag,
            iterations=args.iterations,
            e_value=args.e_value,
            max_hits=args.max_hits,
        )
        print(f"Completed local MSA {completed}/{len(pending)}; depth={depth}", flush=True)

    ready_records = []
    for record in records:
        ready = True
        for item in record.get("sequences", []):
            chain = item.get("proteinChain")
            if chain:
                entry = manifest.get(_key(chain["sequence"]), {})
                if not _valid_entry(entry):
                    ready = False
                    break
                chain["unpairedMsaPath"] = entry["unpairedMsaPath"]
        if ready:
            ready_records.append(record)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ready_records, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        report_path,
        manifest=manifest,
        elapsed=time.perf_counter() - started,
        completed=len(pending),
        total=len(sequences),
        database=database,
        scope=args.scope,
        source_tag=args.source_tag,
        iterations=args.iterations,
        e_value=args.e_value,
        max_hits=args.max_hits,
    )
    print(f"Wrote {len(ready_records)} local-MSA records: {output_path}")
    print(f"Local-MSA report: {report_path}")


if __name__ == "__main__":
    main()
