#!/usr/bin/env python3
"""Local UniRef50 versus training-seeded compact MSA; fixed Tm six-seed control.

No remote MSA service is used. Run the CPU MSA phase first or the complete
pipeline. Output directories and a fingerprinted protocol isolate old caches.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, OrderedDict

SCRIPTS = Path(__file__).resolve().parent
EXPECTED_MD5 = "3228886e9d749f050f60e9a0ce1f727d"
EXPECTED_BYTES = 8770260598
TAG = "tm_uniref50_2026_02_trainseed200_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True)


def prepare_parent(archive, root):
    if Path(str(archive) + ".aria2").exists():
        raise RuntimeError("Download control file still exists; complete aria2 first.")
    if archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError("UniRef50 compressed size differs from release 2026_02.")
    fasta, audit = root / "uniref50.fasta", root / "parent.json"
    if audit.exists():
        saved = read(audit)
        if fasta.exists() and fasta.stat().st_size == saved["fasta_bytes"] and fasta.stat().st_mtime_ns == saved["fasta_mtime_ns"]:
            return fasta, saved
        raise RuntimeError("Previously verified parent FASTA changed; inspect parent.json.")
    print("Verifying downloaded UniRef50 MD5...", flush=True)
    if digest(archive, "md5") != EXPECTED_MD5:
        raise ValueError("UniRef50 MD5 verification failed.")
    print("Expanding UniRef50 once to a seekable FASTA for jackhmmer...", flush=True)
    part = root / "uniref50.fasta.part"
    started = time.perf_counter()
    with gzip.open(archive, "rb") as source, part.open("wb") as dest:
        while block := source.read(8 * 1024 * 1024):
            if shutil.disk_usage(root).free < len(block) + 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB disk reserve while expanding FASTA; increase disk.")
            dest.write(block)
    n = 0
    h = hashlib.sha256()
    with part.open("rb") as handle:
        for line in handle:
            h.update(line)
            n += line.startswith(b">")
    if not n:
        raise ValueError("Empty parent FASTA")
    part.replace(fasta)
    saved = dict(release="2026_02", compressed_md5=EXPECTED_MD5,
                 compressed_bytes=EXPECTED_BYTES, fasta_bytes=fasta.stat().st_size,
                 fasta_mtime_ns=fasta.stat().st_mtime_ns, fasta_sha256=h.hexdigest(),
                 sequences=n, decompression_and_inventory_seconds=time.perf_counter() - started)
    write(audit, saved)
    return fasta, saved


def select_seeds(manifest, records, root):
    import pandas as pd
    from audit_train_seeded_antibody_compact_capacity import _select_stratum
    from build_compact_nb_msa_database import _clean_sequence
    frame = pd.read_csv(manifest).fillna("")
    for col in ("heavy", "light"):
        frame[col] = frame[col].map(_clean_sequence)
    if Counter(frame.split) != {"train": 340, "validation": 30, "test": 63}:
        raise ValueError("Expected existing fixed 340/30/63 Tm manifest")
    mapping = {str(r["sample_id"]): (r["heavy"], r["light"]) for r in frame.to_dict("records")}
    if len(mapping) != 433 or len(records) != 433:
        raise ValueError("Expected 433 unique antibody rows")
    for record in records:
        seqs = [i["proteinChain"]["sequence"] for i in record["sequences"]]
        if tuple(seqs + [""] * (2 - len(seqs))) != mapping.get(record["name"]):
            raise ValueError("CSV input and fixed manifest sequence identity mismatch")
    train = frame.loc[frame.split.eq("train")]
    nb, paired = train.loc[train.light.eq("")], train.loc[train.light.ne("")]
    seeds = (_select_stratum(nb, "heavy", 80, "nanobody_heavy")
             + _select_stratum(paired, "heavy", 60, "paired_heavy")
             + _select_stratum(paired, "light", 60, "paired_light"))
    write(root / "selected_seeds.json", seeds)
    with (root / "selected_seeds.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(seeds[0]))
        writer.writeheader()
        writer.writerows(seeds)
    return seeds


def convert_alignment(sto, dest, sequence, cap):
    # Stockholm uses both '-' and '.' as gaps. Treat query insert columns
    # explicitly; the legacy converter handled '-' only.
    from prepare_compact_db_msa_cache import _read_stockholm
    alignment = _read_stockholm(sto)
    if "query" not in alignment:
        raise ValueError("Stockholm query row missing")
    query = alignment["query"]
    matches = [c not in "-." for c in query]
    if "".join(c for c in query if c not in "-.").upper() != sequence:
        raise ValueError("Stockholm query sequence mismatch")
    selected = ["query"] + [n for n in alignment if n != "query"][:cap]
    with dest.open("w", encoding="ascii") as h:
        for name in selected:
            row = alignment[name]
            if len(row) != len(query):
                raise ValueError("Stockholm alignment width mismatch")
            converted = "".join(("-" if c in "-." else c.upper()) if match
                                else ("" if c in "-." else c.lower())
                                for c, match in zip(row, matches))
            h.write(f">{name}\n{converted}\n")
    return len(selected)


def search(sequences, database, cache, executable, cpu):
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {}
    unique = list(dict.fromkeys(sequences))
    for number, sequence in enumerate(unique, 1):
        ident = key(sequence)
        entry = manifest.get(ident)
        if entry and Path(entry["a3m"]).exists() and digest(entry["a3m"]) == entry["sha256"]:
            continue
        request = cache / ident
        request.mkdir(exist_ok=True)
        query, sto, a3m = request / "query.fasta", request / "alignment.sto", request / "hits.a3m"
        query.write_text(f">query\n{sequence}\n", encoding="ascii")
        command = [executable, "--cpu", str(cpu), "--seed", "42", "-N", "1",
                   "-E", "0.001", "--incE", "0.001", "--noali", "-A", str(sto),
                   str(query), str(database)]
        print(f"{cache.name}: search {number}/{len(unique)} {ident[:12]}", flush=True)
        started = time.perf_counter()
        with (request / "jackhmmer.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - started
        with (cache / "attempts.jsonl").open("a") as log:
            log.write(json.dumps(dict(query=ident, seconds=seconds, returncode=result.returncode)) + "\n")
        if result.returncode:
            raise RuntimeError(f"jackhmmer exit {result.returncode}; see {request / 'jackhmmer.log'}")
        started = time.perf_counter()
        part = request / "hits.a3m.part"
        depth = convert_alignment(sto, part, sequence, 5000)
        part.replace(a3m)
        manifest[ident] = dict(sequence=sequence, a3m=str(a3m), sha256=digest(a3m), depth=depth,
                               search_seconds=seconds, conversion_seconds=time.perf_counter() - started)
        write(manifest_path, manifest)
        sto.unlink()
        print(f"  {seconds:.1f}s; {depth} alignment rows", flush=True)
    return manifest


def build_compact(seeds, cache, root):
    from build_compact_nb_msa_database import _read_fasta, _ungapped_sequence
    target, report = root / "compact.fasta", root / "compact.json"
    if report.exists():
        saved = read(report)
        if digest(target) != saved["sha256"]:
            raise ValueError("Compact FASTA differs from its audit")
        return target, saved
    started = time.perf_counter()
    database = OrderedDict()
    per_seed = []
    for seed in seeds:
        entry = cache[key(seed["sequence"])]
        before = len(database)
        # Retain query + at most 5000 homologous aligned regions per seed.
        for _, aligned in _read_fasta(Path(entry["a3m"])):
            seq = _ungapped_sequence(aligned)
            if seq:
                database.setdefault(seq, len(database))
        per_seed.append(dict(stratum=seed["stratum"], query=key(seed["sequence"]),
                             new_sequences=len(database) - before))
    with target.with_suffix(".part").open("w") as h:
        for sequence, i in database.items():
            h.write(f">compact_{i}\n{sequence}\n")
    target.with_suffix(".part").replace(target)
    saved = dict(sequences=len(database), residues=sum(map(len, database)), sha256=digest(target),
                 aggregation_seconds=time.perf_counter() - started, seed_rows=len(seeds),
                 unique_seed_queries=len({r["sequence"] for r in seeds}), per_seed=per_seed,
                 source="Training-selected UniRef50 MSAs; ungapped aligned regions plus seed queries",
                 selection="80 Nb-heavy / 60 paired-heavy / 60 paired-light; length + 3-mer diversity")
    write(report, saved)
    return target, saved


def export_inputs(records, manifest, path):
    from build_compact_nb_msa_database import _read_fasta
    ready = copy.deepcopy(records)
    for record in ready:
        for item in record["sequences"]:
            chain = item["proteinChain"]
            entry = manifest[key(chain["sequence"])]
            a3m = Path(entry["a3m"]).with_name("input_depth4096.a3m")
            with a3m.open("w") as h:
                for i, (name, sequence) in enumerate(_read_fasta(Path(entry["a3m"]))):
                    if i >= 4097:
                        break
                    h.write(f">{name}\n{sequence}\n")
            chain["unpairedMsaPath"] = str(a3m)
            # Do not write ``pairedMsaPath`` at all.  Protenix interprets an
            # empty path as a missing paired MSA and silently falls back to its
            # online MSA service.  These independent per-chain searches supply
            # only the unpaired alignment.
            chain.pop("pairedMsaPath", None)
    write(path, ready)


def feature_check(folder, records):
    index = folder / "index.json"
    if not index.exists():
        return False
    rows = read(index)
    return (len(rows) == len(records)
            and {r["sample_id"] for r in rows} == {r["name"] for r in records}
            and all((folder / r["file"]).is_file() for r in rows))


def verify_unpaired_only_input(path):
    """Fail before extraction if a manifest could trigger online MSA fallback."""
    for record in read(path):
        for item in record["sequences"]:
            chain = item["proteinChain"]
            if "pairedMsaPath" in chain:
                raise RuntimeError(
                    f"{path} contains pairedMsaPath; refusing a possible online MSA fallback"
                )
            a3m = Path(chain.get("unpairedMsaPath", ""))
            if not a3m.is_file() or a3m.stat().st_size == 0:
                raise FileNotFoundError(f"Missing local unpaired MSA: {a3m}")


def local_feature_provenance_path(folder):
    return folder / "local_unpaired_only_provenance.json"


def quarantine_unverified_features(folder, records):
    """Keep, rather than reuse, any outputs begun by the Stage-77 fallback."""
    if not folder.exists() or (feature_check(folder, records) and local_feature_provenance_path(folder).is_file()):
        return
    destination = folder.with_name(folder.name + "_stage77_remote_fallback_quarantined")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite quarantined features: {destination}")
    folder.rename(destination)
    print(f"Quarantined unverified feature output: {destination}", flush=True)


def training_arguments(data, split, esm, features, output, seed, source):
    return ["--data-csv", str(data), "--label-column", "label", "--split-manifest", str(split),
            "--protenix-features-dir", str(features), "--esm-features-dir", str(esm),
            "--output-dir", str(output), "--seed", str(seed), "--protenix-msa-source", source,
            "--fusion-normalization", "branch_only", "--paircond-mode", "additive",
            "--structure-readout", "s_z", "--ema-decay", "0.995", "--esm-context-dropout", "0",
            "--max-steps", "5000", "--learning-rate", "0.0001", "--weight-decay", "0.01",
            "--gradient-accumulation", "3", "--warmup-steps", "100", "--validate-every", "100",
            "--scheduler-patience", "50", "--early-stopping-patience", "0",
            "--pairformer-dropout", "0.1", "--pairformer-n-heads", "16",
            "--pairformer-c-hidden-mul", "128", "--pairformer-c-hidden-pair-att", "32",
            "--pairformer-no-heads-pair", "4"]


def summary(root, run_paths, timing):
    rows = []
    for seed in range(42, 48):
        if not all((run_paths[m][seed] / "metrics.json").exists() for m in run_paths):
            continue
        metrics = {m: read(run_paths[m][seed] / "metrics.json")["metrics"]["test"] for m in run_paths}
        rows.append(dict(seed=seed, regular=metrics["regular"], compact=metrics["compact"]))
    lines = ["# Tm local UniRef50 versus training-seeded compact MSA", "",
             "Fixed 340/30/63 split; EMA=0.995; checkpoints selected by validation normalized MSE.",
             "Database version 2026_02. This is a local UniRef50 comparison, not the full multi-database AF3 benchmark.", "",
             "| Seed | Regular rho | Compact rho | Delta rho | Regular MAE (C) | Compact MAE (C) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        a, b = row["regular"], row["compact"]
        lines.append(f"| {row['seed']} | {a['spearman_rho']:.4f} | {b['spearman_rho']:.4f} | {b['spearman_rho']-a['spearman_rho']:+.4f} | {a['mae']:.3f} | {b['mae']:.3f} |")
    aggregates = {}
    if rows:
        lines += ["", "| Method | rho mean | rho sample std | MAE mean (C) |", "| --- | ---: | ---: | ---: |"]
        for method in ("regular", "compact"):
            values = [r[method]["spearman_rho"] for r in rows]
            aggregates[method] = dict(rho_mean=statistics.mean(values),
                                      rho_std=statistics.stdev(values) if len(values) > 1 else None,
                                      mae_mean=statistics.mean(r[method]["mae"] for r in rows))
            std = f"{aggregates[method]['rho_std']:.5f}" if len(values) > 1 else "NA"
            lines.append(f"| {method} | {statistics.mean(values):.5f} | {std} | {aggregates[method]['mae_mean']:.4f} |")
    lines += ["", f"Completed paired training seeds: {len(rows)}/6.", "", "## Local timing", "",
              f"Unique target chains per method: {timing['target_queries']}.",
              f"Regular successful search wall time: {timing['regular_search_seconds']:.3f} s.",
              f"Compact successful search wall time: {timing['compact_search_seconds']:.3f} s.",
              f"Search-only ratio: {timing['search_ratio']:.3f}x.",
              f"Compact construction (selected-seed search + conversion + aggregation): {timing['build_seconds']:.3f} s.",
              f"Cold compact route (construction + all local compact searches/conversions): {timing['compact_cold_seconds']:.3f} s.",
              f"Regular route (all searches/conversions): {timing['regular_route_seconds']:.3f} s.",
              "Search times include local database I/O and process runtime; exclude remote queues, A3M conversion, features and training.",
              "One serial timing sweep per database; OS cache state is uncontrolled. Ratios are descriptive, not repeated benchmarks.",
              "Full and compact use identical jackhmmer flags, including native database-size-dependent E-values.",
              "Seed queries reuse their ordinary-search results; their construction cost is counted once in the cold compact route."]
    write(root / "summary.json", dict(rows=rows, aggregates=aggregates, timing=timing))
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/foldabs-work"))
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--phase", choices=("msa", "all"), default="all")
    args = parser.parse_args()
    if args.cpu < 1:
        raise ValueError("--cpu must be positive")
    base = args.data_root.resolve()
    root = base / "msa_accelerated" / TAG
    root.mkdir(parents=True, exist_ok=True)
    # flock is released by the OS on exit, including an interrupted process.
    import fcntl
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = base / "official_test_aligned_public_tm/data/foldabs_public_tm_official_test_aligned.csv"
    split = base / "official_test_aligned_public_tm/splits/official_test_public_trainval_manifest.csv"
    esm = base / "features/esm2_650m_public_tm_official_test_aligned"
    archive = base / "databases/uniref50_ddbj/uniref50.fasta.gz"
    for path in (data, split, esm / "index.json", archive):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing artifact missing: {path}")
    executable = shutil.which(os.environ.get("FOLDABS_JACKHMMER", "jackhmmer"))
    if not executable:
        raise FileNotFoundError("jackhmmer missing in activated foldabs environment")
    version = subprocess.run([executable, "-h"], check=True, capture_output=True, text=True).stdout
    protocol = dict(tag=TAG, data_sha256=digest(data), split_sha256=digest(split),
                    esm_index_sha256=digest(esm / "index.json"), parent_md5=EXPECTED_MD5,
                    cpu=args.cpu, jackhmmer=executable, jackhmmer_help=version,
                    search=dict(iterations=1, e_value=.001, inc_e_value=.001, seed=42,
                                retained_nonquery_hits=5000, input_nonquery_hits=4096,
                                evalue_database_size="native for each database"),
                    training_seeds=list(range(42, 48)), ema=.995, paircond="additive",
                    code_sha256={name: digest(SCRIPTS / name) for name in (
                        Path(__file__).name, "train_foldabs_developability.py", "extract_protenix_embeddings.py",
                        "make_developability_protenix_input.py", "audit_train_seeded_antibody_compact_capacity.py",
                        "build_compact_nb_msa_database.py", "prepare_compact_db_msa_cache.py")})
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        # Stage 78 is a narrowly-scoped manifest correction: its only code
        # difference is this runner, which no longer serializes an empty
        # pairedMsaPath.  Preserve completed local searches, but refuse every
        # migration that changes data, MSA, feature, or training configuration.
        previous = read(protocol_path)
        previous_without_runner = copy.deepcopy(previous)
        current_without_runner = copy.deepcopy(protocol)
        runner_name = Path(__file__).name
        old_hash = previous_without_runner.get("code_sha256", {}).pop(runner_name, None)
        new_hash = current_without_runner.get("code_sha256", {}).pop(runner_name, None)
        if not old_hash or not new_hash or previous_without_runner != current_without_runner:
            raise RuntimeError("Protocol/code changed since this run; inspect protocol.json before resuming.")
        write(root / "paired_msa_manifest_fix.json", {
            "scope": "Stage-78 input-manifest correction only",
            "previous_runner_sha256": old_hash,
            "current_runner_sha256": new_hash,
            "change": "omit empty pairedMsaPath so Protenix cannot fall back to ColabFold",
            "preserved": "all local UniRef50 and compact-MSA search caches",
        })
    write(protocol_path, protocol)
    raw = root / "raw.json"
    run("make_developability_protenix_input.py", "--input-csv", data, "--label-column", "label", "--output-json", raw)
    records = read(raw)
    seeds = select_seeds(split, records, root)
    all_sequences = list(dict.fromkeys(i["proteinChain"]["sequence"] for r in records for i in r["sequences"]))
    seed_sequences = list(dict.fromkeys(s["sequence"] for s in seeds))
    if not set(seed_sequences).issubset(all_sequences):
        raise ValueError("Selected chain is absent from target inputs")
    parent, parent_audit = prepare_parent(archive, root)
    # Run seeds first to obtain the compact library early. Each parent search
    # is performed once and reused in the ordinary method's 610-chain cache.
    regular_cache = search(seed_sequences, parent, root / "regular_cache", executable, args.cpu)
    compact, compact_audit = build_compact(seeds, regular_cache, root)
    compact_cache = search(all_sequences, compact, root / "compact_cache", executable, args.cpu)
    regular_cache = search(all_sequences, parent, root / "regular_cache", executable, args.cpu)
    def total(cache, sequences, field):
        return sum(cache[key(s)][field] for s in sequences)
    rs = total(regular_cache, all_sequences, "search_seconds")
    cs = total(compact_cache, all_sequences, "search_seconds")
    rc = total(regular_cache, all_sequences, "conversion_seconds")
    cc = total(compact_cache, all_sequences, "conversion_seconds")
    build = (total(regular_cache, seed_sequences, "search_seconds")
             + total(regular_cache, seed_sequences, "conversion_seconds") + compact_audit["aggregation_seconds"])
    timing = dict(target_queries=len(all_sequences), seed_queries=len(seed_sequences),
                  regular_search_seconds=rs, compact_search_seconds=cs, search_ratio=rs / cs,
                  regular_conversion_seconds=rc, compact_conversion_seconds=cc, build_seconds=build,
                  compact_cold_seconds=build + cs + cc, regular_route_seconds=rs + rc,
                  parent_preparation=parent_audit, compact_database=compact_audit)
    timing["recorded_attempts"] = {}
    for mode in ("regular", "compact"):
        attempts = [json.loads(line) for line in (root / f"{mode}_cache/attempts.jsonl").read_text().splitlines() if line]
        timing["recorded_attempts"][mode] = dict(count=len(attempts),
            seconds=sum(a["seconds"] for a in attempts),
            failed_seconds=sum(a["seconds"] for a in attempts if a["returncode"] != 0))
    write(root / "timing.json", timing)
    paths = {mode: {seed: base / "runs" / f"{TAG}_{mode}_ema995_seed{seed}" for seed in range(42, 48)}
             for mode in ("regular", "compact")}
    summary(root, paths, timing)
    for mode, cache in (("regular", regular_cache), ("compact", compact_cache)):
        export_inputs(records, cache, root / f"{mode}_msa.json")
    if args.phase == "msa":
        print(f"CPU MSA phase complete: {root / 'summary.md'}", flush=True)
        return
    for mode in ("regular", "compact"):
        features = base / "features" / f"protenix_v1_{TAG}_{mode}_c10"
        if not (feature_check(features, records) and local_feature_provenance_path(features).is_file()):
            quarantine_unverified_features(features, records)
            verify_unpaired_only_input(root / f"{mode}_msa.json")
            run("extract_protenix_embeddings.py", "--input-json", root / f"{mode}_msa.json",
                "--output-dir", features, "--cycles", 10, "--limit", 0,
                "--use-msa", "--msa-server-mode", "protenix", "--resume")
        if not feature_check(features, records):
            raise RuntimeError(f"Incomplete 433-row feature index: {features}")
        if not local_feature_provenance_path(features).is_file():
            write(local_feature_provenance_path(features), {
                "msa_input": "local exported unpairedMsaPath only",
                "pairedMsaPath_present": False,
                "remote_msa_fallback_permitted": False,
                "stage": "stage78",
            })
    for seed in range(42, 48):
        for mode in ("regular", "compact"):
            features = base / "features" / f"protenix_v1_{TAG}_{mode}_c10"
            output = paths[mode][seed]
            if (output / "metrics.json").exists() and not (output / "best_model.pt").exists():
                raise RuntimeError(f"Metrics exist but checkpoint is missing: {output}")
            if not (output / "metrics.json").exists():
                print(f"Training {mode}, seed={seed}, EMA=.995", flush=True)
                run("train_foldabs_developability.py", *training_arguments(
                    data, split, esm, features, output, seed, f"local_uniref50_2026_02_{mode}"))
        summary(root, paths, timing)
    print(f"SIX-SEED COMPARISON COMPLETE: {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
