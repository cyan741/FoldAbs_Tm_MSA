#!/usr/bin/env python3
"""Training-sequence-only, stratified farthest-first coverage exploration.

No MSA cache, labels, model, network or database is read. Thresholds describe
sequence geometry, not homolog retention or a model-performance guarantee.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

STRATA = {"nanobody_heavy": 80, "paired_heavy": 60, "paired_light": 60}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def kmers(sequence):
    return frozenset(sequence[i:i + 3] for i in range(len(sequence) - 2)) if len(sequence) >= 3 else frozenset({sequence})


def distance(a, b, words, span):
    union = words[a] | words[b]
    jaccard = 1 - len(words[a] & words[b]) / len(union) if union else 0
    return 0.8 * jaccard + 0.2 * abs(len(a) - len(b)) / max(span, 1)


def read_strata(path):
    groups = {name: set() for name in STRATA}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"split", "heavy", "light"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires split, heavy, light")
        for row in reader:
            if row["split"] not in ("train", "validation", "test"):
                raise ValueError("Unknown split")
            if row["split"] != "train":
                continue
            values = []
            for name in ("heavy", "light"):
                seq = (row[name] or "").strip().upper()
                values.append("" if seq in ("NAN", "NONE", "NULL") else seq)
            heavy, light = values
            if not heavy or any(not set(s) <= AA for s in values):
                raise ValueError("Training rows require a canonical heavy chain and optional canonical light chain")
            groups["paired_heavy" if light else "nanobody_heavy"].add(heavy)
            if light:
                groups["paired_light"].add(light)
    if not any(groups.values()):
        raise ValueError("No training chains")
    return {name: sorted(seqs) for name, seqs in groups.items()}


def traversal(sequences):
    """Preserve original shortest/longest initialization and lexical ties.

    The full order is cheap to compute from sequences. Every threshold uses a
    prefix of this order; computing it does NOT run MSA for those candidates.
    """
    sequences = sorted(set(sequences))
    if not sequences:
        return [], []
    words = {s: kmers(s) for s in sequences}
    span = max(map(len, sequences)) - min(map(len, sequences))
    initial = list(dict.fromkeys((min(sequences, key=lambda s: (len(s), s)),
                                 max(sequences, key=lambda s: (len(s), s)))))
    remaining, selected, curve = set(sequences), [], []
    nearest = {s: float("inf") for s in sequences}
    while remaining:
        seed = initial[len(selected)] if len(selected) < len(initial) else max(remaining, key=lambda s: (nearest[s], s))
        selected.append(seed)
        remaining.remove(seed)
        for s in sequences:
            nearest[s] = min(nearest[s], distance(s, seed, words, span))
        curve.append({"seed_count": len(selected), "radius": max(nearest.values()),
                      "mean_nearest_distance": sum(nearest.values()) / len(sequences),
                      "initialization_complete": len(selected) >= len(initial)})
    return selected, curve


def stop_count(curve, radius):
    if not math.isfinite(radius) or not 0 <= radius <= 1:
        raise ValueError("Radius must be finite and in [0,1]")
    if not curve:
        return 0
    return next(row["seed_count"] for row in curve if row["initialization_complete"] and row["radius"] <= radius)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(manifest, output, radii):
    radii = sorted(set(radii))
    if not radii or any(not math.isfinite(r) or not 0 <= r <= 1 for r in radii):
        raise ValueError("Provide finite radius thresholds in [0,1]")
    groups = read_strata(manifest)
    # Never overwrite an earlier exploration or a selected library.
    output.mkdir(parents=True, exist_ok=False)
    paths = {name: traversal(seqs) for name, seqs in groups.items()}
    curve_rows = [dict(stratum=name, **row) for name, (_, curve) in paths.items() for row in curve]
    write_csv(output / "coverage_curve.csv", curve_rows,
              ["stratum", "seed_count", "radius", "mean_nearest_distance", "initialization_complete"])
    baseline = []
    for name, budget in STRATA.items():
        _, curve = paths[name]
        n = min(budget, len(curve))
        baseline.append({"stratum": name, "requested_count": budget, "actual_count": n,
                         "radius": curve[n - 1]["radius"] if n else None,
                         "exact_original_budget_available": n == budget})
    summaries = []
    for radius in radii:
        folder = output / ("radius_" + str(radius))
        folder.mkdir()
        rows, details = [], []
        for name, (order, curve) in paths.items():
            n = stop_count(curve, radius)
            rows.extend({"stratum": name, "sequence": s, "length": len(s)} for s in order[:n])
            details.append({"stratum": name, "candidate_count": len(order), "seed_count": n,
                            "achieved_radius": curve[n - 1]["radius"] if n else None})
        for index, row in enumerate(rows):
            row["seed_index"] = index
        (folder / "selected_seeds.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        write_csv(folder / "selected_seeds.csv", rows, ["seed_index", "stratum", "sequence", "length"])
        summary = {"radius_threshold": radius, "strata": details, "seed_rows": len(rows),
                   "unique_seed_queries": len({row["sequence"] for row in rows})}
        summaries.append(summary)
        (folder / "selection_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = {"scope": "sequence-only exploration; no library construction or MSA search",
              "k": 3, "jaccard_weight": 0.8, "length_weight": 0.2,
              "length_normalization": "fixed max-minus-min training length within each stratum",
              "initialization": "shortest and longest; same as original selector",
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "uses_msa_or_labels_for_selection": False,
              "training_unique_chains": len(set().union(*map(set, groups.values()))),
              "original_budget_sequence_coverage": baseline, "exploratory_summaries": summaries,
              "boundary": "No threshold is selected as optimal. Not a proven minimum set, homolog-retention guarantee, or performance guarantee."}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Tm sequence-only adaptive seed exploration", "",
             "Uses training sequences only. No MSA cache, labels, features or model results are used.", "",
             "Fixed k=3 and distance=0.8*Jaccard distance + 0.2*normalized length difference.",
             "Stop separately when every chain has a same-stratum seed within the radius threshold.", "",
             "| Radius threshold | Nb heavy | Paired heavy | Paired light | Unique queries |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        counts = [str(s["seed_count"]) for s in item["strata"]]
        lines.append("| " + " | ".join([str(item["radius_threshold"]), *counts, str(item["unique_seed_queries"])]) + " |")
    lines += ["", "## Original 80/60/60 budget: sequence-coverage reference", "",
              "| Stratum | Selected | Achieved radius |", "| --- | ---: | ---: |"]
    for row in baseline:
        lines.append(f"| {row['stratum']} | {row['actual_count']} | {row['radius']} |")
    lines += ["", "Thresholds are exploratory, not MSA-hit percentages. Smaller thresholds require at least as many seeds.",
              "A geometric coverage criterion does not establish biological sufficiency or unchanged downstream performance.",
              "This retains the original length-extreme initialization; it is not a minimum-cardinality optimizer.",
              "Seed lists can be obtained without any candidate MSA. This run does not build libraries or launch searches."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radii", default="0.15,0.20,0.25,0.30,0.35,0.40",
                        help="Exploratory geometry thresholds; no preferred default winner")
    args = parser.parse_args()
    run(args.split_manifest, args.output_dir, [float(s) for s in args.radii.split(",")])
    print("Sequence-only selection summary:", args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
