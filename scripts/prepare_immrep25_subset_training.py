"""Adapt delivered positive-only splits without changing their membership."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from negative_sampling.peptide_mismatch import COLUMNS, build_blacklist, build_peptide_pool, sample_epoch
from train_tmp_specificity import PeptideMismatchSampling


def prepare(input_dir, prefix, known_positives, output_dir, seed=42):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared data: {output_dir}")
    frames = {}
    identities = set()
    peptides = set()
    sources = []
    source_hashes = {}
    identity_columns = ["tcra_v", "tcrb_v", "peptide", "hla_clip"]
    for split in ("train", "val", "test"):
        path = input_dir / f"{prefix}_{split}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        if list(frame.columns) != COLUMNS or not frame.label.eq(1).all():
            raise ValueError(f"Expected positive-only delivered schema: {path}")
        if frame.empty or frame[identity_columns].eq("").any().any():
            raise ValueError(f"Empty split or sequence: {path}")
        current = set(frame[identity_columns].itertuples(index=False, name=None))
        if len(current) != len(frame) or identities & current:
            raise ValueError("Duplicate four-chain inputs within/across splits")
        if peptides & set(frame.peptide):
            raise ValueError("Expected peptide-disjoint IMMREP-style splits")
        identities.update(current)
        peptides.update(frame.peptide)
        frames[split] = frame
        source = frame.copy()
        source["record_id"] = [f"cap50_{split}_{index:06d}" for index in range(len(frame))]
        source["split"] = split
        sources.append(source)
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = pd.concat(sources, ignore_index=True)
    known = pd.read_csv(known_positives, keep_default_na=False)
    if "label" in known and not known.label.eq(1).all():
        raise ValueError("Known-positive source must contain only positives")
    known = known[["tcra_v", "tcrb_v", "peptide"]].drop_duplicates()
    if known.eq("").any().any():
        raise ValueError("Empty known-positive sequence")
    blacklist = build_blacklist(pd.concat([source, known], ignore_index=True))
    outputs = {}
    reports = {}
    pools = {}
    for split, frame in frames.items():
        pools[split] = build_peptide_pool(frame)
        sampled, report = sample_epoch(frame, pools[split], blacklist, seed=seed, epoch=1, n=2)
        outputs[split] = frame if split == "train" else sampled
        reports[split] = {**report, "hla_positive_counts": frame.hla_alleles.value_counts().to_dict(),
                          "unique_peptides": frame.peptide.nunique()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in frames:
        outputs[split].to_csv(output_dir / f"{split}.csv", index=False)
        np.save(output_dir / f"{split}_pep2candidates.npy", pools[split], allow_pickle=True)
    source.to_csv(output_dir / "positive_splits.csv", index=False)
    known.to_csv(output_dir / "known_positives.csv", index=False)
    np.save(output_dir / "blacklilst.npy", blacklist, allow_pickle=True)
    sampling = PeptideMismatchSampling(SimpleNamespace(
        splits_dir=output_dir, negatives_per_positive=2, negative_sampling_seed=seed))
    for epoch in (1, 2, 10):
        manifest, _ = sampling.epoch(epoch)
        assert manifest.label.eq(0).sum() == 2 * manifest.label.eq(1).sum()
    report = {"source_sha256": source_hashes,
              "known_positive_source_sha256": hashlib.sha256(Path(known_positives).read_bytes()).hexdigest(),
              "seed": seed, "negative_ratio": 2, "peptide_disjoint": True,
              "train_negatives": "resampled_each_epoch", "evaluation_negatives": "fixed",
              "splits": reports}
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="immrep25_subset_cdr3_cap50")
    parser.add_argument("--known-positives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.prefix, args.known_positives,
                             args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
