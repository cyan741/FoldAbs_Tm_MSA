import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_immrep25_subset_training import prepare
from negative_sampling.peptide_mismatch import COLUMNS
from train_tmp_specificity import PeptideMismatchSampling


def make_sources(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for split in ("train", "val", "test"):
        rows = [dict(tcra_v=f"alpha_{split}_{index}", tcrb_v=f"beta_{split}_{index}",
                     peptide=f"pep_{split}_{index}", hla_alleles_raw="HLA-A*02:01",
                     hla_alleles="HLA-A*02", hla_clip="MHC", label=1, neg_count=2)
                for index in range(4)]
        pd.DataFrame(rows, columns=COLUMNS).to_csv(source_dir / f"subset_{split}.csv", index=False)
    known = source_dir / "known.csv"
    pd.DataFrame([dict(tcra_v="alpha_train_0", tcrb_v="beta_train_0", peptide="pep_train_1")]).to_csv(known, index=False)
    return source_dir, known


def test_prepare_keeps_splits_and_excludes_additional_known_positives(tmp_path):
    source, known = make_sources(tmp_path)
    output = tmp_path / "prepared"
    report = prepare(source, "subset", known, output)
    assert report["negative_ratio"] == 2
    pd.testing.assert_frame_equal(pd.read_csv(source / "subset_train.csv"), pd.read_csv(output / "train.csv"))
    sampling = PeptideMismatchSampling(SimpleNamespace(
        splits_dir=output, negatives_per_positive=2, negative_sampling_seed=42))
    for split in ("val", "test"):
        assert sampling.frames[split].label.value_counts().to_dict() == {0: 8, 1: 4}
    for epoch in (1, 2, 10):
        frame, _ = sampling.epoch(epoch)
        assert frame.label.value_counts().to_dict() == {0: 8, 1: 4}
        assert not ((frame.tcra == "alpha_train_0") & (frame.peptide == "pep_train_1")).any()
    with pytest.raises(FileExistsError):
        prepare(source, "subset", known, output)
    known_frame = pd.read_csv(output / "known_positives.csv")
    known_frame.loc[0, "peptide"] = "different_peptide"
    known_frame.to_csv(output / "known_positives.csv", index=False)
    with pytest.raises(ValueError, match="Blacklist"):
        PeptideMismatchSampling(SimpleNamespace(
            splits_dir=output, negatives_per_positive=2, negative_sampling_seed=42))


def test_prepare_rejects_cross_split_peptides(tmp_path):
    source, known = make_sources(tmp_path)
    validation = pd.read_csv(source / "subset_val.csv")
    validation.loc[0, "peptide"] = "pep_train_0"
    validation.to_csv(source / "subset_val.csv", index=False)
    with pytest.raises(ValueError, match="peptide-disjoint"):
        prepare(source, "subset", known, tmp_path / "prepared")
