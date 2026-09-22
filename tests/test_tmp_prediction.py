import json

import pandas as pd
import pytest

from foldabs_repro.tmp_prediction.data import TMPDatasetSchema, canonical_hla, load_tmp_csv
from foldabs_repro.tmp_prediction.input_builder import build_protenix_inputs
from foldabs_repro.tmp_prediction.metrics import macro_pmhc_metrics
from foldabs_repro.tmp_prediction.negative_sampling import build_pseudo_negatives


def test_hla_canonicalization():
    assert canonical_hla("A0201") == "HLA-A*02:01"
    assert canonical_hla("HLA-B*40:01") == "HLA-B*40:01"


def test_load_and_build_tmp_input(tmp_path):
    path = tmp_path / "tmp.csv"
    pd.DataFrame(
        {
            "ID": ["pair-1"],
            "TCRa": [" CASSL "],
            "TCRb": ["CASSP"],
            "Peptide": ["GILGFVFTL"],
            "HLA": ["A0201"],
            "label": [1],
            "split": ["train"],
        }
    ).to_csv(path, index=False)
    frame = load_tmp_csv(path, require_label=True, require_split=True)
    records, manifest = build_protenix_inputs(frame, {"HLA-A*02:01": "GSHSMRYFYTAMSRPGR"})
    assert records[0]["name"] == "pair-1"
    assert len(records[0]["sequences"]) == 4
    assert json.loads(manifest.loc[0, "entity_roles"]) == ["tcra", "tcrb", "peptide", "mhc"]
    assert manifest.loc[0, "pmhc_id"] == "GILGFVFTL|HLA-A*02:01"


def test_missing_alpha_is_rejected(tmp_path):
    path = tmp_path / "tmp.csv"
    pd.DataFrame(
        {"TCRa": [""], "TCRb": ["CASSP"], "Peptide": ["GILGFVFTL"], "HLA": ["A0201"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="TCRa is empty"):
        load_tmp_csv(path)


def test_macro_pmhc_skips_single_class_groups():
    frame = pd.DataFrame(
        {
            "pmhc_id": ["p1", "p1", "p2", "p2"],
            "label": [0, 1, 1, 1],
            "prediction": [0.1, 0.9, 0.2, 0.3],
        }
    )
    result = macro_pmhc_metrics(frame)
    assert result["eligible_pmhc"] == 1
    assert result["macro_auroc"] == 1.0


def test_negatives_stay_inside_split_and_exclude_known_positives():
    positives = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "original_id": ["a", "b", "c", "d"],
            "tcra": ["AAA", "CCC", "DDD", "EEE"],
            "tcrb": ["FFF", "GGG", "HHH", "III"],
            "peptide": ["PEPTIDEA", "PEPTIDEB", "PEPTIDEA", "PEPTIDEB"],
            "hla": ["HLA-A*02:01"] * 4,
            "pmhc_id": ["PEPTIDEA|HLA-A*02:01", "PEPTIDEB|HLA-A*02:01"] * 2,
            "split": ["train", "train", "test", "test"],
        }
    )
    sampled = build_pseudo_negatives(positives, seed=1)
    assert set(sampled["split"]) == {"train", "test"}
    assert {"same_hla_cross_epitope", "tcr_swap"} <= set(sampled["negative_type"])
    known = set(zip(positives.tcra, positives.tcrb, positives.peptide))
    generated = sampled[sampled.label.eq(0)]
    assert not any(key in known for key in zip(generated.tcra, generated.tcrb, generated.peptide))
