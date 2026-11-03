"""CPU regression checks for epoch boundaries and test-only exports."""

import json
import math
import sys

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("protenix")

from scripts import predict_tmp_specificity as inference
from scripts import train_tmp_specificity as training
from scripts import run_tmp_pooling_ablation as ablation
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics


def make_inputs(tmp_path):
    """Create tiny frozen features with a short final accumulation group."""
    features = tmp_path / "features"
    features.mkdir()
    rows = []
    for split, count in (("train", 5), ("validation", 2), ("test", 2)):
        for index in range(count):
            name = f"{split}_{index}"
            rows.append(dict(sample_id=name, original_id=name, pmhc_id="p1",
                             label=index % 2, split=split))
            torch.save({"sample_name": name,
                        "token_asym_id": torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]),
                        "s": torch.randn(8, 6), "z": torch.randn(8, 8, 4),
                        "p_dist": torch.softmax(torch.randn(8, 8, 3), dim=-1)},
                       features / f"{name}.pt")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest, features


@pytest.mark.parametrize("epochs,expected_epochs", [(None, 2), (1, 1)])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_epoch_validation_and_test_only_outputs(tmp_path, monkeypatch, epochs, expected_epochs, precision):
    """Validation follows full epochs; the final stage evaluates only test."""
    torch.set_num_threads(1)
    training.set_seed(42)
    manifest, features = make_inputs(tmp_path)
    output = tmp_path / "run"
    argv = ["train", "--manifest", str(manifest), "--features-dir", str(features),
            "--output-dir", str(output), "--max-steps", "3",
            "--gradient-accumulation", "3", "--pairformer-blocks", "0",
            "--hidden-dim", "8", "--dropout", "0", "--log-every", "1",
            "--precision", precision]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    gradient_scales = []
    original_evaluate = training.evaluate
    original_loss = training.F.binary_cross_entropy_with_logits

    def track_loss(*args, **kwargs):
        loss = original_loss(*args, **kwargs)
        if loss.requires_grad:
            loss.register_hook(lambda gradient: gradient_scales.append(float(gradient)))
        return loss

    def track_evaluation(model, loader, *args):
        calls.append([row.sample_id for row in loader.dataset.examples])
        return original_evaluate(model, loader, *args)

    monkeypatch.setattr(training, "evaluate", track_evaluation)
    monkeypatch.setattr(training.F, "binary_cross_entropy_with_logits", track_loss)
    training.run_training(0, 1, training.parse_args(), [], torch.device("cpu"))
    assert gradient_scales == pytest.approx(([1 / 3] * 3 + [1 / 2] * 2) * expected_epochs)
    assert calls == [["validation_0", "validation_1"]] * expected_epochs + [["test_0", "test_1"]]
    log = (output / "train.log").read_text()
    assert "{" not in log and "train_bce_moving" not in log
    assert log.count("complete |") == expected_epochs
    assert f"Step {expected_epochs * 2}" in log
    predictions = pd.read_csv(output / "predictions.csv")
    assert list(predictions.sample_id) == ["test_0", "test_1"]
    assert set(predictions.split) == {"test"}
    assert set(predictions.prediction).issubset({0, 1})
    assert list(predictions.prediction) == list((predictions.logit >= 0).astype(int))
    metrics = json.loads((output / "metrics.json").read_text())
    assert "train" not in metrics
    for split in ("validation", "test"):
        assert "per_pmhc" not in metrics[split]["macro_pmhc"]
    config = json.loads((output / "run_config.json").read_text())
    assert not any("sha" in key for key in config)
    checkpoint = torch.load(output / metrics["best_checkpoint"], weights_only=True)
    assert checkpoint["step"] == metrics["best_step"]
    assert checkpoint["epoch"] == metrics["best_epoch"]

    # Standalone inference also defaults to test, never all manifest rows.
    inferred = tmp_path / "inference.csv"
    monkeypatch.setattr(sys, "argv", ["predict", "--manifest", str(manifest),
                         "--features-dir", str(features), "--checkpoint",
                         str(output / metrics["best_checkpoint"]), "--output-csv", str(inferred),
                         "--device", "cpu"])
    inference.main()
    restored = pd.read_csv(inferred)
    assert list(restored.sample_id) == list(predictions.sample_id)
    assert list(restored.prediction) == list(predictions.prediction)
    assert len(list(output.glob("*.pt"))) == 2 + expected_epochs
    for epoch in range(1, expected_epochs + 1):
        archived = torch.load(output / f"bs3_lr_0.0001sd_42_ep{epoch}.pt", weights_only=True)
        assert archived["completed_epoch"] == archived["epoch"] == epoch
        assert archived["step"] == epoch * 2 and archived["optimizer_state"] is not None
    for path in output.glob("lr*.pt"):
        assert path.name.startswith("lr0.0001_bs3_wd0.01_ep")
        image = path.with_suffix(".png")
        assert image.is_file() and image.read_bytes().startswith(b"\x89PNG")
    assert len(pd.read_csv(output / "loss_history.csv")) == expected_epochs
    assert metrics["test"]["auroc"] == binary_metrics(predictions.label, predictions.logit)["auroc"]


def test_macro_metrics_empty_and_partial_auc_scaling():
    """No eligible groups yield NaNs; standardized partial AUC can exceed AUROC."""
    empty = pd.DataFrame({"pmhc_id": ["p"], "label": [1], "prediction": [0.5]})
    result = macro_pmhc_metrics(empty)
    assert result["eligible_pmhc"] == 0 and "per_pmhc" not in result
    assert all(math.isnan(result[key]) for key in ("macro_auroc", "macro_auprc", "macro_auc_0_1"))
    metrics = binary_metrics([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 0
    assert metrics["auc_0_1"] == pytest.approx(0.47368421052631576)
    groups = pd.DataFrame({"pmhc_id": ["p1"] * 2 + ["p2"] * 4 + ["p3"],
                           "label": [0, 1, 0, 0, 1, 1, 1],
                           "prediction": [0.9, 0.1, 0.1, 0.2, 0.8, 0.9, 0.3]})
    macro = macro_pmhc_metrics(groups)
    assert macro["eligible_pmhc"] == 2
    assert macro["macro_auroc"] == 0.5
    assert macro["macro_auc_0_1"] == pytest.approx((metrics["auc_0_1"] + 1) / 2)


def test_ablation_epoch_argument_and_audit(tmp_path, monkeypatch, capsys):
    """The ablation launcher forwards epochs and emits no SHA metadata."""
    manifest, features = make_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ablation", "--manifest", str(manifest),
                         "--features-dir", str(features), "--output-root",
                         str(tmp_path / "runs"), "--expected-counts", "5", "2", "2",
                         "--modes", "A", "--seeds", "42", "--epochs", "2", "--dry-run"])
    ablation.main()
    output = capsys.readouterr().out
    assert "--epochs 2" in output and "sha" not in output
    assert not (tmp_path / "runs").exists()


def test_early_stopping_and_scheduler_rules():
    """Patience counts consecutive failures, resets on improvement, and is optional."""
    stopper = training.EarlyStopping(enabled=True, patience=5)
    assert stopper.update(1.0, 1)
    for epoch in range(2, 6):
        assert not stopper.update(1.0, epoch)
        assert not stopper.should_stop
    assert stopper.update(0.9, 6) and stopper.bad_epochs == 0
    for epoch in range(7, 12):
        stopper.update(1.0, epoch)
    assert stopper.should_stop and stopper.best_epoch == 6
    disabled = training.EarlyStopping(enabled=False, patience=1)
    disabled.update(1, 1)
    disabled.update(2, 2)
    assert not disabled.should_stop
    with pytest.raises(ValueError, match="Non-finite"):
        disabled.update(float("nan"), 3)


def test_early_stop_checkpoint_and_curve_outputs(tmp_path, monkeypatch):
    """A forced validation plateau stops at epoch 7, retains best epoch 2 and terminal 7."""
    torch.set_num_threads(1)
    training.set_seed(42)
    manifest, features = make_inputs(tmp_path)
    output = tmp_path / "run_es"
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest),
                         "--features-dir", str(features), "--output-dir", str(output),
                         "--epochs", "12", "--gradient-accumulation", "3",
                         "--pairformer-blocks", "0", "--hidden-dim", "8", "--dropout", "0",
                         "--early_stop", "--early_stop_patience", "5",
                         "--lr-scheduler", "plateau", "--lr-scheduler-patience", "2"])
    real_evaluate = training.evaluate
    validation_losses = iter([1.0, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])

    def plateau_evaluation(model, loader, *args):
        loss, predictions = real_evaluate(model, loader, *args)
        if loader.dataset.examples[0].sample_id.startswith("validation"):
            loss = next(validation_losses)
        return loss, predictions

    monkeypatch.setattr(training, "evaluate", plateau_evaluation)
    training.run_training(0, 1, training.parse_args(), [], torch.device("cpu"))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["stopped_early"] and metrics["final_epoch"] == 7
    assert metrics["best_epoch"] == 2 and metrics["best_validation_bce"] == 0.8
    assert set(path.name for path in output.glob("*.pt")) == {
        "lr0.0001_bs3_wd0.01_ep2_val_best_ep2.pt",
        "lr0.0001_bs3_wd0.01_ep7_es_ep7.pt",
    } | {f"bs3_lr_0.0001sd_42_ep{epoch}.pt" for epoch in range(1, 8)}
    history = pd.read_csv(output / "loss_history.csv")
    assert list(history.epoch) == list(range(1, 8))
    assert history.loc[history.epoch == 5, "next_lr"].item() == pytest.approx(0.00005)
    assert history.iloc[-1].lr == pytest.approx(0.00005)
    assert len(list(output.glob("*.png"))) == 2
    best = torch.load(output / metrics["best_checkpoint"], weights_only=True)
    final = torch.load(output / metrics["final_checkpoint"], weights_only=True)
    assert best["epoch"] == 2 and final["epoch"] == 7 and final["bad_epochs"] == 5
    assert any(not torch.equal(best["model"][key], final["model"][key]) for key in best["model"])
