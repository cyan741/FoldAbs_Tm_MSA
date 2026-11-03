import random
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_mini_esm_online_ddp import DeterministicFeatures, epoch_order, rank_plan, record_for, window_count
import train_mini_esm_online_ddp as training


def test_offline_encoder_assets_do_not_require_esm_weights(tmp_path):
    configs = SimpleNamespace(model_name="protenix_mini_esm_v0.5.0", use_msa=False,
        use_template=False, use_rna_msa=False, load_checkpoint_dir=tmp_path,
        data={name: tmp_path / name for name in (
            "ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file", "obsolete_release_data_csv")})
    checkpoint = tmp_path / f"{configs.model_name}.pt"
    for path in [checkpoint, *configs.data.values()]:
        path.write_bytes(b"asset")
    training.validate_cached_encoder_assets(configs)
    assert not (tmp_path / "esm2_t36_3B_UR50D.pt").exists()
    configs.use_template = True
    with pytest.raises(ValueError):
        training.validate_cached_encoder_assets(configs)
    configs.use_template = False
    checkpoint.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        training.validate_cached_encoder_assets(configs)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        training.validate_cached_encoder_assets(configs)


@pytest.mark.parametrize("length,world_size", [(1, 8), (17, 8), (48, 2), (238395, 8)])
def test_masked_shards_cover_exactly_once(length, world_size):
    plans = [rank_plan(length, world_size, rank) for rank in range(world_size)]
    assert len({len(plan) for plan in plans}) == 1
    indices = [index for plan in plans for index, valid in plan if valid]
    assert sorted(indices) == list(range(length))


@pytest.mark.parametrize("length,world_size,accumulation", [(1, 8, 3), (17, 8, 3), (25, 2, 3), (48, 2, 3)])
def test_ddp_loss_normalization_matches_global_mean(length, world_size, accumulation):
    values = torch.arange(1, length + 1, dtype=torch.float64)
    plans = [rank_plan(length, world_size, rank) for rank in range(world_size)]
    for start in range(0, len(plans[0]), accumulation):
        stop = min(start + accumulation, len(plans[0]))
        count = window_count(length, world_size, start, stop)
        gradients = []
        for plan in plans:
            parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
            for index, valid in plan[start:stop]:
                (parameter * values[index] * int(valid) * world_size / count).backward()
            gradients.append(parameter.grad)
        expected = values[start * world_size:min(stop * world_size, length)].mean()
        torch.testing.assert_close(torch.stack(gradients).mean(), expected)


def test_resume_cursor_does_not_replay_consumed_samples():
    for rank in range(2):
        full = rank_plan(25, 2, rank)
        resumed = rank_plan(25, 2, rank, start_round=6)
        assert resumed == full[6:]


def test_smoke_keeps_both_labels_and_longest_samples():
    frame = pd.DataFrame([
        {"sample_id": str(index), "label": index % 3 == 0, "tcra": "A" * (index + 1),
         "tcrb": "C", "peptide": "D", "hla_sequence": "E"} for index in range(30)
    ])
    first = epoch_order(frame, 42, 1, smoke_samples=12)
    pd.testing.assert_frame_equal(first, epoch_order(frame, 42, 1, smoke_samples=12))
    assert first.label.sum() == 4
    assert {"27", "29"}.issubset(first.sample_id)
    assert len(first) == first.sample_id.nunique() == 12
    assert record_for(first.iloc[0])["sequences"][0]["proteinChain"]["sequence"] == first.iloc[0].tcra


class RandomDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return random.random(), np.random.rand(), torch.rand(1)


def test_featurization_does_not_reset_training_rng():
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    dataset = DeterministicFeatures(RandomDataset())
    first, second = dataset[0], dataset[1]
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


class EvaluationProvider:
    def batches(self, frame):
        return frame.sample_id.tolist()

    def infer(self, batch, expected_id):
        assert batch == expected_id
        return torch.tensor([float(int(batch) - 1)])


def test_evaluation_gathers_predictions_and_restores_train_mode(tmp_path, monkeypatch):
    def gather(destination, local):
        destination[0] = local

    monkeypatch.setattr(training.dist, "all_gather_object", gather)
    monkeypatch.setattr(training, "forward", lambda head, features: features)
    head = torch.nn.Identity()
    frame = pd.DataFrame({"sample_id": ["0", "1", "2"], "label": [0, 0, 1], "pmhc_id": ["context"] * 3})
    args = SimpleNamespace(chunk_rounds=2, output_dir=tmp_path)
    metrics = training.evaluate(head, EvaluationProvider(), frame, args, 0, 1, "validation")
    assert head.training
    assert metrics["auroc"] == 1.0
    assert metrics["macro_auroc"] == 1.0
    assert np.isfinite(metrics["bce"])
    saved = pd.read_csv(tmp_path / "validation_predictions.csv")
    assert len(saved) == saved.sample_id.nunique() == 3
    frame.loc[1, "sample_id"] = "0"
    with pytest.raises(ValueError, match="lost or duplicated"):
        training.evaluate(head, EvaluationProvider(), frame, args, 0, 1, "duplicate")


@pytest.mark.parametrize("gpu_ids,free_memory,success", [("3,5", 35000, True), ("3,3", 35000, False),
                                                         ("3,5", 20000, False)])
def test_launcher_checks_gpu_selection_and_memory(tmp_path, gpu_ids, free_memory, success):
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text(f"#!/bin/bash\necho {free_memory}\n")
    nvidia.chmod(0o755)
    python = tmp_path / "fake-python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    accel = tmp_path / "accel.sh"
    accel.write_text("")
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "PYTHON": str(python),
           "ACCEL_ENV": str(accel), "GPU_IDS": gpu_ids, "MIN_FREE_MIB": "30000"}
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_mini_esm_online_ddp.sh"
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == success
    if success:
        assert "--nproc-per-node=2" in result.stdout
