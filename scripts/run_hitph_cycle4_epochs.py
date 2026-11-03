#!/usr/bin/env python3
"""Rotate one epoch of HiTPH negatives around complete training checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCRIPT = ROOT / "scripts/make_protenix_input/make_tmp_protenix_input.py"
EXTRACT_SCRIPT = ROOT / "scripts/extract_protenix_embeddings.py"
TRAIN_SCRIPT = ROOT / "scripts/train_tmp_specificity.py"


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def final_checkpoint(output_dir: Path, epoch: int) -> Path | None:
    matches = [path for path in output_dir.glob("*.pt")
               if re.search(rf"_ep{epoch}\.pt$", path.name)
               and "_val_best_" not in path.name]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple final checkpoints for epoch {epoch}: {matches}")
    if not matches:
        return None
    state = torch.load(matches[0], map_location="cpu", weights_only=True)
    if int(state["epoch"]) != epoch or not state.get("optimizer_state"):
        raise ValueError(f"Incomplete epoch checkpoint: {matches[0]}")
    model = state["model_config"]
    if model.get("pairformer_blocks") != 4 or model.get("pooling_mode") != "C":
        raise ValueError(f"Wrong head in checkpoint: {matches[0]}")
    return matches[0]


def check_permanent(index_path: Path, expected: int) -> None:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError(f"Permanent index must contain {expected} unique samples")


def epoch_cache_complete(features: Path, manifest: Path, expected: int = 158930) -> bool:
    index_path, stats_path = features / "index.json", features / "extraction_stats.json"
    if not index_path.is_file() or not stats_path.is_file():
        return False
    stats = json.loads(stats_path.read_text())
    if stats.get("samples_total") != expected or stats.get("cycles") != 4 or stats.get("use_msa"):
        return False
    ids = {row["sample_id"] for row in json.loads(index_path.read_text())}
    expected_ids = set(pd.read_csv(manifest, usecols=["sample_id"]).sample_id)
    return len(ids) == expected and ids == expected_ids


def remove_epoch_negatives(
    manifest: Path, features_dir: Path, keep_ids: set[str] | None = None,
) -> int:
    frame = pd.read_csv(manifest, usecols=["sample_id", "label"])
    if not frame.label.eq(0).all() or frame.sample_id.duplicated().any():
        raise ValueError(f"Invalid train negative manifest: {manifest}")
    if not frame.sample_id.str.startswith("pepneg_train_").all():
        raise ValueError("Refusing to delete non-training features")
    keep_ids = set() if keep_ids is None else set(keep_ids)
    if any(not sample_id.startswith("pepneg_train_") for sample_id in keep_ids):
        raise ValueError("Refusing to retain an invalid training negative ID")
    count = 0
    for sample_id in frame.sample_id:
        if sample_id in keep_ids:
            continue
        path = features_dir / f"{sample_id}.pt"
        if path.is_file():
            path.unlink()
            count += 1
    return count


def select_extract_gpus(min_free_mib: int = 10240, maximum: int = 5) -> str:
    """Select cards solely by currently free VRAM, irrespective of processes."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    available = []
    for line in result.stdout.splitlines():
        gpu_id, free_mib = (piece.strip() for piece in line.split(","))
        if int(free_mib) >= min_free_mib:
            available.append((int(free_mib), int(gpu_id)))
    available.sort(reverse=True)
    return ",".join(str(gpu_id) for _, gpu_id in available[:maximum])


def wait_for_permanent(config: dict, config_path: Path, expected: int = 96670) -> Path:
    output_root = Path(config["output_root"])
    index = output_root / "features/index.json"
    last_message = 0.0
    while not index.is_file() or not (index.parent / "extraction_stats.json").is_file():
        now = time.monotonic()
        if now - last_message >= 3600 or last_message == 0:
            print(f"Waiting for {expected} permanent features: {index}", flush=True)
            last_message = now
        pid = int(json.loads(config_path.read_text())["permanent_extraction_pid"])
        proc = Path(f"/proc/{pid}/cmdline")
        if not proc.is_file() or not any(
            name in proc.read_bytes()
            for name in (b"extract_protenix_embeddings.py", b"monitor_permanent.py")
        ):
            raise RuntimeError("Permanent extraction stopped before publishing its index")
        time.sleep(60)
    check_permanent(index, expected)
    stats = json.loads((index.parent / "extraction_stats.json").read_text())
    if stats["cycles"] != 4 or stats["use_msa"] or stats["samples_total"] != expected:
        raise ValueError("Permanent extraction statistics do not match cycle4/no-MSA")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hitph_protenix_cycle4_nomsa.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--extract-gpu-ids", help="Physical IDs with sufficient free VRAM, comma-separated")
    parser.add_argument("--train-gpu-ids", help="Physical IDs with sufficient free VRAM, comma-separated")
    parser.add_argument("--output-dir", type=Path, help="Local training checkpoints and metrics")
    parser.add_argument("--cache-epoch1-after-permanent", action="store_true",
                        help="Wait for permanent extraction, then cache epoch 1 negatives only")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work/hitph_cycle4_nomsa")
    args = parser.parse_args()
    if not args.cache_epoch1_after_permanent and (
        args.epochs is None or args.epochs < 1 or not args.extract_gpu_ids
        or not args.train_gpu_ids or args.output_dir is None
    ):
        parser.error("Training requires --epochs, --extract-gpu-ids, --train-gpu-ids and --output-dir")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["cycles"] != 4 or config["use_msa"] or config["head_pooling_mode"] != "C" or config["head_pairformer_blocks"] != 4:
        raise ValueError("Expected cycle4/no-MSA/Pairformer4/C configuration")
    features = Path(config["output_root"]) / "features"
    if args.cache_epoch1_after_permanent:
        permanent_index = wait_for_permanent(config, args.config)
        permanent_bytes = sum(int(row["file_bytes"]) for row in json.loads(permanent_index.read_text()))
        epoch1_budget = 10_714_000_000_000
        if permanent_bytes + epoch1_budget + 1_000_000_000_000 >= config["storage_quota_bytes"]:
            raise RuntimeError("Epoch 1 cache would leave less than 1 TB under the 20 TB cap")
        permanent_snapshot = Path(config["output_root"]) / "permanent_index.json"
        if not permanent_snapshot.is_file():
            shutil.copy2(permanent_index, permanent_snapshot)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        epoch_input = args.work_dir / "train_negatives_epoch_0001.json"
        epoch_manifest = epoch_input.with_suffix(".manifest.csv")
        environment = os.environ.copy()
        environment.update({"PROTENIX_ROOT_DIR": config["assets_root"], "LAYERNORM_TYPE": "torch",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                            "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts"),
                                                           environment.get("PYTHONPATH", "")])})
        if not epoch_input.is_file() or not epoch_manifest.is_file():
            run([config["python"], str(INPUT_SCRIPT), "--hitph-splits-dir", config["splits_dir"],
                 "--no-msa", "--train-negative-epoch", "1", "--output-json", str(epoch_input)], env=environment)
        if config.get("epoch1_distributed"):
            run([config["python"], str(ROOT / "scripts/protenix_migration/epoch1_distributed.py"),
                 "finish", "--config", str(args.config), "--input-json", str(epoch_input)],
                env=environment)
            if not epoch_cache_complete(features, epoch_manifest):
                raise RuntimeError("Distributed epoch 1 extraction did not publish all 158,930 negatives")
            print("Epoch 1 negative cache complete; training may begin", flush=True)
            return
        selected = args.extract_gpu_ids or select_extract_gpus()
        if not selected:
            raise RuntimeError("No GPU has at least 10 GiB free for epoch 1 extraction")
        print(f"Permanent cache complete ({permanent_bytes} bytes); caching epoch 1 on GPUs {selected}", flush=True)
        if not epoch_cache_complete(features, epoch_manifest):
            run([config["python"], "-B", str(EXTRACT_SCRIPT), "--input-json", str(epoch_input),
                 "--output-dir", str(features), "--work-dir", str(args.work_dir / "extract_epoch_0001"),
                 "--gpu-ids", selected, "--cycles", "4", "--limit", "0", "--resume"], env=environment)
        if not epoch_cache_complete(features, epoch_manifest):
            raise RuntimeError("Epoch 1 extraction did not publish all 158,930 negatives")
        print("Epoch 1 negative cache complete; training may begin", flush=True)
        return
    if not features.is_dir():
        raise FileNotFoundError(f"Permanent feature directory missing: {features}")
    permanent_index = Path(config["output_root"]) / "permanent_index.json"
    if not permanent_index.is_file():
        current = features / "index.json"
        check_permanent(current, 96670)
        shutil.copy2(current, permanent_index)
    check_permanent(permanent_index, 96670)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PROTENIX_ROOT_DIR": config["assets_root"], "LAYERNORM_TYPE": "torch",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts"),
                                       environment.get("PYTHONPATH", "")]),
    })
    python = config["python"]
    previous: Path | None = None
    for epoch in range(1, args.epochs + 1):
        input_path = args.work_dir / f"train_negatives_epoch_{epoch:04d}.json"
        manifest = input_path.with_suffix(".manifest.csv")
        if not input_path.is_file() or not manifest.is_file():
            run([python, str(INPUT_SCRIPT), "--hitph-splits-dir", config["splits_dir"],
                 "--no-msa", "--train-negative-epoch", str(epoch),
                 "--output-json", str(input_path)], env=environment)
        checkpoint = final_checkpoint(args.output_dir, epoch)
        if checkpoint is None:
            if not epoch_cache_complete(features, manifest):
                run([python, "-B", str(EXTRACT_SCRIPT), "--input-json", str(input_path),
                     "--output-dir", str(features), "--work-dir", str(args.work_dir / f"extract_epoch_{epoch:04d}"),
                     "--gpu-ids", args.extract_gpu_ids, "--cycles", "4", "--limit", "0", "--resume"], env=environment)
            command = [python, "-B", str(TRAIN_SCRIPT), "--features-dir", str(features),
                       "--output-dir", str(args.output_dir), "--negative-sampling", "peptide_mismatch",
                       "--splits-dir", config["splits_dir"], "--epochs", str(epoch),
                       "--pairformer-blocks", "4", "--pooling-mode", "C", "--precision", "bf16",
                       "--gpu", *args.train_gpu_ids.split(",")]
            if previous is not None:
                command.extend(["--resume", str(previous)])
            run(command, env=environment)
            checkpoint = final_checkpoint(args.output_dir, epoch)
            if checkpoint is None:
                raise RuntimeError(f"Training did not produce epoch {epoch} checkpoint")
        keep_ids: set[str] = set()
        if epoch < args.epochs:
            next_input = args.work_dir / f"train_negatives_epoch_{epoch + 1:04d}.json"
            next_manifest = next_input.with_suffix(".manifest.csv")
            if not next_input.is_file() or not next_manifest.is_file():
                run([python, str(INPUT_SCRIPT), "--hitph-splits-dir", config["splits_dir"],
                     "--no-msa", "--train-negative-epoch", str(epoch + 1),
                     "--output-json", str(next_input)], env=environment)
            keep_ids = set(pd.read_csv(next_manifest, usecols=["sample_id"]).sample_id)
        removed = remove_epoch_negatives(manifest, features, keep_ids)
        retained = int(pd.read_csv(manifest, usecols=["sample_id"]).sample_id.isin(keep_ids).sum())
        message = f"Epoch {epoch} checkpoint verified: {checkpoint}; removed {removed} train negatives"
        if epoch < args.epochs:
            message += f"; retained {retained} shared with epoch {epoch + 1}"
        print(message, flush=True)
        previous = checkpoint


if __name__ == "__main__":
    main()
