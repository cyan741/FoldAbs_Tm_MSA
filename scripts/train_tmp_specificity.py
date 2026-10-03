#!/usr/bin/env python3
"""Train a TMP binary classifier on frozen Protenix s/z/p_dist features."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from foldabs_repro.tmp_prediction.dataset import TMPFeatureDataset, examples_from_manifest, single_item
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_line(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()


@torch.no_grad()
def evaluate(model, loader, device, loss_pos_weight=None):
    model.eval()
    rows, losses = [], []
    for batch in loader:
        logit = model(
            batch["s"].to(device), batch["z"].to(device), batch["p_dist"].to(device),
            batch["token_asym_id"].to(device), batch["roles"],
        )
        target = batch["label"].to(device)
        losses.append(float(F.binary_cross_entropy_with_logits(logit, target, pos_weight=loss_pos_weight).item()))
        rows.append(
            {
                "sample_id": batch["sample_id"],
                "original_id": batch["original_id"],
                "pmhc_id": batch["pmhc_id"],
                "label": int(target.item()),
                "logit": float(logit.item()),
                "prediction": float(torch.sigmoid(logit).item()),
            }
        )
    frame = pd.DataFrame(rows)
    return float(np.mean(losses)), frame


def parse_gpu_ids(values: list[str] | None) -> list[str]:
    """Normalize ``--gpu 0 1`` and ``--gpu 0,1`` into unique device IDs."""
    if values is None:
        return []
    gpu_ids = [piece.strip() for value in values for piece in value.split(",")]
    if any(not gpu_id for gpu_id in gpu_ids):
        raise ValueError("--gpu contains an empty device ID")
    if any(not gpu_id.isdigit() for gpu_id in gpu_ids):
        raise ValueError(f"--gpu accepts non-negative integer IDs, got {gpu_ids}")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpu contains duplicate device IDs: {gpu_ids}")
    return gpu_ids


def find_free_port() -> int:
    """Reserve an ephemeral localhost port for DDP process-group startup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def parse_args() -> argparse.Namespace:
    """Parse training arguments without initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=None,
        help=(
            "Physical GPU IDs. One ID selects single-GPU training; multiple IDs "
            "automatically enable DDP. Accepts '--gpu 0 1' or '--gpu 0,1'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-loss-window", type=int, default=50)
    parser.add_argument("--pairformer-blocks", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auto-pos-weight", action="store_true")
    return parser.parse_args()


def train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    gpu_ids: list[str],
    master_port: int | None,
) -> None:
    """Run one single-GPU worker, synchronizing gradients when world_size > 1."""
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TMP Pairformer head")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Requested {world_size} GPUs but only {torch.cuda.device_count()} are visible"
        )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if distributed:
        if master_port is None:
            raise ValueError("master_port is required for multi-GPU training")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{master_port}",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

    try:
        set_seed(args.seed + rank)
        run_training(rank, world_size, args, gpu_ids, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def run_training(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    gpu_ids: list[str],
    device: torch.device,
) -> None:
    """Train on one rank; rank zero owns evaluation and output artifacts."""
    distributed = world_size > 1
    is_main = rank == 0

    if min(
        args.max_steps,
        args.gradient_accumulation,
        args.validate_every,
        args.log_every,
        args.train_loss_window,
    ) < 1:
        raise ValueError("Step counts and train-loss window must be positive")

    manifest = pd.read_csv(args.manifest)
    if set(manifest.get("split", [])) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain explicit train, validation and test splits")

    example_sets = {
        split: examples_from_manifest(manifest, args.features_dir, split=split, require_label=True)
        for split in ("train", "validation", "test")
    }
    datasets = {name: TMPFeatureDataset(rows) for name, rows in example_sets.items()}
    train_sampler = (
        DistributedSampler(
            datasets["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=1,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=single_item,
    )
    evaluation_loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_item)
        for name, dataset in datasets.items()
    }

    first = datasets["train"][0]
    raw_model = TMPTrunkSpecificityHead(
        c_s=int(first["s"].shape[-1]),
        c_z=int(first["z"].shape[-1]),
        c_dist=int(first["p_dist"].shape[-1]),
        pairformer_blocks=args.pairformer_blocks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model = (
        DistributedDataParallel(raw_model, device_ids=[rank], output_device=rank)
        if distributed
        else raw_model
    )

    train_labels = np.asarray([row.label for row in example_sets["train"]], dtype=int)
    positives, negatives = int(train_labels.sum()), int((train_labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both positive and negative labels")
    pos_weight = (
        torch.tensor(negatives / positives, dtype=torch.float32, device=device)
        if args.auto_pos_weight else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = output_dir / "train.log"

    best_loss, optimizer_step = float("inf"), 0
    recent_train_losses: deque[float] = deque(maxlen=args.train_loss_window)
    train_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    log = log_path.open("w", encoding="utf-8") if is_main else None
    try:
        if is_main:
            selected_gpus = gpu_ids or ["first-visible"]
            log_line(
                log,
                json.dumps(
                    {
                        "event": "start",
                        "train_examples": len(datasets["train"]),
                        "validation_examples": len(datasets["validation"]),
                        "test_examples": len(datasets["test"]),
                        "train_positive": positives,
                        "train_negative": negatives,
                        "pairformer_blocks": args.pairformer_blocks,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "gradient_accumulation": args.gradient_accumulation,
                        "gpu_ids": selected_gpus,
                        "world_size": world_size,
                        "distributed": distributed,
                        "per_device_batch_size": 1,
                        "effective_batch_size": (
                            world_size * args.gradient_accumulation
                        ),
                    }
                ),
            )

        while optimizer_step < args.max_steps:
            model.train()
            micro_losses = []
            for micro_step in range(args.gradient_accumulation):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(train_epoch)
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                synchronize = micro_step == args.gradient_accumulation - 1
                sync_context = (
                    nullcontext()
                    if not distributed or synchronize
                    else model.no_sync()
                )
                with sync_context:
                    logit = model(
                        batch["s"].to(device),
                        batch["z"].to(device),
                        batch["p_dist"].to(device),
                        batch["token_asym_id"].to(device),
                        batch["roles"],
                    )
                    target = batch["label"].to(device)
                    loss = F.binary_cross_entropy_with_logits(
                        logit, target, pos_weight=pos_weight
                    )
                    micro_losses.append(float(loss.detach().item()))
                    (loss / args.gradient_accumulation).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            step_loss_tensor = torch.tensor(
                float(np.mean(micro_losses)), dtype=torch.float64, device=device
            )
            if distributed:
                dist.reduce(step_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            if is_main:
                step_loss = float(step_loss_tensor.item()) / world_size
                recent_train_losses.append(step_loss)

            if is_main and (
                optimizer_step % args.log_every == 0 or optimizer_step == 1
            ):
                log_line(
                    log,
                    json.dumps(
                        {
                            "event": "train",
                            "step": optimizer_step,
                            "train_bce": step_loss,
                            "train_bce_moving": float(np.mean(recent_train_losses)),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        }
                    ),
                )

            if optimizer_step % args.validate_every == 0 or optimizer_step == args.max_steps:
                if distributed:
                    dist.barrier()
                if is_main:
                    validation_loss, validation = evaluate(
                        raw_model,
                        evaluation_loaders["validation"],
                        device,
                        pos_weight,
                    )
                    record = {
                        "event": "validation",
                        "step": optimizer_step,
                        "train_bce_moving": float(np.mean(recent_train_losses)),
                        "validation_bce": validation_loss,
                        **binary_metrics(validation.label, validation.prediction),
                    }
                    log_line(log, json.dumps(record))
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        torch.save(
                            {
                                "model": raw_model.state_dict(),
                                "model_config": raw_model.config,
                                "step": optimizer_step,
                                "threshold": 0.5,
                                "feature_schema": "protenix_trunk_s_z_pdist_v1",
                            },
                            output_dir / "best_model.pt",
                        )
                        log_line(
                            log,
                            json.dumps(
                                {
                                    "event": "checkpoint",
                                    "step": optimizer_step,
                                    "validation_bce": validation_loss,
                                    "path": str(output_dir / "best_model.pt"),
                                }
                            ),
                        )
                if distributed:
                    dist.barrier()

        if is_main:
            checkpoint = torch.load(
                output_dir / "best_model.pt", map_location=device, weights_only=True
            )
            raw_model.load_state_dict(checkpoint["model"])
            result_frames = []
            metrics = {
                "best_validation_bce": best_loss,
                "best_step": checkpoint["step"],
            }
            for split, loader in evaluation_loaders.items():
                loss, predictions = evaluate(raw_model, loader, device, pos_weight)
                predictions.insert(1, "split", split)
                result_frames.append(predictions)
                metrics[split] = {
                    "bce": loss,
                    **binary_metrics(predictions.label, predictions.prediction),
                    "macro_pmhc": macro_pmhc_metrics(predictions),
                }

            pd.concat(result_frames, ignore_index=True).to_csv(
                output_dir / "predictions.csv", index=False
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
            )
            log_line(
                log,
                json.dumps({"event": "complete", "metrics": metrics}, allow_nan=True),
            )
        if distributed:
            dist.barrier()
    finally:
        if log is not None:
            log.close()


def main() -> None:
    """Select requested GPUs and launch single-process or DDP training."""
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    world_size = len(gpu_ids) if gpu_ids else 1
    if world_size == 1:
        train_worker(0, world_size, args, gpu_ids, None)
        return

    mp.spawn(
        train_worker,
        args=(world_size, args, gpu_ids, find_free_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
