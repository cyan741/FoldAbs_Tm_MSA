#!/usr/bin/env python3
"""Train a TMP binary classifier on frozen Protenix s/z/p_dist features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from foldabs_repro.tmp_prediction.dataset import TMPFeatureDataset, examples_from_manifest, single_item
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.pooling import POOLING_MODES


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-loss-window", type=int, default=50)
    parser.add_argument("--pairformer-blocks", type=int, default=4)
    parser.add_argument(
        "--pooling-mode", choices=POOLING_MODES, default="A",
        help="A: all mean; B: single mean/pair attention; C: all attention",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auto-pos-weight", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TMP Pairformer head")
    if min(
        args.max_steps,
        args.gradient_accumulation,
        args.validate_every,
        args.log_every,
        args.train_loss_window,
    ) < 1:
        raise ValueError("Step counts and train-loss window must be positive")

    set_seed(args.seed)
    device = torch.device("cuda")
    manifest = pd.read_csv(args.manifest)
    if set(manifest.get("split", [])) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain explicit train, validation and test splits")

    example_sets = {
        split: examples_from_manifest(manifest, args.features_dir, split=split, require_label=True)
        for split in ("train", "validation", "test")
    }
    datasets = {name: TMPFeatureDataset(rows) for name, rows in example_sets.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=single_item)
        for name, dataset in datasets.items()
    }

    first = datasets["train"][0]
    model = TMPTrunkSpecificityHead(
        c_s=int(first["s"].shape[-1]),
        c_z=int(first["z"].shape[-1]),
        c_dist=int(first["p_dist"].shape[-1]),
        pairformer_blocks=args.pairformer_blocks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pooling_mode=args.pooling_mode,
    ).to(device)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "git_sha": git_sha(),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "model_config": model.config,
        "feature_schema": "protenix_trunk_s_z_pdist_v1",
        "selection_metric": "validation_bce",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "pooling_parameters": sum(p.numel() for p in model.pooling.parameters()),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    log_path = output_dir / "train.log"

    best_loss, optimizer_step = float("inf"), 0
    recent_train_losses: deque[float] = deque(maxlen=args.train_loss_window)
    train_iterator = iter(loaders["train"])
    optimizer.zero_grad(set_to_none=True)

    with log_path.open("w", encoding="utf-8") as log:
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
                    "pooling_mode": args.pooling_mode,
                    "seed": args.seed,
                    "pooling_parameters": run_config["pooling_parameters"],
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "gradient_accumulation": args.gradient_accumulation,
                }
            ),
        )

        while optimizer_step < args.max_steps:
            model.train()
            micro_losses = []
            for _ in range(args.gradient_accumulation):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(loaders["train"])
                    batch = next(train_iterator)

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

            step_loss = float(np.mean(micro_losses))
            recent_train_losses.append(step_loss)

            if optimizer_step % args.log_every == 0 or optimizer_step == 1:
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
                validation_loss, validation = evaluate(
                    model, loaders["validation"], device, pos_weight
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
                            "model": model.state_dict(),
                            "model_config": model.config,
                            "step": optimizer_step,
                            "threshold": 0.5,
                            "feature_schema": "protenix_trunk_s_z_pdist_v1",
                            "run_config": run_config,
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

        checkpoint = torch.load(
            output_dir / "best_model.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        result_frames = []
        metrics = {
            "pooling_mode": args.pooling_mode,
            "seed": args.seed,
            "best_validation_bce": best_loss,
            "best_step": checkpoint["step"],
        }
        for split, loader in loaders.items():
            loss, predictions = evaluate(model, loader, device, pos_weight)
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
        log_line(log, json.dumps({"event": "complete", "metrics": metrics}, allow_nan=True))


if __name__ == "__main__":
    main()
