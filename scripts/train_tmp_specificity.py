#!/usr/bin/env python3
"""Train a TMP binary classifier on frozen Protenix s/z/p_dist features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
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

from foldabs_repro.tmp_prediction.checkpoints import archive_epoch, epoch_checkpoint_name
from foldabs_repro.tmp_prediction.cache_audit import audit_cache
from foldabs_repro.tmp_prediction.dataset import TMPFeatureDataset, examples_from_manifest, single_item
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.pooling import POOLING_MODES


@dataclass
class EarlyStopping:
    """Track the strict validation-BCE minimum and consecutive non-improvements."""

    enabled: bool = False
    patience: int = 5
    best_loss: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, loss: float, epoch: int) -> bool:
        """Return whether BCE improved; equal BCE counts toward patience."""
        if not np.isfinite(loss):
            raise ValueError(f"Non-finite validation BCE at epoch {epoch}: {loss}")
        if loss < self.best_loss:
            self.best_loss, self.best_epoch, self.bad_epochs = loss, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Stop only when enabled and patience consecutive epochs did not improve."""
        return self.enabled and self.bad_epochs >= self.patience


def checkpoint_stem(args, world_size: int, epoch: int, kind: str = "final") -> str:
    """Name artifacts using initial LR, effective batch size, decay and epoch."""
    batch_size = world_size * args.gradient_accumulation
    stem = f"lr{args.learning_rate:g}_bs{batch_size}_wd{args.weight_decay:g}_ep{epoch}"
    if kind == "val_best":
        return f"{stem}_val_best_ep{epoch}"
    if kind == "es":
        return f"{stem}_es_ep{epoch}"
    if kind != "final":
        raise ValueError(f"Unknown checkpoint kind: {kind}")
    return stem


def make_scheduler(optimizer, args):
    """Build an optional validation-BCE-driven learning-rate scheduler."""
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience, threshold=0.0,
        min_lr=args.min_lr,
    )


def save_loss_curves(history: list[dict], output_dir: Path, stems: list[str]) -> None:
    """Save full epoch loss history and matching PNGs for retained checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd.DataFrame(history).to_csv(output_dir / "loss_history.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_bce_loss"] for row in history], label="Train BCE", marker="o")
    axis.plot(epochs, [row["valid_bce_loss"] for row in history], label="Validation BCE", marker="o")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="Training / validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for stem in stems:
        figure.savefig(output_dir / f"{stem}.png", dpi=160)
    plt.close(figure)


def save_checkpoint(path, model, run_config, epoch, step, validation_metrics,
                    stopper, scheduler, current_lr, *, optimizer=None, sampling_state=None,
                    rng_states=None, history=None, best_checkpoint=None) -> dict | None:
    """Save weights and complete epoch-boundary resume state."""
    payload = {
        "model": model.state_dict(), "model_config": model.config,
        "run_config": run_config, "epoch": epoch, "step": step,
        "threshold": 0.5, "feature_schema": "protenix_trunk_s_z_pdist_v1",
        "validation_metrics": validation_metrics,
        "best_validation_bce": stopper.best_loss, "best_epoch": stopper.best_epoch,
        "bad_epochs": stopper.bad_epochs, "learning_rate": current_lr,
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "sampling_state": sampling_state, "rng_states": rng_states,
        "history": history, "best_checkpoint": best_checkpoint,
    }
    if path is None:
        return payload
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible initialization/shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def log_line(handle, message: str) -> None:
    """Write a timestamped, human-readable line to console and train.log."""
    message = f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | {message}"
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()


@torch.no_grad()
def evaluate(model, loader, device, loss_pos_weight=None, precision="fp32"):
    """Return BCE and binary predictions; logit retains ranking for ROC/PR metrics."""
    model.eval()
    rows, losses = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16"):
            logit = model(
                batch["s"].to(device), batch["z"].to(device), batch["p_dist"].to(device),
                batch["token_asym_id"].to(device), batch["roles"],
            ).float()
        target = batch["label"].to(device)
        losses.append(float(F.binary_cross_entropy_with_logits(logit, target, pos_weight=loss_pos_weight).item()))
        rows.append(
            {
                "sample_id": batch["sample_id"],
                "original_id": batch["original_id"],
                "pmhc_id": batch["pmhc_id"],
                "label": int(target.item()),
                "logit": float(logit.item()),
                "prediction": int(logit.item() >= 0.0),
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
    parser.add_argument("--manifest", help="Required for static mode")
    parser.add_argument("--features-dir", help="Required for training; joint four-chain Protenix cache")
    parser.add_argument("--output-dir", help="Required for training")
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
    parser.add_argument("--negative-sampling", choices=["static", "peptide_mismatch"], default="static")
    parser.add_argument("--splits-dir", default="/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits",
                        help="peptide_mismatch: train/val/test.csv and four NumPy dictionaries")
    parser.add_argument("--sampling-preview-epoch", type=int,
                        help="Print peptide_mismatch epoch audit without CUDA, features or file output")
    parser.add_argument("--negatives-per-positive", type=float, default=2)
    parser.add_argument("--negative-sampling-seed", type=int, default=42)
    parser.add_argument("--resume", help="Resume a complete epoch checkpoint")
    parser.add_argument("--max-steps", type=int, default=4000,
                        help="Optimizer-step budget; stop at the first complete epoch reaching it")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of complete epochs; overrides --max-steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--validate-every", type=int, default=None,
                        help="Deprecated compatibility option; validation always runs once per epoch")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--pairformer-blocks", type=int, default=4)
    parser.add_argument('--head-kernel', choices=['auto', 'torch', 'cuequivariance'], default='auto',
                        help='auto uses cuEquivariance on CUDA and torch on CPU')
    parser.add_argument('--activation-checkpoint', action='store_true',
                        help='Trade extra head computation for lower activation memory')
    parser.add_argument(
        "--pooling-mode", choices=POOLING_MODES, default="A",
        help="A: all mean; B: single mean/pair attention; C: all attention",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--early_stop", "--early-stop", action="store_true",
                        help="Stop after patience epochs without validation BCE improvement")
    parser.add_argument("--early_stop_patience", "--early-stop-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                        help="Optional CUDA bfloat16 autocast to reduce activation memory")
    args = parser.parse_args()
    if args.sampling_preview_epoch is not None:
        if args.negative_sampling != "peptide_mismatch" or args.sampling_preview_epoch < 1:
            parser.error("--sampling-preview-epoch >= 1 requires peptide_mismatch")
    elif not args.features_dir or not args.output_dir:
        parser.error("Training requires --features-dir and --output-dir")
    if args.negative_sampling != "peptide_mismatch" and not args.manifest:
        parser.error("static requires --manifest")
    return args



def shard_epoch_ids(ids, world_size=1, rank=0):
    """Shard one logical epoch; pad only to synchronize DDP optimizer steps."""
    if not ids or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid epoch plan or distributed rank")
    total = math.ceil(len(ids) / world_size) * world_size
    padding = total - len(ids)
    padded = list(ids) + (list(ids) * math.ceil(padding / len(ids)))[:padding]
    return padded[rank:total:world_size], padding


class EpochIndexSampler:
    """Index the current immutable epoch dataset using its exact sample IDs."""

    def __init__(self, sample_ids):
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Dataset IDs must be unique")
        self.index = {name: index for index, name in enumerate(sample_ids)}
        self.indices = []

    def set_ids(self, ids):
        self.indices = [self.index[name] for name in ids]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PeptideMismatchSampling:
    """Use exact delivered CSV/NumPy identities; never borrow another split's peptides.

    Evaluation remains fixed. Train negatives are resampled at each epoch and
    require their own joint four-chain feature tensors, not positive features.
    Numeric negative IDs are stable across epochs and require no SHA files.
    """

    def __init__(self, args):
        import importlib.util
        module_path = Path(__file__).resolve().parent / "negative_sampling" / "peptide_mismatch.py"
        spec = importlib.util.spec_from_file_location("peptide_mismatch", module_path)
        self.sampling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.sampling)
        self.args = args
        ratio = args.negatives_per_positive
        if not np.isfinite(ratio) or ratio < 1 or ratio != int(ratio):
            raise ValueError("peptide_mismatch requires a positive integer ratio")
        self.n = int(ratio)
        root = Path(args.splits_dir)
        self.frames = {split: pd.read_csv(root / f"{split}.csv", keep_default_na=False)
                       for split in ("train", "val", "test")}
        self.blacklist = self.sampling.load_numpy_dictionary(root / "blacklilst.npy")
        self.pools = {split: self.sampling.load_numpy_dictionary(root / f"{split}_pep2candidates.npy")
                      for split in self.frames}
        source = pd.read_csv(root / "positive_splits.csv", keep_default_na=False)
        source["split"] = source.split.replace({"validation": "val"})
        hla_contexts = source[["hla_alleles", "hla_clip"]].drop_duplicates()
        if hla_contexts.hla_alleles.duplicated().any():
            raise ValueError("One normalized HLA context must have exactly one clip")
        self.hla_clips = dict(hla_contexts.itertuples(index=False, name=None))
        self.positive_ids = {}
        self.groups = {}
        self.group_ids = {}
        self.peptide_ids = {}
        sequences = ["tcra_v", "tcrb_v", "peptide", "hla_clip"]
        for split, frame in self.frames.items():
            if list(frame.columns) != self.sampling.COLUMNS:
                raise ValueError(f"Unexpected columns in {split}.csv")
            if frame.duplicated(sequences).any() or not frame.label.isin([0, 1]).all():
                raise ValueError(f"Duplicate input or invalid labels in {split}")
            positive = frame[frame.label.eq(1)]
            original = source[source.split.eq(split)]
            identity = sequences + ["hla_alleles_raw", "hla_alleles"]
            if set(positive[identity].itertuples(index=False, name=None)) != set(original[identity].itertuples(index=False, name=None)):
                raise ValueError(f"Delivered positives differ from the source in {split}")
            self.positive_ids[split] = {tuple(getattr(row, c) for c in sequences): str(row.record_id)
                                       for row in original.itertuples(index=False)}
            if self.pools[split] != self.sampling.build_peptide_pool(positive):
                raise ValueError(f"Peptide dictionary does not match {split} positives")
            self.groups[split] = self.sampling.prepare_groups(positive, self.pools[split], self.blacklist)
            self.group_ids[split] = {group[0]: i for i, group in enumerate(self.groups[split])}
            self.peptide_ids[split] = {context: {pep: i for i, pep in enumerate(sorted(peptides))}
                                       for context, peptides in self.pools[split].items()}
            if split == "train" and not frame.label.eq(1).all():
                raise ValueError("train.csv must contain positives only")
            if split != "train" and int(frame.label.eq(0).sum()) != 2 * len(positive):
                raise ValueError(f"Fixed {split}.csv must have global ratio 1:2")
        blacklist_source = source
        known_path = root / "known_positives.csv"
        if known_path.exists():
            known = pd.read_csv(known_path, keep_default_na=False)
            if list(known.columns) != ["tcra_v", "tcrb_v", "peptide"] or known.eq("").any().any():
                raise ValueError("Invalid additional known-positive blacklist source")
            blacklist_source = pd.concat([source, known], ignore_index=True)
        if self.blacklist != self.sampling.build_blacklist(blacklist_source):
            raise ValueError("Blacklist must contain all positive TCR/peptide pairs across splits")
        self.fixed_negative_inputs = set()
        for split, frame in self.frames.items():
            for row in frame[frame.label.eq(0)].itertuples(index=False):
                identity = (row.tcra_v, row.tcrb_v, row.peptide, row.hla_clip)
                if identity in self.fixed_negative_inputs:
                    raise ValueError("Cross-split duplicate fixed negative")
                self.fixed_negative_inputs.add(identity)
                if row.peptide not in self.pools[split].get(row.hla_alleles, set()) or row.peptide in self.blacklist.get((row.tcra_v, row.tcrb_v), set()):
                    raise ValueError(f"Illegal fixed negative in {split}")
        contexts = sorted({(hla, clip, pep) for pool in self.pools.values()
                           for hla, peptides in pool.items() for pep in peptides
                           for clip in [self.hla_clips[hla]]})
        self.pmhc_ids = {key: f"pepmhc_{i:06d}" for i, key in enumerate(contexts)}

    def epoch(self, epoch):
        frame, report = self.sampling.sample_epoch(
            self.frames["train"], self.pools["train"], self.blacklist,
            seed=self.args.negative_sampling_seed, epoch=epoch, n=self.n,
            groups=self.groups["train"],
        )
        negatives = frame[frame.label.eq(0)]
        identities = set(negatives[["tcra_v", "tcrb_v", "peptide", "hla_clip"]].itertuples(index=False, name=None))
        if identities & self.fixed_negative_inputs:
            raise ValueError("Dynamic train negatives overlap fixed evaluation inputs")
        return self.manifest(frame, "train"), report

    def manifest(self, frame, split):
        rows = []
        for row in frame.itertuples(index=False):
            key = (row.tcra_v, row.tcrb_v, row.peptide, row.hla_clip)
            if row.label:
                sample_id = self.positive_ids[split][key]
            else:
                group = self.group_ids[split][row.hla_alleles, row.hla_clip, row.tcra_v, row.tcrb_v]
                peptide = self.peptide_ids[split][row.hla_alleles][row.peptide]
                sample_id = f"pepneg_{split}_{group:06d}_{peptide:05d}"
            rows.append({"sample_id": sample_id, "original_id": sample_id,
                         "pmhc_id": self.pmhc_ids[row.hla_alleles, row.hla_clip, row.peptide],
                         "split": "validation" if split == "val" else split, "label": row.label,
                         "tcra": row.tcra_v, "tcrb": row.tcrb_v,
                         "peptide": row.peptide, "hla_sequence": row.hla_clip})
        return pd.DataFrame(rows)

    def state(self):
        # Exact values detect changed source/pools on resume without file hashes.
        return {"positive_rows": sorted(self.frames["train"].itertuples(index=False, name=None)),
                "blacklist": sorted((key, sorted(value)) for key, value in self.blacklist.items()),
                "peptide_pools": {split: sorted((key, sorted(value)) for key, value in pool.items())
                                  for split, pool in self.pools.items()}}

    def audit_features(self, frame, examples):
        sequences = {r.sample_id: [r.tcra, r.tcrb, r.peptide, r.hla_sequence]
                     for r in frame.itertuples(index=False)}
        report = audit_cache(examples, sequences)
        if not report["ready"]:
            raise ValueError(f"Joint peptide-mismatch feature preflight failed: {report['failures'][:5]}")


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
        args.log_every,
    ) < 1:
        raise ValueError("Step counts must be positive")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.early_stop_patience < 1 or args.lr_scheduler_patience < 0:
        raise ValueError("Early-stop patience must be positive; scheduler patience nonnegative")
    if not 0 < args.lr_scheduler_factor < 1 or not 0 <= args.min_lr <= args.learning_rate:
        raise ValueError("Scheduler factor must be in (0,1), min LR in [0, initial LR]")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support bfloat16")

    peptide_mode = args.negative_sampling == "peptide_mismatch"
    peptide_sampling = PeptideMismatchSampling(args) if peptide_mode else None
    if peptide_mode:
        # A resumed run only needs the upcoming epoch's feature files. The
        # previous epoch's negatives may already have been reclaimed.
        initial_epoch = int(torch.load(args.resume, map_location="cpu", weights_only=True)["epoch"]) + 1 if args.resume else 1
        epoch_frame, epoch_report = peptide_sampling.epoch(initial_epoch)
        manifest = pd.concat([epoch_frame] + [peptide_sampling.manifest(peptide_sampling.frames[split], split)
                                             for split in ("val", "test")], ignore_index=True)
    else:
        manifest = pd.read_csv(args.manifest)
    if set(manifest.get("split", [])) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain explicit train, validation and test splits")

    example_sets = {
        split: examples_from_manifest(manifest, args.features_dir, split=split, require_label=True)
        for split in ("train", "validation", "test")
    }
    sampling_state = {"mode": args.negative_sampling, "seed": args.negative_sampling_seed,
                      "ratio": args.negatives_per_positive, "world_size": world_size}
    all_ids = [row.sample_id for rows in example_sets.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Sample IDs must be unique across all splits")
    if peptide_mode:
        sampling_state["peptide_identity"] = peptide_sampling.state()
    if peptide_mode:
        stable = pd.concat([peptide_sampling.manifest(peptide_sampling.frames[split], split)
                            for split in ("train", "val", "test")], ignore_index=True)
        sampling_state["records"] = sorted((row.sample_id, row.pmhc_id, row.label, row.split)
                                           for row in stable.itertuples(index=False))
    else:
        sampling_state["records"] = sorted((row.sample_id, row.pmhc_id, row.label, split)
                                           for split, rows in example_sets.items() for row in rows)
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

    if peptide_mode:
        peptide_sampling.audit_features(manifest, [row for rows in example_sets.values() for row in rows])
    first = datasets["train"][0]
    raw_model = TMPTrunkSpecificityHead(
        c_s=int(first["s"].shape[-1]),
        c_z=int(first["z"].shape[-1]),
        c_dist=int(first["p_dist"].shape[-1]),
        pairformer_blocks=args.pairformer_blocks,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pooling_mode=args.pooling_mode,
        triangle_multiplicative=('cuequivariance' if device.type == 'cuda' else 'torch') if args.head_kernel == 'auto' else args.head_kernel,
        triangle_attention=('cuequivariance' if device.type == 'cuda' else 'torch') if args.head_kernel == 'auto' else args.head_kernel,
        activation_checkpoint=args.activation_checkpoint,
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
    scheduler = make_scheduler(optimizer, args)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume and any(output_dir.glob("*.pt")):
            raise FileExistsError(f"Use a fresh output directory to preserve checkpoints: {output_dir}")
        run_config = {
            **vars(args),
            "validation_schedule": "every_epoch",
            "checkpoint_policy": "retain_every_epoch",
            "negative_pool_size": (epoch_report["unique_negative_capacity"] if peptide_mode
                                   else 0),
            "logical_epoch_samples": positives + negatives,
            "model_config": raw_model.config,
            "feature_schema": "protenix_trunk_s_z_pdist_v1",
            "selection_metric": "validation_bce",
            "effective_batch_size": world_size * args.gradient_accumulation,
            "trainable_parameters": sum(
                parameter.numel() for parameter in raw_model.parameters()
                if parameter.requires_grad
            ),
            "pooling_parameters": sum(
                parameter.numel() for parameter in raw_model.pooling.parameters()
            ),
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
    if distributed:
        dist.barrier()
    log_path = output_dir / "train.log"

    stopper = EarlyStopping(args.early_stop, args.early_stop_patience)
    optimizer_step = 0
    best_path = None
    history = []
    stopped_early = False
    train_epoch = 0

    rng_states = None
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        if resumed.get("sampling_state") != sampling_state:
            raise ValueError("Checkpoint sampling configuration or candidate IDs differ")
        for key in ("gradient_accumulation", "seed", "precision", "learning_rate", "weight_decay",
                    "early_stop", "early_stop_patience", "lr_scheduler", "lr_scheduler_factor",
                    "lr_scheduler_patience", "min_lr", "auto_pos_weight"):
            if resumed["run_config"].get(key) != getattr(args, key):
                raise ValueError(f"Resume configuration differs: {key}")
        if resumed["model_config"] != raw_model.config:
            raise ValueError("Resume model configuration differs")
        if resumed.get("optimizer_state") is None or not resumed.get("rng_states"):
            raise ValueError("Checkpoint lacks epoch-boundary resume state")
        raw_model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(resumed["scheduler_state"])
        stopper.best_loss = resumed["best_validation_bce"]
        stopper.best_epoch = resumed["best_epoch"]
        stopper.bad_epochs = resumed["bad_epochs"]
        train_epoch, optimizer_step = resumed["epoch"], resumed["step"]
        if stopper.should_stop or (args.epochs is not None and train_epoch >= args.epochs) or (args.epochs is None and optimizer_step >= args.max_steps):
            raise ValueError("Resume budget must extend beyond the completed epoch and early stopping")
        history = resumed["history"]
        best_path = Path(resumed["best_checkpoint"])
        if not best_path.is_file():
            archived_best = Path(args.resume).parent / epoch_checkpoint_name(
                world_size * args.gradient_accumulation, args.learning_rate, args.seed, resumed["best_epoch"])
            if not archived_best.is_file():
                raise FileNotFoundError(f"Resume needs retained validation-best checkpoint: {best_path}")
            best_path = archived_best
        if is_main and best_path.parent.resolve() != output_dir.resolve():
            import shutil
            target = output_dir / best_path.name
            shutil.copy2(best_path, target)
            best_path = target
        restore_rng(resumed["rng_states"][rank])
    log = log_path.open("a" if args.resume else "w", encoding="utf-8") if is_main else None
    try:
        if is_main:
            selected_gpus = gpu_ids or [os.environ.get("CUDA_VISIBLE_DEVICES", "first-visible")]
            log_line(
                log,
                f"Start | train={len(datasets['train'])} "
                f"validation={len(datasets['validation'])} test={len(datasets['test'])} "
                f"pooling={args.pooling_mode} seed={args.seed} "
                f"GPUs={','.join(selected_gpus)} world_size={world_size} "
                f"batch_size_per_device=1 accumulation={args.gradient_accumulation} "
                f"lr={args.learning_rate:g} weight_decay={args.weight_decay:g}",
            )
            if args.validate_every is not None:
                log_line(log, "--validate-every is ignored; validation runs once per epoch.")

        while True:
            train_epoch += 1
            if peptide_mode:
                epoch_frame, epoch_report = peptide_sampling.epoch(train_epoch)
                epoch_examples = examples_from_manifest(epoch_frame, args.features_dir, require_label=True)
                # A fresh immutable dataset prevents stale epoch negatives in workers.
                peptide_sampling.audit_features(epoch_frame, epoch_examples)
                epoch_ids = epoch_frame.sample_id.tolist()
                local_ids, padding = shard_epoch_ids(epoch_ids, world_size, rank)
                epoch_sampler = EpochIndexSampler(epoch_ids)
                epoch_sampler.set_ids(local_ids)
                train_loader = DataLoader(TMPFeatureDataset(epoch_examples), batch_size=1,
                                          sampler=epoch_sampler, collate_fn=single_item)
                if is_main:
                    log_line(log, f"Peptide mismatch epoch={train_epoch} {json.dumps(epoch_report)} "
                             f"logical_samples={len(epoch_ids)} padding={padding}")
            elif train_sampler is not None:
                train_sampler.set_epoch(train_epoch - 1)
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum, epoch_samples = 0.0, 0
            micro_losses = []
            batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                # Normalize the last, possibly short, accumulation group by its
                # actual size. Never carry gradients/samples across epochs.
                group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
                group_size = min(args.gradient_accumulation, batches - group_start)
                synchronize = (batch_index - group_start + 1) == group_size
                sync_context = (
                    nullcontext() if not distributed or synchronize else model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=args.precision == "bf16"):
                        logit = model(
                            batch["s"].to(device), batch["z"].to(device),
                            batch["p_dist"].to(device), batch["token_asym_id"].to(device),
                            batch["roles"],
                        ).float()
                    target = batch["label"].to(device)
                    loss = F.binary_cross_entropy_with_logits(
                        logit, target, pos_weight=pos_weight
                    )
                    loss_value = float(loss.detach().item())
                    epoch_loss_sum += loss_value
                    epoch_samples += 1
                    micro_losses.append(loss_value)
                    (loss / group_size).backward()
                if not synchronize:
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                step_loss_tensor = torch.tensor(
                    float(np.mean(micro_losses)), dtype=torch.float64, device=device
                )
                micro_losses.clear()
                if distributed:
                    dist.reduce(step_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
                if is_main and (optimizer_step % args.log_every == 0 or optimizer_step == 1):
                    log_line(
                        log,
                        f"Epoch {train_epoch} | Batch {batch_index + 1}/{batches} "
                        f"| Step {optimizer_step} | loss={step_loss_tensor.item() / world_size:.6f} "
                        f"| lr={optimizer.param_groups[0]['lr']:g}",
                    )

            totals = torch.tensor(
                [epoch_loss_sum, epoch_samples], dtype=torch.float64, device=device
            )
            validation_loss_tensor = torch.zeros((), dtype=torch.float64, device=device)
            if distributed:
                dist.reduce(totals, dst=0, op=dist.ReduceOp.SUM)
                dist.barrier()
            if is_main:
                train_loss = float((totals[0] / totals[1]).item())
                validation_loss, validation = evaluate(
                    raw_model, evaluation_loaders["validation"], device, pos_weight,
                    args.precision,
                )
                validation_loss_tensor.fill_(validation_loss)
                validation_metrics = {
                    "bce": validation_loss,
                    **binary_metrics(validation.label, validation.logit),
                    "macro_pmhc": macro_pmhc_metrics(validation),
                }
            # All ranks step the scheduler using rank-zero validation BCE and
            # receive the same early-stop decision before starting the next epoch.
            if distributed:
                dist.broadcast(validation_loss_tensor, src=0)
            if not torch.isfinite(validation_loss_tensor):
                raise ValueError(f"Non-finite validation BCE at epoch {train_epoch}")
            if scheduler is not None:
                scheduler.step(float(validation_loss_tensor.item()))
            next_lr = float(optimizer.param_groups[0]["lr"])
            stop_tensor = torch.zeros((), dtype=torch.int64, device=device)
            local_rng = capture_rng()
            rng_states = [None] * world_size
            if distributed:
                dist.all_gather_object(rng_states, local_rng)
            else:
                rng_states = [local_rng]
            if is_main:
                improved = stopper.update(validation_loss, train_epoch)
                history.append({
                    "epoch": train_epoch, "step": optimizer_step,
                    "train_bce_loss": train_loss, "valid_bce_loss": validation_loss,
                    "lr": epoch_lr, "next_lr": next_lr,
                })
                log_line(
                    log,
                    f"Epoch {train_epoch} complete | Step {optimizer_step} "
                    f"| loss={train_loss:.6f} | val_loss={validation_loss:.6f} "
                    f"| val_AUROC={validation_metrics['auroc']:.6f} "
                    f"| val_AUPRC={validation_metrics['auprc']:.6f} "
                    f"| val_AUC_0.1={validation_metrics['auc_0_1']:.6f} "
                    f"| lr={epoch_lr:g} next_lr={next_lr:g} "
                    f"| no_improvement={stopper.bad_epochs}/{stopper.patience}",
                )
                if improved:
                    new_path = output_dir / (
                        checkpoint_stem(args, world_size, train_epoch, "val_best") + ".pt"
                    )
                    save_checkpoint(
                        new_path, raw_model, run_config, train_epoch, optimizer_step,
                        validation_metrics, stopper, scheduler, next_lr,
                        optimizer=optimizer, sampling_state=sampling_state, rng_states=rng_states,
                        history=history, best_checkpoint=str(new_path.resolve()),
                    )
                    # Delete only the previous best created by this run; keep
                    # exactly one validation-best checkpoint at any time.
                    if (best_path is not None and best_path != new_path
                            and best_path.name.startswith("lr") and "_val_best_ep" in best_path.name):
                        best_path.unlink()
                    best_path = new_path
                    log_line(
                        log, f"Checkpoint saved | epoch={train_epoch} "
                        f"step={optimizer_step} val_loss={stopper.best_loss:.6f} path={best_path}",
                    )
                archive_epoch(
                    save_checkpoint(None, raw_model, run_config, train_epoch, optimizer_step,
                                    validation_metrics, stopper, scheduler, next_lr,
                                    optimizer=optimizer, sampling_state=sampling_state, rng_states=rng_states,
                                    history=history, best_checkpoint=str(best_path.resolve())),
                    output_dir, batch_size=world_size * args.gradient_accumulation,
                    learning_rate=args.learning_rate, seed=args.seed, epoch=train_epoch,
                )
                stop_tensor.fill_(int(stopper.should_stop))
            if distributed:
                dist.broadcast(stop_tensor, src=0)
            stopped_early = bool(stop_tensor.item())
            if stopped_early or (
                args.epochs is not None and train_epoch >= args.epochs
            ) or (
                args.epochs is None and optimizer_step >= args.max_steps
            ):
                break

        if is_main:
            final_kind = "es" if stopped_early else "final"
            final_stem = checkpoint_stem(args, world_size, train_epoch, final_kind)
            final_path = output_dir / f"{final_stem}.pt"
            save_checkpoint(
                final_path, raw_model, run_config, train_epoch, optimizer_step,
                validation_metrics, stopper, scheduler, next_lr,
                optimizer=optimizer, sampling_state=sampling_state, rng_states=rng_states,
                history=history, best_checkpoint=str(best_path.resolve()),
            )
            log_line(
                log, f"{'Early stopping' if stopped_early else 'Training finished'} "
                f"| epoch={train_epoch} step={optimizer_step} "
                f"| checkpoint={final_path}",
            )
            save_loss_curves(history, output_dir, [best_path.stem, final_stem])
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
            raw_model.load_state_dict(checkpoint["model"])
            log_line(
                log, f"Testing validation-best checkpoint | epoch={checkpoint['epoch']} "
                f"step={checkpoint['step']}",
            )
            # Export only hard labels and logits. ROC/PR metrics always use
            # continuous logits, never thresholded class predictions.
            test_loss, predictions = evaluate(
                raw_model, evaluation_loaders["test"], device, pos_weight, args.precision
            )
            predictions.insert(1, "split", "test")
            if "nearest_train_peptide_distance" in manifest:
                distances = manifest.set_index("sample_id").nearest_train_peptide_distance
                predictions["nearest_train_peptide_distance"] = predictions.sample_id.map(distances)
            predictions.to_csv(output_dir / "predictions.csv", index=False)
            test_metrics = {
                "bce": test_loss,
                **binary_metrics(predictions.label, predictions.logit),
                "macro_pmhc": macro_pmhc_metrics(predictions),
            }
            if "nearest_train_peptide_distance" in predictions:
                test_metrics["peptide_distance_strata"] = {
                    str(int(distance)): {"samples": len(part), **binary_metrics(part.label, part.logit)}
                    for distance, part in predictions.groupby("nearest_train_peptide_distance", sort=True)
                }
            metrics = {
                "pooling_mode": args.pooling_mode, "seed": args.seed,
                "best_validation_bce": stopper.best_loss,
                "best_step": checkpoint["step"], "best_epoch": checkpoint["epoch"],
                "stopped_early": stopped_early, "final_epoch": train_epoch,
                "best_checkpoint": best_path.name, "final_checkpoint": final_path.name,
                "validation": checkpoint["validation_metrics"], "test": test_metrics,
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
            )
            log_line(
                log, f"Test | samples={len(predictions)} | loss={test_loss:.6f} "
                f"| AUROC={test_metrics['auroc']:.6f} "
                f"| AUPRC={test_metrics['auprc']:.6f} "
                f"| AUC_0.1={test_metrics['auc_0_1']:.6f}",
            )
        if distributed:
            dist.barrier()
    finally:
        if log is not None:
            log.close()


def main() -> None:
    """Select requested GPUs and launch single-process or DDP training."""
    args = parse_args()
    if args.sampling_preview_epoch is not None:
        sampling = PeptideMismatchSampling(args)
        frame, report = sampling.epoch(args.sampling_preview_epoch)
        print(json.dumps({"seed": args.negative_sampling_seed, "epoch": args.sampling_preview_epoch,
                          **report, "unique_ids": frame.sample_id.nunique(),
                          "feature_example": frame.loc[frame.label.eq(0), "sample_id"].iloc[0] + ".pt"}, indent=2))
        return
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
