#!/usr/bin/env python3
"""Train a DDP TMP head with frozen mini-Protenix online features on every rank."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from foldabs_repro.tmp_prediction.checkpoints import archive_epoch, nominal_batch_size
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from train_tmp_specificity import PeptideMismatchSampling, capture_rng, restore_rng, set_seed


def epoch_order(frame, seed, epoch, smoke_samples=0):
    if smoke_samples:
        lengths = sum(frame[column].str.len() for column in ("tcra", "tcrb", "peptide", "hla_sequence"))
        selected = []
        for label, count in ((1, smoke_samples // 3), (0, smoke_samples - smoke_samples // 3)):
            candidates = frame[frame.label.eq(label)]
            longest = lengths.loc[candidates.index].idxmax()
            selected.append(candidates.loc[[longest]])
            selected.append(candidates.drop(longest).sample(n=count - 1, random_state=seed))
        frame = pd.concat(selected, ignore_index=True)
    return frame.sample(frac=1, random_state=seed + epoch).reset_index(drop=True)


def rank_plan(length, world_size, rank, start_round=0, rounds=None):
    total_rounds = math.ceil(length / world_size)
    stop = total_rounds if rounds is None else min(total_rounds, start_round + rounds)
    positions = [step * world_size + rank for step in range(start_round, stop)]
    return [(position if position < length else 0, position < length) for position in positions]


def window_count(length, world_size, start_round, stop_round):
    return min(length, stop_round * world_size) - min(length, start_round * world_size)


def atomic_json(value, path):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def record_execution_change(saved, config, allow_change):
    """Require explicit opt-in for precision/recomputation changes on resume."""
    previous = saved['run_config']
    defaults = {'feature_handoff': 'fp16', 'activation_checkpoint': False}
    changes = {key: {'before': previous.get(key, default), 'after': config[key]}
               for key, default in defaults.items()
               if previous.get(key, default) != config[key]}
    history = list(previous.get('execution_changes', []))
    if changes:
        if not allow_change:
            raise ValueError(f'Resume execution configuration mismatch: {changes}; '
                             'use --allow-execution-config-change for an intentional migration')
        history.append({'step': saved['step'], 'epoch': saved['epoch'],
                        'cursor': saved['cursor'], 'changes': changes})
    config['execution_changes'] = history


def record_for(row):
    return {"name": row.sample_id, "sequences": [
        {"proteinChain": {"sequence": sequence, "count": 1}}
        for sequence in (row.tcra, row.tcrb, row.peptide, row.hla_sequence)
    ]}


class DeterministicFeatures(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            random.seed(42)
            np.random.seed(42)
            torch.random.default_generator.manual_seed(42)
            try:
                return self.dataset[index]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


def initialize_worker(worker_id):
    from extract_protenix_embeddings import _install_empty_template_fast_path

    torch.set_num_threads(1)
    _install_empty_template_fast_path()


def validate_cached_encoder_assets(configs):
    if configs.model_name != "protenix_mini_esm_v0.5.0" or configs.use_msa or configs.use_template or configs.use_rna_msa:
        raise ValueError("Offline cached encoder requires mini-ESM without MSA/templates")
    required = [Path(configs.load_checkpoint_dir) / f"{configs.model_name}.pt"]
    required.extend(Path(configs.data[name]) for name in (
        "ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file", "obsolete_release_data_csv"))
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Required offline Protenix asset is missing or empty: {path}")


class OnlineFeatures:
    def __init__(self, args, rank, device):
        from configs.configs_model_type import model_configs
        from extract_protenix_embeddings import _install_empty_template_fast_path, _load_checkpointed_runner
        from protenix.utils.distributed import DIST_WRAPPER

        self.args, self.device = args, device
        self.work = args.output_dir / f"rank_{rank}"
        self.work.mkdir(parents=True, exist_ok=True)
        self.cache = args.local_esm_cache
        for directory in ("embeddings/chains", "metadata"):
            (self.cache / directory).mkdir(parents=True, exist_ok=True)
        self.spec = json.loads((args.esm_cache_dir / "run_spec.json").read_text())
        spec_path = self.cache / "run_spec.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != self.spec:
            raise ValueError("Local ESM cache belongs to a different profile")
        atomic_json(self.spec, spec_path)
        DIST_WRAPPER.world_size = 1
        DIST_WRAPPER.rank = 0
        DIST_WRAPPER.local_rank = device.index
        DIST_WRAPPER.local_world_size = 1
        model_configs["protenix_mini_esm_v0.5.0"]["load_strict"] = True
        with patch("runner.batch_inference.download_inference_cache", validate_cached_encoder_assets):
            self.runner = _load_checkpointed_runner(
                model_name="protenix_mini_esm_v0.5.0", seeds=[42], n_cycle=4,
                n_step=1, n_sample=1, dtype="bf16", use_msa=False, use_template=False,
                use_rna_msa=False, trimul_kernel="cuequivariance", triatt_kernel="cuequivariance",
                enable_cache=True, enable_fusion=False,
            )
        self.runner.model.eval().requires_grad_(False)
        _install_empty_template_fast_path()

    def stage(self, records):
        from protenix_embedding_profile import chain_key, load_chain, sequences

        staged = 0
        started = time.monotonic()
        for sequence in sorted({sequence for record in records for sequence in sequences(record)}):
            key = chain_key(sequence)
            metadata_path = self.cache / "metadata" / f"{key}.json"
            tensor_path = self.cache / "embeddings/chains" / f"{key}.pt"
            expected = {"sequence": sequence, "spec": self.spec}
            if metadata_path.exists() and tensor_path.exists():
                if json.loads(metadata_path.read_text()) != expected:
                    raise ValueError(f"Local chain metadata mismatch: {key}")
                continue
            tensor = load_chain(self.args.esm_cache_dir, sequence, self.spec)
            temporary = tensor_path.with_name(f"{key}.{os.getpid()}.tmp")
            torch.save(tensor, temporary)
            os.replace(temporary, tensor_path)
            atomic_json(expected, metadata_path)
            staged += 1
            if staged % 16 == 0:
                print(json.dumps({"event": "stage_chains", "device": str(self.device),
                                  "new_chains": staged, "seconds": time.monotonic() - started}), flush=True)
        print(json.dumps({"event": "stage_complete", "device": str(self.device),
                          "new_chains": staged, "seconds": time.monotonic() - started}), flush=True)

    def batches(self, frame):
        from protenix.data.inference.infer_dataloader import get_inference_dataloader
        from protenix_embedding_profile import cached_esm_dataloader, stabilize_esm_paths
        from runner.batch_inference import preprocess_input

        records = [record_for(row) for row in frame.itertuples(index=False)]
        self.stage(records)
        source = self.work / "input.json"
        atomic_json(records, source)
        prepared = preprocess_input(str(source), out_dir=str(self.work), use_msa=False,
                                    use_template=False, msa_server_mode="protenix")
        self.runner.configs.input_json_path = prepared
        self.runner.configs.num_workers = 0
        with cached_esm_dataloader(self.runner.configs, self.cache, self.work, validate_tensors=False):
            loader = get_inference_dataloader(configs=self.runner.configs)
            stabilize_esm_paths(loader, self.runner.configs)
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if self.args.cpu_workers else {}
        return torch.utils.data.DataLoader(
            DeterministicFeatures(loader.dataset), batch_size=1, collate_fn=loader.collate_fn,
            num_workers=self.args.cpu_workers, generator=torch.Generator().manual_seed(42),
            worker_init_fn=initialize_worker, **options,
        )

    @torch.no_grad()
    def infer(self, batch, expected_id):
        from protenix.model.protenix import update_input_feature_dict
        from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
        from runner.inference import update_inference_configs

        data, _, error = batch[0]
        if error or data["sample_name"] != expected_id:
            raise RuntimeError(f"Featurization failed for {expected_id}: {error}")
        data = to_device(data, self.device)
        self.runner.update_model_configs(update_inference_configs(self.runner.configs, int(data["N_token"].item())))
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.random.default_generator.manual_seed(42)
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = update_input_feature_dict(
                    self.runner.model.relative_position_encoding.generate_relp(data["input_feature_dict"]))
                s_inputs, single, pair = self.runner.model.get_pairformer_output(
                    input_feature_dict=features, N_cycle=4, inplace_safe=True,
                    chunk_size=self.runner.model.configs.infer_setting.chunk_size)
                probability = autocasting_disable_decorator(True)(torch.softmax)(
                    self.runner.model.distogram_head(pair), dim=-1)
                d_geom = None
                if self.args.use_d_geom:
                    from foldabs_repro.tmp_prediction.geometry import sample_foldabs_specificity_features
                    d_geom, p_pae, iptm = sample_foldabs_specificity_features(
                        self.runner.model, features, s_inputs, single, pair,
                        steps=self.args.diffusion_steps, samples=self.args.diffusion_samples,
                    )
                else:
                    p_pae = iptm = None
        # Online consumers use these tensors immediately; avoid an FP16 round trip.
        result = (single, pair, probability, features["asym_id"].long())
        return (*result, d_geom, p_pae, iptm) if d_geom is not None else result


def forward(head, features):
    if len(features) == 4:
        single, pair, probability, asym = features
        d_geom = p_pae = iptm = None
    elif len(features) == 7:
        single, pair, probability, asym, d_geom, p_pae, iptm = features
    else:
        raise ValueError("Online TMP features must have 4 or 7 entries")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return head(single.float(), pair.float(), probability.float(), asym,
                    REQUIRED_ROLES,
                    d_geom=None if d_geom is None else d_geom.float(),
                    p_pae=None if p_pae is None else p_pae.float(),
                    iptm=iptm).float()


@torch.no_grad()
def evaluate(head, provider, frame, args, rank, world_size, split):
    head.eval()
    local = frame.iloc[rank::world_size].reset_index(drop=True)
    rows = []
    for start in range(0, len(local), args.chunk_rounds):
        chunk = local.iloc[start:start + args.chunk_rounds]
        for row, batch in zip(chunk.itertuples(index=False), provider.batches(chunk), strict=True):
            features = provider.infer(batch, row.sample_id)
            logit = forward(head, features)
            loss = F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, row.label))
            rows.append({"sample_id": row.sample_id, "pmhc_id": row.pmhc_id, "label": row.label,
                         "logit": logit.item(), "bce": loss.item()})
            del features, logit, loss
        print(json.dumps({"event": "evaluation_progress", "rank": rank, "split": split,
                          "done": len(rows), "total": len(local)}), flush=True)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, rows)
    result = pd.DataFrame([row for shard in gathered for row in shard])
    if len(result) != len(frame) or result.sample_id.duplicated().any():
        raise ValueError("Evaluation sharding lost or duplicated records")
    metrics = {"bce": float(result.bce.mean()), **binary_metrics(result.label, result.logit),
               **macro_pmhc_metrics(result)}
    if not np.isfinite(metrics["bce"]):
        raise FloatingPointError("Non-finite evaluation BCE")
    if rank == 0:
        result.to_csv(args.output_dir / f"{split}_predictions.csv", index=False)
        atomic_json(metrics, args.output_dir / f"{split}_metrics.json")
    head.train()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--esm-cache-dir", type=Path, required=True)
    parser.add_argument("--local-esm-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--chunk-rounds", type=int, default=96)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-sampling-seed", type=int, default=42)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--smoke-evaluation-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--loss-type", choices=("bce", "group_rank"), default="bce")
    parser.add_argument("--group-negatives", type=int, default=2)
    parser.add_argument("--group-margin", type=float, default=1.0)
    parser.add_argument("--group-accumulation-steps", type=int, default=1)
    parser.add_argument("--group-require-full", action="store_true")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-d-geom", action="store_true",
                        help="Enable FoldAbS specificity features: d_geom + PAE conditioning + iPTM fusion")
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--diffusion-samples", type=int, default=32)
    parser.add_argument("--allow-execution-config-change", action="store_true",
                        help="Allow and record precision/recomputation changes when resuming")
    args = parser.parse_args()
    if args.diffusion_steps < 1 or args.diffusion_samples < 1:
        parser.error("diffusion steps/samples must be positive")
    if args.group_negatives < 1 or args.group_accumulation_steps < 1 or not math.isfinite(args.group_margin):
        parser.error("Group counts must be positive and margin finite")
    if args.loss_type == "group_rank":
        args.gradient_accumulation = args.group_accumulation_steps
    if args.loss_type == "group_rank" and 0 < args.smoke_evaluation_samples < 6:
        parser.error("group smoke evaluation needs at least six records")
    for name in ("splits_dir", "esm_cache_dir", "local_esm_cache", "output_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if min(args.epochs, args.gradient_accumulation, args.chunk_rounds,
           args.checkpoint_every, args.log_every, args.early_stop_patience) < 1:
        parser.error("Counts must be positive")
    if args.chunk_rounds % args.gradient_accumulation:
        parser.error("chunk-rounds must be divisible by gradient-accumulation")
    if args.smoke_samples and args.smoke_samples < 3:
        parser.error("smoke-samples must be zero or at least three")
    if args.smoke_evaluation_samples and (not args.smoke_samples or args.smoke_evaluation_samples < 3):
        parser.error("smoke-evaluation-samples requires smoke mode and at least three samples")
    if min(args.cpu_workers, args.max_steps, args.smoke_samples) < 0:
        parser.error("Workers and optional limits must be nonnegative")
    if args.local_esm_cache == args.esm_cache_dir:
        parser.error("Keep source cache read-only; use a separate local cache")
    return args


def main():
    args = parse_args()
    if args.loss_type == "group_rank":
        from train_mini_esm_group_rank import train
        return train(args)
    rank, local_rank, world_size = (int(os.environ.get(name, default)) for name, default in
                                   (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPUs are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        run_lock = (args.output_dir / "run.lock").open("a")
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and (args.output_dir / "last.pt").exists():
        raise FileExistsError("Existing checkpoint: use --resume or a fresh output directory")
    set_seed(args.seed)
    sampling = PeptideMismatchSampling(args)
    fingerprint = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(args.splits_dir.iterdir()) if path.suffix in (".npy", ".csv")}
    fingerprint["esm_spec"] = hashlib.sha256((args.esm_cache_dir / "run_spec.json").read_bytes()).hexdigest()
    config = {**vars(args), "world_size": world_size, "effective_batch_size": world_size * args.gradient_accumulation,
              "checkpoint_policy": "retain_every_epoch",
              "epoch_checkpoint_template": "bs{batch_size}_lr_{learning_rate}sd_{seed}_ep{epoch}.pt",
              "hostname": os.uname().nodename, "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "input_fingerprint": fingerprint, "encoder_cycles": 4, "head_blocks": 4,
              "head_pooling": "C", "precision": "bf16", "joint_feature_disk_cache": False,
              "activation_checkpoint": args.activation_checkpoint, "feature_handoff": "native",
              "kernels": "cuequivariance", "ordering": "seed+epoch shuffle, masked tail",
              "triangle_attention_mask_policy": "omit_all_valid_boolean_masks",
              "diffusion_feature": "d_geom" if args.use_d_geom else None,
              "geometry_bins": 64 if args.use_d_geom else None,
              "geometry_max_angstrom": 22.0 if args.use_d_geom else None,
              "pae_bins": 64 if args.use_d_geom else None,
              "iptm_embedding_dim": 32 if args.use_d_geom else None,
              "confidence_executed": bool(args.use_d_geom),
              "specificity_feature_version": ("foldabs_d_geom_pae_iptm_v1" if args.use_d_geom else "trunk_p_dist_v1")}
    provider = OnlineFeatures(args, rank, device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        head = TMPTrunkSpecificityHead(c_s=384, c_z=128, c_dist=64, pairformer_blocks=4,
            pooling_mode="C", hidden_dim=512, dropout=0.1, triangle_multiplicative="cuequivariance",
            triangle_attention="cuequivariance", activation_checkpoint=args.activation_checkpoint,
            use_d_geom=args.use_d_geom).to(device)
        ddp = DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)
        epoch, cursor, step, best, bad_epochs, history = 1, 0, 0, float("inf"), 0, []
        running = [0.0, 0.0]
        resumed_rng = None
        if args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            record_execution_change(saved, config, args.allow_execution_config_change)
            for key in ("world_size", "input_fingerprint", "seed", "negative_sampling_seed", "negatives_per_positive",
                        "gradient_accumulation", "learning_rate", "weight_decay", "smoke_samples", "early_stop_patience"):
                if saved["run_config"][key] != config[key]:
                    raise ValueError(f"Resume configuration mismatch: {key}")
            for key, default in (("use_d_geom", False), ("diffusion_steps", 20), ("diffusion_samples", 32)):
                if saved["run_config"].get(key, default) != config[key]:
                    raise ValueError(f"Resume diffusion configuration mismatch: {key}")
            head.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            epoch, cursor, step = saved["epoch"], saved["cursor"], saved["step"]
            if args.max_steps and args.max_steps <= step:
                raise ValueError("Resume max-steps must exceed the saved optimizer step")
            best, bad_epochs, history = saved["best_bce"], saved["bad_epochs"], saved["history"]
            running = saved["running_by_rank"][rank]
            resumed_rng = saved["rng_by_rank"][rank]
            if args.resume.parent != args.output_dir:
                raise ValueError("Resume in the original output directory to retain best.pt")
        set_seed(args.seed + rank)
        if resumed_rng is not None:
            restore_rng(resumed_rng)
        if rank == 0:
            atomic_json(config, args.output_dir / "run_config.json")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()

        def save(best_checkpoint=False, completed_epoch=None):
            states = [None] * world_size
            dist.all_gather_object(states, {"rng": capture_rng(), "running": running})
            if rank == 0:
                value = {"model": head.state_dict(), "model_config": head.config, "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "run_config": config, "epoch": epoch, "cursor": cursor,
                         "step": step, "best_bce": best, "bad_epochs": bad_epochs, "history": history,
                         "rng_by_rank": [state["rng"] for state in states],
                         "running_by_rank": [state["running"] for state in states]}
                if completed_epoch is not None:
                    value["checkpoint_stage"] = "training_complete_validation_pending"
                    archive_epoch(value, args.output_dir, batch_size=nominal_batch_size(args, world_size),
                                  learning_rate=args.learning_rate, seed=args.seed, epoch=completed_epoch)
                atomic_checkpoint(value, args.output_dir / "last.pt")
                if best_checkpoint:
                    atomic_checkpoint(value, args.output_dir / "best.pt")
                atomic_json({"epoch": epoch, "next_round": cursor, "step": step,
                             "elapsed_seconds": time.monotonic() - started}, args.output_dir / "progress.json")
            dist.barrier()

        while epoch <= args.epochs and bad_epochs < args.early_stop_patience:
            frame, report = sampling.epoch(epoch)
            frame = epoch_order(frame, args.seed, epoch, args.smoke_samples)
            if rank == 0:
                atomic_json({**report, "executed_samples": len(frame), "smoke": bool(args.smoke_samples)},
                            args.output_dir / f"epoch_{epoch:03d}_sampling.json")
                print(json.dumps({"event": "epoch_start", "epoch": epoch, "samples": len(frame),
                                  "positives": int(frame.label.sum()), "negatives": int(frame.label.eq(0).sum()),
                                  "world_size": world_size, "resume_round": cursor}), flush=True)
            total_rounds = math.ceil(len(frame) / world_size)
            head.train()
            while cursor < total_rounds:
                plan = rank_plan(len(frame), world_size, rank, cursor, args.chunk_rounds)
                chunk = frame.iloc[[index for index, _ in plan]]
                iterator = iter(provider.batches(chunk))
                chunk_start = cursor
                for offset, (row, (_, valid)) in enumerate(zip(chunk.itertuples(index=False), plan, strict=True)):
                    round_index = chunk_start + offset
                    window_start = round_index // args.gradient_accumulation * args.gradient_accumulation
                    window_stop = min(window_start + args.gradient_accumulation, total_rounds)
                    sync = round_index + 1 == window_stop
                    if round_index == window_start:
                        optimizer.zero_grad(set_to_none=True)
                    features = provider.infer(next(iterator), row.sample_id)
                    with nullcontext() if sync else ddp.no_sync():
                        logit = forward(ddp, features)
                        loss = F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, row.label))
                        scaled = loss * (world_size / window_count(len(frame), world_size, window_start, window_stop)) * int(valid)
                        scaled.backward()
                    running[0] += loss.item() * int(valid)
                    running[1] += int(valid)
                    del features, logit, loss, scaled
                    if sync:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                        optimizer.step()
                        step += 1
                        cursor = round_index + 1
                        if step == 1 or step % args.log_every == 0:
                            stats = torch.tensor([*running, torch.cuda.max_memory_reserved() / 2**20], device=device)
                            aggregate = stats.clone()
                            dist.all_reduce(aggregate)
                            checks = torch.stack([parameter.detach().double().sum() for parameter in head.parameters()]).sum()
                            checksum_list = [torch.zeros_like(checks) for _ in range(world_size)]
                            dist.all_gather(checksum_list, checks)
                            if not all(torch.equal(checksum_list[0], value) for value in checksum_list[1:]):
                                raise RuntimeError("DDP head parameter checksums differ")
                            print(json.dumps({"event": "train", "rank": rank, "epoch": epoch, "step": step,
                                "round": cursor, "train_bce": (aggregate[0] / aggregate[1]).item(),
                                "valid_samples": int(aggregate[1].item()), "gradient_norm": gradient_norm.item(),
                                "peak_reserved_mib": stats[2].item(), "ddp_checksum_equal": True,
                                "elapsed_seconds": time.monotonic() - started}), flush=True)
                        if step == 1 or step % args.checkpoint_every == 0 or (args.max_steps and step >= args.max_steps):
                            save(completed_epoch=epoch if cursor == total_rounds else None)
                        if args.max_steps and step >= args.max_steps:
                            return
                del iterator
            save(completed_epoch=epoch)
            if args.smoke_samples:
                save()
                if args.smoke_evaluation_samples:
                    for split in ("val", "test"):
                        selected = epoch_order(sampling.manifest(sampling.frames[split], split), args.seed,
                                               epoch, args.smoke_evaluation_samples)
                        evaluate(head, provider, selected, args, rank, world_size, f"smoke_{split}")
                if rank == 0:
                    atomic_json({"status": "smoke_complete", "step": step, "samples": len(frame)},
                                args.output_dir / "completion.json")
                return
            validation = sampling.manifest(sampling.frames["val"], "val")
            metrics = evaluate(head, provider, validation, args, rank, world_size, "validation")
            totals = torch.tensor(running, device=device)
            dist.all_reduce(totals)
            history.append({"epoch": epoch, "train_bce": (totals[0] / totals[1]).item(), "validation": metrics})
            improved = metrics["bce"] < best
            best, bad_epochs = (metrics["bce"], 0) if improved else (best, bad_epochs + 1)
            scheduler.step(metrics["bce"])
            epoch, cursor, running = epoch + 1, 0, [0.0, 0.0]
            save(best_checkpoint=improved)
            if rank == 0:
                atomic_json(history, args.output_dir / "history.json")
        best_checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(best_checkpoint["model"])
        evaluate(head, provider, sampling.manifest(sampling.frames["test"], "test"), args, rank, world_size, "test")
        if rank == 0:
            atomic_json({"status": "complete", "step": step, "epochs": epoch - 1, "best_bce": best},
                        args.output_dir / "completion.json")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
