#!/usr/bin/env python3
"""Time no-write cycle-4 negative inference and a resident Pairformer head.

Only a small JSON input, timing report, and preprocessing files are written to
the local work directory. No inferred s/z/p_dist is copied to CPU or saved.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault("LAYERNORM_TYPE", "torch")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from extract_protenix_embeddings import _install_empty_template_fast_path, _load_checkpointed_runner
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import update_input_feature_dict
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from runner.batch_inference import preprocess_input
from runner.inference import update_inference_configs

from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def gpu_free_mib() -> int:
    return int(torch.cuda.mem_get_info()[0] / 1024**2)


def timed_gpu(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def choose_records(path: Path, samples: int, offset: int) -> list[dict]:
    # The source is a 112 MB local metadata file. Spacing samples through it
    # avoids benchmarking only one TCR/HLA group.
    records = json.loads(path.read_text())
    if samples > len(records):
        raise ValueError("Requested more samples than the input contains")
    # Use half-open bins: the last point never wraps onto the first bin. Small
    # offsets (for example 0, 6, 7 across three GPUs) remain disjoint.
    indices = [((i * len(records)) // samples + offset) % len(records)
               for i in range(samples)]
    picked = [records[index] for index in indices]
    if len({record["name"] for record in picked}) != samples:
        raise ValueError("Selected duplicate sample names")
    return picked


def enable_activation_checkpoint(model: TMPTrunkSpecificityHead) -> None:
    # Benchmark-only wrapper. Preserves the model math and parameters.
    for block in model.pairformer:
        original = block.forward

        def wrapped(*, s, z, pair_mask, triangle_multiplicative,
                    triangle_attention, inplace_safe, chunk_size, _original=original):
            return checkpoint(
                lambda a, b: _original(
                    s=a, z=b, pair_mask=pair_mask,
                    triangle_multiplicative=triangle_multiplicative,
                    triangle_attention=triangle_attention,
                    inplace_safe=inplace_safe, chunk_size=chunk_size,
                ),
                s, z, use_reentrant=False,
            )

        block.forward = wrapped


def run(args: argparse.Namespace) -> dict:
    run_start = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    records = choose_records(args.input_json, args.samples, args.sample_offset)
    source_path = args.work_dir / "sample_input.json"
    source_path.write_text(json.dumps(records))

    device = torch.device("cuda:0")
    info: dict = {
        "gpu_name": torch.cuda.get_device_name(device),
        "input_json": str(args.input_json),
        "sample_ids": [record["name"] for record in records],
        "initial_free_mib": gpu_free_mib(),
        "activation_checkpoint": args.activation_checkpoint,
        "cycles": 4,
        "samples": [],
        "sample_offset": args.sample_offset,
        "negative_features_written": 0,
        "triangle_multiplicative": args.trimul_kernel,
        "triangle_attention": args.triatt_kernel,
    }
    print(f"initial free {info['initial_free_mib']} MiB; loading frozen Protenix", flush=True)

    prepared_path = preprocess_input(
        str(source_path), out_dir=str(args.work_dir), use_msa=False,
        use_template=False, msa_server_mode="protenix",
    )
    runner = _load_checkpointed_runner(
        seeds=[101], n_cycle=4, n_step=1, n_sample=1, dtype="bf16",
        use_msa=False, trimul_kernel=args.trimul_kernel,
        triatt_kernel=args.triatt_kernel,
        enable_cache=True, enable_fusion=False, use_template=False,
    )
    runner.model.eval().requires_grad_(False)
    runner.configs.input_json_path = prepared_path
    runner.configs.num_workers = 0
    _install_empty_template_fast_path()
    dataloader = get_inference_dataloader(configs=runner.configs)
    iterator = iter(dataloader)

    # Construct the head before resetting peaks, so every measured forward
    # includes both resident models and the optimizer state on the same card.
    head = TMPTrunkSpecificityHead(
        c_s=384, c_z=128, c_dist=64, pairformer_blocks=4,
        pooling_mode="C", hidden_dim=512, dropout=0.1,
    ).to(device)
    if args.activation_checkpoint:
        enable_activation_checkpoint(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    info["resident_models_free_mib"] = gpu_free_mib()
    print(f"both models resident; free {info['resident_models_free_mib']} MiB", flush=True)

    for index, record in enumerate(records):
        row_start = time.perf_counter()
        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        cpu_started = time.perf_counter()
        batch = next(iterator)
        cpu_seconds = time.perf_counter() - cpu_started
        data, _, error = batch[0]
        if error:
            raise RuntimeError(f"Featurization failed for {record['name']}: {error}")
        if data["sample_name"] != record["name"]:
            raise ValueError("Dataloader order changed")
        transfer_start = time.perf_counter()
        data = to_device(data, runner.device)
        n_token = int(data["N_token"].item())
        runner.update_model_configs(update_inference_configs(runner.configs, n_token))
        transfer_and_update_seconds = time.perf_counter() - transfer_start
        free_before = gpu_free_mib()
        torch.cuda.reset_peak_memory_stats()

        def infer(input_features=None):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = runner.model.relative_position_encoding.generate_relp(
                    data["input_feature_dict"] if input_features is None else input_features
                )
                features = update_input_feature_dict(features)
                _s_inputs, s, z = runner.model.get_pairformer_output(
                    input_feature_dict=features, N_cycle=4, inplace_safe=True,
                    chunk_size=runner.model.configs.infer_setting.chunk_size,
                )
                p = autocasting_disable_decorator(True)(torch.softmax)(
                    runner.model.distogram_head(z), dim=-1,
                )
            # Match the existing fp16 cache semantics without touching CPU/disk.
            return {
                "s": s.detach().to(torch.float16).contiguous(),
                "z": z.detach().to(torch.float16).contiguous(),
                "p_dist": p.detach().to(torch.float16).contiguous(),
                "token_asym_id": features["asym_id"].detach().to(torch.int64),
            }

        if index == 0 and args.compare_torch:
            # Isolate in-place feature updates and restore RNG so the two
            # paths receive identical input and stochastic state.
            reference_input = copy.deepcopy(data["input_feature_dict"])
            accelerated_input = copy.deepcopy(data["input_feature_dict"])
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state()
            runner.model.configs.triangle_multiplicative = "torch"
            runner.model.configs.triangle_attention = "torch"
            reference, reference_seconds = timed_gpu(lambda: infer(reference_input))
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            runner.model.configs.triangle_multiplicative = args.trimul_kernel
            runner.model.configs.triangle_attention = args.triatt_kernel
            feature, inference_seconds = timed_gpu(lambda: infer(accelerated_input))
        else:
            reference = None
            feature, inference_seconds = timed_gpu(infer)
        infer_peak = int(torch.cuda.max_memory_allocated() / 1024**2)
        free_after_infer = gpu_free_mib()
        row = {
            "sample_id": record["name"],
            "tokens": n_token,
            "cpu_featurization_seconds": round(cpu_seconds, 3),
            "transfer_and_update_seconds": round(transfer_and_update_seconds, 3),
            "gpu_inference_seconds": round(inference_seconds, 3),
            "inference_peak_allocated_mib": infer_peak,
            "inference_peak_reserved_mib": int(torch.cuda.max_memory_reserved() / 1024**2),
            "free_before_inference_mib": free_before,
            "free_after_inference_mib": free_after_infer,
            "feature_bytes": sum(t.numel() * t.element_size() for t in feature.values()),
        }
        if reference is not None:
            row["torch_reference_inference_seconds"] = round(reference_seconds, 3)
            row["difference_vs_torch"] = {
                name: {
                    "max_abs": float((feature[name].float() - reference[name].float()).abs().max().item()),
                    "mean_abs": float((feature[name].float() - reference[name].float()).abs().mean().item()),
                    "reference_rms": float(reference[name].float().square().mean().sqrt().item()),
                }
                for name in ("s", "z", "p_dist")
            }
            del reference

        if index < args.train_samples:
            # Matching current dataset: fp16 cache is converted to fp32 before
            # the bf16-autocast trainable head.
            torch.cuda.reset_peak_memory_stats()
            free_before_train = gpu_free_mib()

            def train_step():
                head.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit = head(
                        feature["s"].float(), feature["z"].float(),
                        feature["p_dist"].float(), feature["token_asym_id"],
                        REQUIRED_ROLES,
                    ).float()
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.zeros((), device=device),
                )
                loss.backward()
                optimizer.step()
                return float(loss.detach().item())

            loss, train_seconds = timed_gpu(train_step)
            row.update({
                "head_train_seconds": round(train_seconds, 3),
                "loss": loss,
                "combined_peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                "combined_peak_reserved_mib": int(torch.cuda.max_memory_reserved() / 1024**2),
                "free_before_train_mib": free_before_train,
                "free_after_train_mib": gpu_free_mib(),
            })
        row["sample_wall_seconds"] = round(time.perf_counter() - row_start, 3)
        info["samples"].append(row)
        print(json.dumps(row), flush=True)
        del feature, data

    warm = info["samples"][1:] if len(info["samples"]) > 1 else info["samples"]
    info["summary"] = {
        "cpu_featurization_seconds": summarize([r["cpu_featurization_seconds"] for r in warm]),
        "gpu_inference_seconds": summarize([r["gpu_inference_seconds"] for r in warm]),
        "head_train_seconds": summarize([r["head_train_seconds"] for r in warm if "head_train_seconds" in r]),
        "sample_wall_seconds": summarize([r["sample_wall_seconds"] for r in warm]),
        "max_combined_peak_allocated_mib": max(
            (r["combined_peak_allocated_mib"] for r in info["samples"] if "combined_peak_allocated_mib" in r),
            default=None,
        ),
        "final_free_mib": gpu_free_mib(),
    }
    info["run_wall_seconds"] = round(time.perf_counter() - run_start, 3)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=3)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--trimul-kernel", choices=("torch", "cuequivariance"), default="cuequivariance")
    parser.add_argument("--triatt-kernel", choices=("torch", "cuequivariance", "triattention"), default="cuequivariance")
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.train_samples <= args.samples:
        parser.error("Need 0 <= train-samples <= samples")
    report = run(args)
    report_path = args.work_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
