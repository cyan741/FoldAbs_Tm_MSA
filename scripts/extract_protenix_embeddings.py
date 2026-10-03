#!/usr/bin/env python3
"""Extract frozen Protenix-v1 Pairformer representations for TMP inputs.

This intentionally stops after the Pairformer trunk. It does not run diffusion
structure sampling. Stage 1 caches only the tensors consumed by the TMP head:
s, z, p_dist, token_asym_id and token_residue_index. s_inputs is deliberately
not stored because the current/follow-up TMP experiments do not consume it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROTENIX_ROOT = Path(__file__).resolve().parents[1] / ".foldtmp-run" / "protenix_assets"
os.environ.setdefault("PROTENIX_ROOT_DIR", str(DEFAULT_PROTENIX_ROOT))
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch

from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import update_input_feature_dict
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from runner.batch_inference import get_default_runner, preprocess_input
from runner.inference import update_inference_configs


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(dtype=torch.float16, device="cpu").contiguous()


REQUIRED_CACHE_KEYS = {
    "token_asym_id", "token_residue_index", "s", "z", "p_dist"
}


def _cached_index_row(
    output_path: Path,
    *,
    sample_name: str,
    input_index: int,
) -> dict[str, object]:
    """Validate one existing cache without rewriting it and return metadata."""
    existing = torch.load(output_path, map_location="cpu", weights_only=True)
    if not REQUIRED_CACHE_KEYS.issubset(existing):
        raise ValueError(f"Cannot resume from incomplete feature file: {output_path}")
    return {
        "input_index": input_index,
        "sample_id": sample_name,
        "file": output_path.name,
        "n_token": int(existing["s"].shape[0]),
        "file_bytes": int(output_path.stat().st_size),
        "s_shape": list(existing["s"].shape),
        "z_shape": list(existing["z"].shape),
        "p_dist_shape": list(existing["p_dist"].shape),
    }


def _prepare_worker_input(
    input_path: Path,
    work_dir: Path,
    output_dir: Path,
    *,
    num_shards: int,
    shard_index: int,
    limit: int,
    resume: bool,
) -> tuple[Path, list[int], list[dict[str, object]]]:
    """Validate cached records and write only pending records for featurization."""
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of records in {input_path}")
    if limit:
        records = records[:limit]

    pending_records: list[dict[str, object]] = []
    pending_indices: list[int] = []
    cached_rows: list[dict[str, object]] = []
    selected_count = 0
    for input_index, record in enumerate(records):
        if input_index % num_shards != shard_index:
            continue
        selected_count += 1
        if not isinstance(record, dict) or not str(record.get("name", "")):
            raise ValueError(f"Input record {input_index} has no valid name")
        sample_name = str(record["name"])
        output_path = output_dir / f"{sample_name}.pt"
        if resume and output_path.is_file():
            cached_rows.append(
                _cached_index_row(
                    output_path,
                    sample_name=sample_name,
                    input_index=input_index,
                )
            )
        else:
            pending_records.append(record)
            pending_indices.append(input_index)

    shard_path = work_dir / f"input_shard_{shard_index:02d}_of_{num_shards:02d}.json"
    shard_path.write_text(json.dumps(pending_records), encoding="utf-8")
    print(
        f"Shard {shard_index}: selected={selected_count}, "
        f"reused={len(cached_rows)}, pending={len(pending_records)}",
        flush=True,
    )
    return shard_path, pending_indices, cached_rows


def _resolve_gpu_ids(value: str) -> list[str]:
    """Resolve a comma-separated GPU list or ``all`` for worker processes."""
    if value.strip().lower() == "all":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            gpu_ids = [str(index) for index in range(torch.cuda.device_count())]
    else:
        gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids did not select any GPUs")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpu-ids contains duplicate devices: {gpu_ids}")
    return gpu_ids


def _aggregate_shards(
    output_dir: Path,
    *,
    num_shards: int,
    gpu_ids: list[str],
    total_wall_seconds: float,
) -> None:
    """Merge successful worker metadata into the single-process filenames."""
    rows: list[dict[str, object]] = []
    shard_stats: list[dict[str, object]] = []
    for shard_index in range(num_shards):
        suffix = f".shard_{shard_index:02d}_of_{num_shards:02d}"
        rows.extend(json.loads((output_dir / f"index{suffix}.json").read_text()))
        shard_stats.append(
            json.loads((output_dir / f"extraction_stats{suffix}.json").read_text())
        )

    rows.sort(key=lambda row: int(row["input_index"]))
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Shard outputs contain duplicate sample IDs")

    total_bytes = sum(int(row.get("file_bytes", 0)) for row in rows)
    generated_count = sum(int(stats["samples_generated"]) for stats in shard_stats)
    reused_count = sum(int(stats["samples_reused"]) for stats in shard_stats)
    generated_seconds = sum(float(stats["generated_seconds"]) for stats in shard_stats)
    peak_by_gpu = {
        gpu_id: shard_stats[index].get("gpu_peak_memory_bytes")
        for index, gpu_id in enumerate(gpu_ids)
    }
    numeric_peaks = [int(value) for value in peak_by_gpu.values() if value is not None]
    stats = {
        "samples_total": len(rows),
        "samples_generated": generated_count,
        "samples_reused": reused_count,
        "mean_n_token": (
            sum(int(row["n_token"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "total_cache_bytes": total_bytes,
        "mean_cache_bytes": total_bytes / len(rows) if rows else 0.0,
        "generated_seconds": generated_seconds,
        "generated_seconds_per_sample": (
            generated_seconds / generated_count if generated_count else None
        ),
        "total_wall_seconds": total_wall_seconds,
        "gpu_peak_memory_bytes": max(numeric_peaks) if numeric_peaks else None,
        "gpu_peak_memory_bytes_by_device": peak_by_gpu,
        "use_msa": bool(shard_stats[0]["use_msa"]),
        "cycles": int(shard_stats[0]["cycles"]),
        "num_shards": num_shards,
        "shard_index": None,
        "gpu_ids": gpu_ids,
        "parallel_workers": num_shards,
    }
    (output_dir / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "extraction_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"Merged {len(rows)} representations from {num_shards} GPUs", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def _run_multi_gpu(args: argparse.Namespace, gpu_ids: list[str]) -> None:
    """Launch one isolated extraction worker per GPU and merge their metadata."""
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError(
            "--gpu-ids cannot be combined with explicit --num-shards/--shard-index"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    processes: list[tuple[str, subprocess.Popen]] = []
    run_start = time.perf_counter()
    try:
        for shard_index, gpu_id in enumerate(gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--multi-gpu-worker",
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
            ]
            print(
                f"Launching shard {shard_index}/{num_shards - 1} on GPU {gpu_id}",
                flush=True,
            )
            processes.append((gpu_id, subprocess.Popen(command, env=env)))

        pending = set(range(len(processes)))
        while pending:
            for process_index in list(pending):
                gpu_id, process = processes[process_index]
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(process_index)
                if return_code != 0:
                    raise RuntimeError(
                        f"Extraction worker on GPU {gpu_id} exited with code {return_code}"
                    )
            if pending:
                time.sleep(0.2)
    except BaseException:
        for _gpu_id, process in processes:
            if process.poll() is None:
                process.terminate()
        for _gpu_id, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise

    _aggregate_shards(
        output_dir,
        num_shards=num_shards,
        gpu_ids=gpu_ids,
        total_wall_seconds=time.perf_counter() - run_start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen Protenix-v1 s, z, and p_dist tensors."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--use-msa", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help=(
            "Run one extraction worker per selected GPU, for example '0,1,3' or "
            "'all'. Device IDs are passed through CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split records deterministically across this many extraction processes.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard handled by this process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse valid per-sample .pt files after an interrupted run. "
            "Existing valid cache files are validated but never rewritten."
        ),
    )
    parser.add_argument(
        "--msa-server-mode", choices=["protenix", "colabfold"], default="protenix"
    )
    parser.add_argument(
        "--multi-gpu-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    if args.gpu_ids is not None and not args.multi_gpu_worker:
        _run_multi_gpu(args, _resolve_gpu_ids(args.gpu_ids))
        return

    protenix_root = Path(os.environ["PROTENIX_ROOT_DIR"]).resolve()
    checkpoint = protenix_root / "checkpoint" / "protenix_base_default_v1.0.0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Protenix checkpoint not found at {checkpoint}. "
            "Set PROTENIX_ROOT_DIR to the directory containing checkpoint/ and common/."
        )
    print(f"PROTENIX_ROOT_DIR={protenix_root}", flush=True)

    input_path = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        "protenix_input_cache"
        if args.num_shards == 1
        else f"protenix_input_cache_shard_{args.shard_index:02d}"
    )
    work_dir = output_dir / cache_name
    work_dir.mkdir(exist_ok=True)

    run_start = time.perf_counter()
    worker_input_path, pending_input_indices, cached_rows = _prepare_worker_input(
        input_path,
        work_dir,
        output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
        resume=args.resume,
    )
    prepared_json = preprocess_input(
        str(worker_input_path),
        out_dir=str(work_dir),
        use_msa=args.use_msa,
        use_template=False,
        msa_server_mode=args.msa_server_mode,
    )
    runner = get_default_runner(
        seeds=[101],
        n_cycle=args.cycles,
        n_step=1,
        n_sample=1,
        dtype="bf16",
        use_msa=args.use_msa,
        trimul_kernel="torch",
        triatt_kernel="torch",
        enable_cache=True,
        enable_fusion=False,
        use_template=False,
    )
    configs = runner.configs
    configs.input_json_path = prepared_json
    dataloader = get_inference_dataloader(configs=configs)

    index_rows: list[dict[str, object]] = cached_rows
    generated_count = 0
    reused_count = len(cached_rows)
    generated_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for local_item_index, batch in enumerate(dataloader):
        input_index = pending_input_indices[local_item_index]
        data, _atom_array, data_error_message = batch[0]
        sample_name = str(data["sample_name"])
        if data_error_message:
            raise RuntimeError(f"Input preparation failed for {sample_name}: {data_error_message}")

        output_path = output_dir / f"{sample_name}.pt"
        if args.resume and output_path.is_file():
            reused_count += 1
            index_rows.append(
                _cached_index_row(
                    output_path,
                    sample_name=sample_name,
                    input_index=input_index,
                )
            )
            print(f"[{input_index + 1}] reused {output_path.name}", flush=True)
            continue

        sample_start = time.perf_counter()
        data = to_device(data, runner.device)
        token_count = int(data["N_token"].item())
        runner.update_model_configs(update_inference_configs(configs, token_count))
        features = data["input_feature_dict"]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = runner.model.relative_position_encoding.generate_relp(features)
            features = update_input_feature_dict(features)
            _s_inputs, s, z = runner.model.get_pairformer_output(
                input_feature_dict=features,
                N_cycle=args.cycles,
                inplace_safe=True,
                chunk_size=runner.model.configs.infer_setting.chunk_size,
            )
            p_dist = autocasting_disable_decorator(True)(torch.softmax)(
                runner.model.distogram_head(z), dim=-1
            )

        torch.save(
            {
                "sample_name": sample_name,
                "token_asym_id": _cpu_tensor(features["asym_id"]),
                "token_residue_index": _cpu_tensor(features["residue_index"]),
                "s": _cpu_tensor(s),
                "z": _cpu_tensor(z),
                "p_dist": _cpu_tensor(p_dist),
            },
            output_path,
        )
        elapsed = time.perf_counter() - sample_start
        generated_count += 1
        generated_seconds += elapsed

        index_rows.append(
            {
                "input_index": input_index,
                "sample_id": sample_name,
                "file": output_path.name,
                "n_token": token_count,
                "file_bytes": int(output_path.stat().st_size),
                "seconds": elapsed,
                "s_shape": list(s.shape),
                "z_shape": list(z.shape),
                "p_dist_shape": list(p_dist.shape),
            }
        )
        del _s_inputs, s, z, p_dist, features, data
        torch.cuda.empty_cache()
        print(
            f"[{input_index + 1}] saved {output_path.name} "
            f"({elapsed:.2f}s, {output_path.stat().st_size / 1024**2:.2f} MiB)",
            flush=True,
        )

    metadata_suffix = (
        "" if args.num_shards == 1
        else f".shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    index_path = output_dir / f"index{metadata_suffix}.json"
    index_path.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")

    total_seconds = time.perf_counter() - run_start
    total_bytes = sum(int(row.get("file_bytes", 0)) for row in index_rows)
    mean_tokens = (
        sum(int(row["n_token"]) for row in index_rows) / len(index_rows)
        if index_rows else 0.0
    )
    stats = {
        "samples_total": len(index_rows),
        "samples_generated": generated_count,
        "samples_reused": reused_count,
        "mean_n_token": mean_tokens,
        "total_cache_bytes": total_bytes,
        "mean_cache_bytes": total_bytes / len(index_rows) if index_rows else 0.0,
        "generated_seconds": generated_seconds,
        "generated_seconds_per_sample": (
            generated_seconds / generated_count if generated_count else None
        ),
        "total_wall_seconds": total_seconds,
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "use_msa": bool(args.use_msa),
        "cycles": int(args.cycles),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    stats_path = output_dir / f"extraction_stats{metadata_suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Extracted {len(index_rows)} representations; index: {index_path}")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
