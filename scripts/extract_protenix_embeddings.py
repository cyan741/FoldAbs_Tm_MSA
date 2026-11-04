#!/usr/bin/env python3
"""Extract frozen Protenix-base or Mini-ESM representations without MSA.

This intentionally stops after the Pairformer trunk. It does not run diffusion
structure sampling. Stage 1 caches only the tensors consumed by the TMP head:
s, z, p_dist, token_asym_id and token_residue_index. s_inputs is deliberately
not stored because the current/follow-up TMP experiments do not consume it.
"""

from __future__ import annotations

import argparse
import fcntl
import numpy as np
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from contextlib import nullcontext
from protenix_embedding_profile import (DTYPES, atomic_save, cached_esm_dataloader,
    esm_spec, load_chain, reject_msa_templates, seed_rng, sequences, stabilize_esm_paths, validate_joint, versions)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foldabs_repro.tmp_prediction.geometry import sample_d_geomt

DEFAULT_PROTENIX_ROOT = Path(__file__).resolve().parents[1] / ".foldtmp-run" / "protenix_assets"
os.environ.setdefault("PROTENIX_ROOT_DIR", str(DEFAULT_PROTENIX_ROOT))
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch

from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import update_input_feature_dict
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from runner.batch_inference import get_default_runner, preprocess_input
from runner.inference import update_inference_configs


def _load_checkpointed_runner(**kwargs):
    """Skip slow SciPy random initialization of weights replaced by the checkpoint.

    Protenix initializes each linear layer with scipy.stats.truncnorm before
    InferenceRunner loads its full checkpoint. The selected model loads strictly,
    so none of these temporary random weights can affect inference.
    """
    import protenix.model.modules.primitives as primitives
    import protenix.model.triangular.layers as layers

    old_primitive = primitives.trunc_normal_init_
    old_layer = layers.trunc_normal_init_

    def skip_temporary_init(weights, *args, **kwargs):
        return None

    try:
        primitives.trunc_normal_init_ = skip_temporary_init
        layers.trunc_normal_init_ = skip_temporary_init
        runner = get_default_runner(**kwargs)
    finally:
        primitives.trunc_normal_init_ = old_primitive
        layers.trunc_normal_init_ = old_layer
    if not runner.configs.load_strict:
        raise ValueError("Skipping initialization requires strict checkpoint loading")
    print("Protenix checkpoint loaded strictly; skipped overwritten random initialization", flush=True)
    return runner


def _install_empty_template_fast_path() -> None:
    """Avoid geometric work for the four fully masked template placeholders.

    Protenix builds distance bins and unit vectors for all-zero positions even
    with use_template=False. Their masks then make every derived value zero.
    Keep the original aatype/position/mask arrays, including their padding.
    """
    from protenix.data.template.template_featurizer import Templates

    original = Templates.as_protenix_dict

    def as_protenix_dict(self):
        if np.any(self.atom_mask):
            return original(self)
        num_templates, num_tokens = self.aatype.shape
        features = self.as_data_dict()
        features.update({
            "template_pseudo_beta_mask": np.zeros((num_templates, num_tokens, num_tokens), dtype=np.float32),
            "template_distogram": np.zeros((num_templates, num_tokens, num_tokens, 39), dtype=np.float32),
            "template_unit_vector": np.zeros((num_templates, num_tokens, num_tokens, 3), dtype=np.float32),
            "template_backbone_frame_mask": np.zeros((num_templates, num_tokens, num_tokens), dtype=np.float32),
        })
        return features

    Templates.as_protenix_dict = as_protenix_dict


def _cpu_tensor(value: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    target = dtype if value.is_floating_point() else torch.int64
    return value.detach().to(dtype=target, device="cpu").contiguous()


REQUIRED_CACHE_KEYS = {
    "token_asym_id", "token_residue_index", "s", "z", "p_dist"
}


BASE_MODEL = 'protenix_base_default_v1.0.0'
MINI_MODEL = 'protenix_mini_esm_v0.5.0'


def _same_profile(left, right):
    # Legacy base jobs installed the same checkpoint at different host paths.
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get('model') == right.get('model') == 'protenix-v1':
        return {k: v for k, v in left.items() if k != 'checkpoint'} == {
            k: v for k, v in right.items() if k != 'checkpoint'}
    return left == right


def _base_spec(args, checkpoint):
    # Preserve the legacy on-disk schema for existing trunk-only caches.
    spec = {'model': 'protenix-v1', 'package_version': version('protenix'),
            'cycles': args.cycles, 'use_msa': False, 'seed': args.seed,
            'sampling_policy': 'fixed_per_record', 'deterministic_algorithms': True,
            'inference_precision': args.dtype, 'cache_precision': args.cache_dtype,
            'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size}
    if args.use_d_geomt:
        spec['diffusion_geometry'] = {'feature': 'd_geomt', 'steps': args.diffusion_steps,
                                     'samples': args.diffusion_samples, 'bins': 64,
                                     'max_angstrom': 22.0, 'atom': 'CA'}
    return spec


def _record_sequences(record, mini):
    if mini:
        return sequences(record)
    # Base extraction also serves the historical antibody inputs (not four chains).
    return [chain['proteinChain']['sequence'] for chain in record['sequences']]


def _validate_geometry(payload, spec):
    geometry_expected = bool(spec.get('diffusion_geometry', {}).get('feature') == 'd_geomt')
    if not geometry_expected:
        return
    value = payload.get('d_geomt')
    n = payload['s'].shape[0]
    if (not isinstance(value, torch.Tensor) or value.shape != (n, n, 64)
            or value.dtype != DTYPES[spec['cache_precision']]):
        raise ValueError('Diffusion d_geomt missing or invalid shape/dtype')
    probs = value.float()
    if (not torch.isfinite(probs).all() or (probs < 0).any()
            or (probs > 1).any()
            or not torch.allclose(probs.sum(-1), torch.ones((n, n)), atol=0.003, rtol=0)):
        raise ValueError('Invalid diffusion geometry probabilities')


def _validate_payload(payload, record, spec):
    if spec.get('model') != 'protenix-v1':
        validate_joint(payload, record, spec)
        _validate_geometry(payload, spec)
        return
    if payload.get('sample_name') != record['name'] or payload.get('input_sequences') != _record_sequences(record, False):
        raise ValueError('Sample or sequence identity mismatch')
    if not _same_profile(payload.get('extraction_config'), spec):
        raise ValueError('Extraction profile mismatch')
    if not REQUIRED_CACHE_KEYS.issubset(payload):
        raise ValueError('Incomplete base feature cache')
    n = payload['s'].shape[0]
    if (payload['s'].ndim != 2 or payload['z'].ndim != 3 or payload['p_dist'].ndim != 3
            or payload['z'].shape[:2] != (n, n) or payload['p_dist'].shape[:2] != (n, n)
            or payload['token_asym_id'].shape != (n,) or payload['token_residue_index'].shape != (n,)):
        raise ValueError('Invalid base feature shapes')
    # Legacy base caches store token indices as FP16. Do not rewrite them on resume.
    if any(payload[k].dtype != DTYPES[spec['cache_precision']] for k in ('s', 'z', 'p_dist')):
        raise ValueError('Base cache precision mismatch')
    _validate_geometry(payload, spec)


def _cached_index_row(
    output_path: Path,
    *,
    sample_name: str,
    input_index: int,
    record=None,
    run_spec=None,
) -> dict[str, object]:
    """Validate one existing cache without rewriting it and return metadata."""
    existing = torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
    if existing.get("sample_name") != sample_name:
        raise ValueError("Cache identity mismatch")
    if run_spec is not None:
        _validate_payload(existing, record, run_spec)
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
        **({"d_geomt_shape": list(existing["d_geomt"].shape)} if "d_geomt" in existing else {}),
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
    run_spec=None,
    no_msa=True,
) -> tuple[Path, list[int], list[dict[str, object]]]:
    """Validate cached records and write only pending records for featurization."""
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of records in {input_path}")
    names = [record['name'] for record in records]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('Duplicate or unsafe sample names')
    for record in records:
        _record_sequences(record, run_spec is not None and run_spec.get("model") == MINI_MODEL)
    reject_msa_templates(records)
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
                    record=record, run_spec=run_spec,
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
        "extraction_config": shard_stats[0]["extraction_config"],
    }
    if any(not _same_profile(stat['extraction_config'], shard_stats[0]['extraction_config']) for stat in shard_stats):
        raise ValueError('Worker profiles differ')
    (output_dir / 'run_spec.json').write_text(json.dumps(shard_stats[0]['extraction_config'],indent=2))
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


def _finish_worker(args, output_dir, index_rows, generated_count, reused_count,
                   generated_seconds, run_start, run_spec, failures):
    failure_path = output_dir / f'failures.shard_{args.shard_index:02d}.json'
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
        "extraction_config": run_spec,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    stats_path = output_dir / f"extraction_stats{metadata_suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    failure_path.write_text(json.dumps(failures, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} samples failed; see {failure_path}')
    print(f"Extracted {len(index_rows)} representations; index: {index_path}")
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(
        description="Extract frozen Protenix-v1 s, z, and p_dist tensors."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument('--model-name', choices=[BASE_MODEL, MINI_MODEL], default=BASE_MODEL)
    parser.add_argument('--no-msa', action='store_true', help='Compatibility flag; MSA is always disabled')
    parser.add_argument('--seed', type=int, help='Default: base=101, Mini-ESM=42')
    parser.add_argument('--dtype', choices=DTYPES, default='bf16')
    parser.add_argument('--cache-dtype', choices=DTYPES, default='fp16')
    parser.add_argument('--use-d-geomt', action='store_true',
                        help='Run frozen diffusion and cache 64-bin CA geometry per pair')
    parser.add_argument('--diffusion-steps', type=int, default=20)
    parser.add_argument('--diffusion-samples', type=int, default=1)
    parser.add_argument('--trimul-kernel', choices=['torch', 'cuequivariance'], default='cuequivariance')
    parser.add_argument('--triatt-kernel', choices=['torch', 'cuequivariance', 'triattention'], default='cuequivariance')
    parser.add_argument('--work-dir', help='Local preprocessing directory, separate from the feature mount')
    parser.add_argument('--esm-cache-dir', help='Precomputed cache from extract_protenix_esm_chains.py')
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
        "--multi-gpu-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    is_mini = args.model_name == MINI_MODEL
    args.seed = args.seed if args.seed is not None else (42 if is_mini else 101)
    args.use_msa = False
    args.no_msa = True
    if not is_mini:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not is_mini and args.esm_cache_dir:
        raise ValueError('--esm-cache-dir is only valid for Mini-ESM')
    if 'cuequivariance' in (args.trimul_kernel, args.triatt_kernel):
        import cuequivariance_torch  # Fail before writing output if acceleration is unavailable.
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.diffusion_steps < 1 or args.diffusion_samples < 1:
        raise ValueError("Diffusion steps/samples must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    if not args.multi_gpu_worker:
        lock_root = Path(args.output_dir).resolve()
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_name = ('.extraction.lock' if args.num_shards == 1 else
                     f'.extraction.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.lock')
        extraction_lock = (lock_root / lock_name).open('a')
        fcntl.flock(extraction_lock, fcntl.LOCK_EX)
    if args.gpu_ids is not None and not args.multi_gpu_worker:
        _run_multi_gpu(args, _resolve_gpu_ids(args.gpu_ids))
        return

    seed_rng(args.seed)
    if is_mini and (args.cycles != 4 or not args.esm_cache_dir):
        raise ValueError('mini-ESM experiment requires --cycles 4 --esm-cache-dir')
    if args.esm_cache_dir:
        args.esm_cache_dir = str(Path(args.esm_cache_dir).resolve())
        actual_esm_spec = json.loads((Path(args.esm_cache_dir) / 'run_spec.json').read_text())
        if actual_esm_spec != esm_spec(args.seed, args.dtype):
            raise ValueError('ESM stage seed/dtype/package profile differs')
    else:
        actual_esm_spec = None
    protenix_root = Path(os.environ["PROTENIX_ROOT_DIR"]).resolve()
    checkpoint = protenix_root / "checkpoint" / f"{args.model_name}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Protenix checkpoint not found at {checkpoint}. "
            "Set PROTENIX_ROOT_DIR to the directory containing checkpoint/ and common/."
        )
    if is_mini:
        run_spec = {'model': args.model_name, 'package_version': versions()['protenix'],
                    'package_versions': versions(), 'cycles': args.cycles, 'seed': args.seed,
                    'inference_precision': args.dtype, 'cache_precision': args.cache_dtype,
                    'use_msa': args.use_msa, 'use_template': False, 'use_rna_msa': False,
                    'esm_enable': args.model_name == 'protenix_mini_esm_v0.5.0',
                    'esm_model': 'esm2-3b' if args.model_name == 'protenix_mini_esm_v0.5.0' else None,
                    'esm_config': actual_esm_spec, 'channels': {'s':384,'z':128,'p_dist':64},
                    'diffusion_executed': args.use_d_geomt, 'confidence_executed': False,
                    'kernel_backend': f'{args.trimul_kernel}/{args.triatt_kernel}'}
        if args.use_d_geomt:
            run_spec['diffusion_geometry'] = {'feature': 'd_geomt', 'steps': args.diffusion_steps,
                                              'samples': args.diffusion_samples, 'bins': 64,
                                              'max_angstrom': 22.0, 'atom': 'CA'}
    else:
        run_spec = _base_spec(args, checkpoint)
    print(f"PROTENIX_ROOT_DIR={protenix_root}", flush=True)

    input_path = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        "protenix_input_cache"
        if args.num_shards == 1
        else f"protenix_input_cache_shard_{args.shard_index:02d}"
    )
    work_dir = Path(args.work_dir).resolve() / cache_name if args.work_dir else output_dir / cache_name
    work_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / 'run_spec.json'
    if profile_path.exists() and not _same_profile(json.loads(profile_path.read_text()), run_spec):
        raise ValueError('Output directory already belongs to a different extraction profile')
    if not args.multi_gpu_worker:
        profile_path.write_text(json.dumps(run_spec, indent=2))

    run_start = time.perf_counter()
    worker_input_path, pending_input_indices, cached_rows = _prepare_worker_input(
        input_path,
        work_dir,
        output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
        resume=args.resume, run_spec=run_spec, no_msa=args.no_msa,
    )
    if not pending_input_indices:
        _finish_worker(args, output_dir, cached_rows, 0, len(cached_rows), 0.0,
                       run_start, run_spec, [])
        return
    prepared_json = preprocess_input(
        str(worker_input_path),
        out_dir=str(work_dir),
        use_msa=args.use_msa,
        use_template=False,
        msa_server_mode="protenix",
    )
    # The installed official mini configuration defaults to load_strict=False.
    # Override it before runner construction so missing/unexpected keys abort.
    from configs.configs_model_type import model_configs
    model_configs[args.model_name]['load_strict'] = True
    runner = _load_checkpointed_runner(
        model_name=args.model_name,
        seeds=[args.seed],
        n_cycle=args.cycles,
        n_step=1,
        n_sample=1,
        dtype=args.dtype,
        use_msa=args.use_msa,
        trimul_kernel=args.trimul_kernel,
        triatt_kernel=args.triatt_kernel,
        enable_cache=True,
        enable_fusion=False,
        use_template=False,
        use_rna_msa=False,
    )
    runner.model.eval().requires_grad_(False)
    configs = runner.configs
    configs.input_json_path = prepared_json
    configs.num_workers = 0
    if (configs.model_name != args.model_name or bool(configs.esm.enable) != is_mini
            or configs.use_msa != args.use_msa or configs.use_template or configs.use_rna_msa
            or configs.model.N_cycle != args.cycles or configs.dtype != args.dtype or not configs.load_strict):
        raise ValueError('Runner configuration differs from requested profile')
    if is_mini and (configs.esm.model_name != 'esm2-3b' or configs.esm.embedding_dim != 2560):
        raise ValueError('Unexpected official ESM configuration')
    # init_env may update settings; reset the actual RNG immediately before data/trunk work.
    seed_rng(args.seed)
    _install_empty_template_fast_path()
    adapter = cached_esm_dataloader(configs, args.esm_cache_dir, work_dir, validate_tensors=False) if is_mini else nullcontext()
    with adapter:
        dataloader = get_inference_dataloader(configs=configs)
        if is_mini:
            stabilize_esm_paths(dataloader, configs)
    records_by_name = {record['name']:record for record in json.loads(worker_input_path.read_text())}
    failure_path = output_dir / f'failures.shard_{args.shard_index:02d}.json'
    failures = []

    index_rows: list[dict[str, object]] = cached_rows
    generated_count = 0
    reused_count = len(cached_rows)
    generated_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iterator = iter(dataloader)
    for local_item_index in range(len(pending_input_indices)):
        try:
            if not is_mini:
                from protenix.utils.seed import seed_everything
                seed_everything(seed=args.seed, deterministic=True)
                seed_rng(args.seed)
            batch = next(data_iterator)
            input_index = pending_input_indices[local_item_index]
            data, _atom_array, data_error_message = batch[0]
            sample_name = str(data["sample_name"])
            record = records_by_name[sample_name]
            if data_error_message:
                raise RuntimeError(f"Input preparation failed for {sample_name}: {data_error_message}")

            output_path = output_dir / f"{sample_name}.pt"
            sample_start = time.perf_counter()
            data = to_device(data, runner.device)
            token_count = int(data["N_token"].item())
            runner.update_model_configs(update_inference_configs(configs, token_count))
            features = data["input_feature_dict"]

            seqs = _record_sequences(record, is_mini)
            if is_mini:
                expected = torch.cat([load_chain(args.esm_cache_dir, seq, actual_esm_spec) for seq in seqs]).to(runner.device)
                actual = features['esm_token_embedding']
                if actual.shape != expected.shape or not torch.equal(actual.float(), expected):
                    raise ValueError('Featurizer ESM tokens differ from validated chain cache')
                del expected, actual
            # A sample's RNG is independent of shard count and resume position.
            if is_mini:
                seed_rng(args.seed)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPES[args.dtype], enabled=args.dtype != 'fp32'):
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
                d_geomt = (sample_d_geomt(
                    runner.model, features, _s_inputs, s, z,
                    steps=args.diffusion_steps, samples=args.diffusion_samples,
                ) if args.use_d_geomt else None)

            payload = {
                'kernel_backend': f'{args.trimul_kernel}/{args.triatt_kernel}',
                'sample_name': sample_name, 'input_sequences': seqs,
                'extraction_config': run_spec,
                'token_asym_id': features['asym_id'].detach().to(device='cpu',dtype=torch.int64),
                'token_residue_index': features['residue_index'].detach().to(device='cpu',dtype=torch.int64),
                's': _cpu_tensor(s, DTYPES[args.cache_dtype]),
                'z': _cpu_tensor(z, DTYPES[args.cache_dtype]),
                'p_dist': _cpu_tensor(p_dist, DTYPES[args.cache_dtype]),
            }
            if d_geomt is not None:
                payload['d_geomt'] = _cpu_tensor(d_geomt, DTYPES[args.cache_dtype])
            _validate_payload(payload, record, run_spec)
            atomic_save(payload, output_path)
            del payload
            elapsed = time.perf_counter() - sample_start
            generated_count += 1
            generated_seconds += elapsed
            if generated_count == 1 or generated_count % 20 == 0:
                suffix = '' if args.num_shards == 1 else f'.shard_{args.shard_index:02d}_of_{args.num_shards:02d}'
                progress_path = output_dir / f'progress{suffix}.json'
                temporary = progress_path.with_suffix(f'.json.{os.getpid()}.tmp')
                temporary.write_text(json.dumps({
                    'samples_generated': generated_count, 'samples_reused': reused_count,
                    'selected_samples': len(pending_input_indices) + reused_count,
                    'last_sample': sample_name, 'generated_seconds': generated_seconds,
                    'cycles': args.cycles, 'use_msa': False,
                    'kernel_backend': f'{args.trimul_kernel}/{args.triatt_kernel}'}, indent=2))
                temporary.replace(progress_path)

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
                    **({"d_geomt_shape": list(d_geomt.shape)} if d_geomt is not None else {}),
                }
            )
            del _s_inputs, s, z, p_dist, d_geomt, features, data
            torch.cuda.empty_cache()
            print(
                f"[{input_index + 1}] saved {output_path.name} "
                f"({elapsed:.2f}s, {output_path.stat().st_size / 1024**2:.2f} MiB)",
                flush=True,
            )

        except Exception as error:
            failures.append({'input_index':pending_input_indices[local_item_index],
                             'sample_id':list(records_by_name)[local_item_index], 'error':repr(error)})
            failure_path.write_text(json.dumps(failures,indent=2))
            print(f'Extraction failed: {failures[-1]}', flush=True)
            if len(failures) >= 3:
                break

    _finish_worker(args, output_dir, index_rows, generated_count, reused_count,
                   generated_seconds, run_start, run_spec, failures)


if __name__ == "__main__":
    main()
