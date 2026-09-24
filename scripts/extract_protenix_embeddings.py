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


def _strip_unused_s_inputs(path: Path, payload: dict) -> dict:
    if "s_inputs" not in payload:
        return payload
    payload = dict(payload)
    payload.pop("s_inputs", None)
    torch.save(payload, path)
    return payload


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
        "--resume",
        action="store_true",
        help=(
            "Reuse valid per-sample .pt files after an interrupted run. "
            "Legacy caches containing s_inputs are rewritten without it."
        ),
    )
    parser.add_argument(
        "--msa-server-mode", choices=["protenix", "colabfold"], default="protenix"
    )
    args = parser.parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

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
    work_dir = output_dir / "protenix_input_cache"
    work_dir.mkdir(exist_ok=True)

    run_start = time.perf_counter()
    prepared_json = preprocess_input(
        str(input_path),
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

    index_rows: list[dict[str, object]] = []
    generated_count = 0
    reused_count = 0
    generated_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for item_index, batch in enumerate(dataloader):
        if args.limit and item_index >= args.limit:
            break
        data, _atom_array, data_error_message = batch[0]
        sample_name = str(data["sample_name"])
        if data_error_message:
            raise RuntimeError(f"Input preparation failed for {sample_name}: {data_error_message}")

        output_path = output_dir / f"{sample_name}.pt"
        if args.resume and output_path.is_file():
            existing = torch.load(output_path, map_location="cpu", weights_only=True)
            required = {
                "token_asym_id", "token_residue_index", "s", "z", "p_dist"
            }
            if not required.issubset(existing):
                raise ValueError(f"Cannot resume from incomplete feature file: {output_path}")
            existing = _strip_unused_s_inputs(output_path, existing)
            reused_count += 1
            index_rows.append(
                {
                    "sample_id": sample_name,
                    "file": output_path.name,
                    "n_token": int(existing["s"].shape[0]),
                    "file_bytes": int(output_path.stat().st_size),
                    "s_shape": list(existing["s"].shape),
                    "z_shape": list(existing["z"].shape),
                    "p_dist_shape": list(existing["p_dist"].shape),
                }
            )
            print(f"[{item_index + 1}] reused {output_path.name}", flush=True)
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
            f"[{item_index + 1}] saved {output_path.name} "
            f"({elapsed:.2f}s, {output_path.stat().st_size / 1024**2:.2f} MiB)",
            flush=True,
        )

    index_path = output_dir / "index.json"
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
    }
    stats_path = output_dir / "extraction_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Extracted {len(index_rows)} representations; index: {index_path}")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
