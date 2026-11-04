#!/usr/bin/env python3
"""Predict TMP binding classes and optionally save pooled latent features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from foldabs_repro.tmp_prediction.dataset import TMPFeatureDataset, examples_from_manifest, single_item
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def main() -> None:
    """Export test classes and logits; use continuous logits for ranking metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", choices=["test"], default="test",
                        help="Only the test split is exported")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--latent-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default=None,
                        help="Defaults to the checkpoint's training precision")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    precision = args.precision or checkpoint.get("run_config", {}).get("precision", "fp32")
    model = TMPTrunkSpecificityHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold)
    if not 0 < threshold < 1:
        raise ValueError("--threshold must lie strictly between 0 and 1")
    logit_threshold = math.log(threshold / (1 - threshold))

    manifest = pd.read_csv(args.manifest)
    examples = examples_from_manifest(manifest, args.features_dir, split=args.split, require_label=False)
    loader = DataLoader(TMPFeatureDataset(examples), batch_size=1, shuffle=False, collate_fn=single_item)
    latent_dir = Path(args.latent_dir) if args.latent_dir else None
    if latent_dir:
        latent_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16"):
                logit, latent, pools = model(
                    batch["s"].to(device), batch["z"].to(device), batch["p_dist"].to(device),
                    batch["token_asym_id"].to(device), batch["roles"],
                    d_geom=(batch["d_geom"].to(device)
                             if model.config["use_d_geom"] and batch["d_geom"] is not None else None),
                    p_pae=(batch["p_pae"].to(device)
                           if model.config["use_d_geom"] and batch["p_pae"] is not None else None),
                    iptm=(batch["iptm"].to(device)
                          if model.config["use_d_geom"] and batch["iptm"] is not None else None),
                    return_latent=True,
                )
            row = {
                "ID": batch["original_id"],
                "sample_id": batch["sample_id"],
                "pmhc_id": batch["pmhc_id"],
                "logit": float(logit.item()),
                "prediction": int(logit.item() >= logit_threshold),
            }
            if checkpoint.get("smoke_output", False):
                row["smoke_output"] = True
            if batch["label"] is not None:
                row["label"] = int(batch["label"].item())
            rows.append(row)
            if latent_dir:
                torch.save(
                    {
                        "sample_id": batch["sample_id"],
                        "latent": latent.detach().to(device="cpu", dtype=torch.float16),
                        "pools": {key: value.detach().to(device="cpu", dtype=torch.float16) for key, value in pools.items()},
                        "feature_schema": checkpoint.get("feature_schema", "unknown"),
                    },
                    latent_dir / f"{batch['sample_id']}.pt",
                )
    output = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    if "label" in output:
        metrics = {
            **binary_metrics(output.label, output.logit),
            "macro_pmhc": macro_pmhc_metrics(output),
        }
        output_path.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
        )
    print(f"Wrote {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()
