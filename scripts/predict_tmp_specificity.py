#!/usr/bin/env python3
"""Predict TMP binding probabilities and optionally save pooled latent features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from foldabs_repro.tmp_prediction.dataset import TMPFeatureDataset, examples_from_manifest, single_item
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--latent-dir", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TMP Pairformer inference")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TMPTrunkSpecificityHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold)

    manifest = pd.read_csv(args.manifest)
    examples = examples_from_manifest(manifest, args.features_dir, split=args.split, require_label=False)
    loader = DataLoader(TMPFeatureDataset(examples), batch_size=1, shuffle=False, collate_fn=single_item)
    latent_dir = Path(args.latent_dir) if args.latent_dir else None
    if latent_dir:
        latent_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in loader:
            logit, latent, pools = model(
                batch["s"].to(device), batch["z"].to(device), batch["p_dist"].to(device),
                batch["token_asym_id"].to(device), batch["roles"], return_latent=True,
            )
            score = float(torch.sigmoid(logit).item())
            row = {
                "ID": batch["original_id"],
                "sample_id": batch["sample_id"],
                "pmhc_id": batch["pmhc_id"],
                "logit": float(logit.item()),
                "Prediction": score,
                "prediction_binary": int(score >= threshold),
            }
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
            **binary_metrics(output.label, output.Prediction),
            "macro_pmhc": macro_pmhc_metrics(
                output.rename(columns={"Prediction": "prediction"})
            ),
        }
        output_path.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
        )
    print(f"Wrote {len(output)} predictions to {output_path}")


if __name__ == "__main__":
    main()
