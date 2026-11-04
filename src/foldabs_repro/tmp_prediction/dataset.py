from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .interfaces import REQUIRED_ROLES


@dataclass(frozen=True)
class TMPFeatureExample:
    sample_id: str
    original_id: str
    pmhc_id: str
    roles: tuple[str, ...]
    feature_path: Path
    label: int | None = None


def examples_from_manifest(
    manifest: pd.DataFrame,
    feature_dir: str | Path,
    *,
    split: str | None = None,
    require_label: bool = False,
) -> list[TMPFeatureExample]:
    required = {"sample_id", "original_id", "pmhc_id"}
    if split is not None:
        required.add("split")
    if require_label:
        required.add("label")
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"TMP manifest is missing {missing}")
    selected = manifest if split is None else manifest[manifest["split"] == split]
    if selected.empty:
        raise ValueError(f"No samples selected for split={split!r}")
    root = Path(feature_dir)
    examples: list[TMPFeatureExample] = []
    for row in selected.itertuples(index=False):
        feature_path = root / f"{row.sample_id}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing Protenix features: {feature_path}")
        label = int(row.label) if hasattr(row, "label") and not pd.isna(row.label) else None
        if require_label and label is None:
            raise ValueError(f"Missing label for {row.sample_id}")
        examples.append(
            TMPFeatureExample(
                sample_id=str(row.sample_id),
                original_id=str(row.original_id),
                pmhc_id=str(row.pmhc_id),
                roles=tuple(REQUIRED_ROLES),
                feature_path=feature_path,
                label=label,
            )
        )
    return examples


class TMPFeatureDataset(Dataset):
    def __init__(self, examples: list[TMPFeatureExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        payload = torch.load(example.feature_path, map_location="cpu", weights_only=True)
        if str(payload.get("sample_name", "")) != example.sample_id:
            raise ValueError(f"Feature ID mismatch in {example.feature_path}")
        required = {"token_asym_id", "s", "z", "p_dist"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Incomplete Protenix feature {example.feature_path}: {sorted(missing)}")
        geometry = payload.get("d_geomt")
        if geometry is not None:
            n = payload["s"].shape[0]
            if (not isinstance(geometry, torch.Tensor) or geometry.shape != (n, n, 64)
                    or not torch.isfinite(geometry.float()).all()):
                raise ValueError(f"Invalid d_geomt in {example.feature_path}")
        return {
            "sample_id": example.sample_id,
            "original_id": example.original_id,
            "pmhc_id": example.pmhc_id,
            "roles": example.roles,
            "token_asym_id": payload["token_asym_id"].to(torch.int64),
            "s": payload["s"].float(),
            "z": payload["z"].float(),
            "p_dist": payload["p_dist"].float(),
            "d_geomt": None if geometry is None else geometry.float(),
            "label": None if example.label is None else torch.tensor(example.label, dtype=torch.float32),
        }


def single_item(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Variable-length TMP complexes require batch_size=1")
    return batch[0]
