#!/usr/bin/env python3
"""Train the paper-aligned FoldAbS developability head on frozen features.

This is intentionally separate from ``train_protenix_pairformer_head.py``.
The older script remains a documented early baseline; this script follows the
published developability branch: distogram conditioning, one Pairformer block,
frozen ESM-2 heavy/light embeddings, and a three-layer regression MLP.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from foldabs_repro.data import DatasetSchema, DatasetSplits, fit_label_normalizer, load_fitness_csv, split_by_group
from foldabs_repro.metrics import regression_metrics
from foldabs_repro.tasks import DEVELOPABILITY_TASKS, get_developability_task
from protenix.model.modules.pairformer import PairformerBlock


@dataclass(frozen=True)
class Example:
    sample_id: str
    label: float
    source: str
    protenix_path: Path
    esm_path: Path


@dataclass(frozen=True)
class ESMEmbeddingNormalizer:
    """Train-split-only per-channel normalization for frozen ESM embeddings."""

    heavy_mean: torch.Tensor
    heavy_std: torch.Tensor
    light_mean: torch.Tensor
    light_std: torch.Tensor

    @classmethod
    def fit(cls, examples: list[Example]) -> "ESMEmbeddingNormalizer":
        if not examples:
            raise ValueError("Cannot fit an ESM normalizer without training examples.")
        heavy, light = [], []
        for example in examples:
            payload = torch.load(example.esm_path, map_location="cpu", weights_only=True)
            heavy.append(payload["e_h"].float())
            light.append(payload["e_l"].float())
        heavy_tensor, light_tensor = torch.stack(heavy), torch.stack(light)
        # Clamp protects an all-zero light-chain dimension in nanobody records.
        heavy_std = heavy_tensor.std(dim=0, unbiased=False)
        light_std = light_tensor.std(dim=0, unbiased=False)
        return cls(
            heavy_mean=heavy_tensor.mean(dim=0),
            heavy_std=torch.where(heavy_std > 1e-6, heavy_std, torch.ones_like(heavy_std)),
            light_mean=light_tensor.mean(dim=0),
            light_std=torch.where(light_std > 1e-6, light_std, torch.ones_like(light_std)),
        )

    def transform(self, heavy: torch.Tensor, light: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (heavy - self.heavy_mean) / self.heavy_std,
            (light - self.light_mean) / self.light_std,
        )


def _load_saved_manifest(path: Path, canonical: pd.DataFrame) -> DatasetSplits:
    """Load a fixed H-disjoint split and refuse any data/version mismatch.

    This prevents a design diagnostic from silently gaining or losing samples
    merely because a new GroupShuffleSplit happened to be regenerated.
    """
    manifest = pd.read_csv(path)
    required = {"sample_id", "split", "heavy", "light", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate sample_id values.")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test rows.")
    if set(manifest["sample_id"].astype(str)) != set(canonical["sample_id"].astype(str)):
        raise ValueError("Split manifest sample_ids do not exactly cover the canonical dataset.")
    ordered = canonical.set_index("sample_id").loc[manifest["sample_id"]].reset_index()
    if not (ordered["heavy"].to_numpy() == manifest["heavy"].fillna("").astype(str).to_numpy()).all():
        raise ValueError("Split manifest heavy sequences do not match canonical data.")
    if not (ordered["light"].to_numpy() == manifest["light"].fillna("").astype(str).to_numpy()).all():
        raise ValueError("Split manifest light sequences do not match canonical data.")
    if not np.allclose(ordered["label"].to_numpy(dtype=float), manifest["label"].to_numpy(dtype=float)):
        raise ValueError("Split manifest labels do not match canonical data.")
    parts = {
        split: ordered.loc[manifest["split"].to_numpy() == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    groups = {split: set(part["heavy"]) for split, part in parts.items()}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("Split manifest leaks a heavy chain across splits.")
    return DatasetSplits(**parts)


class FeatureDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        label_mean: float,
        label_std: float,
        esm_normalizer: ESMEmbeddingNormalizer | None = None,
    ):
        self.examples = examples
        self.label_mean = label_mean
        self.label_std = label_std
        self.esm_normalizer = esm_normalizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        protenix = torch.load(example.protenix_path, map_location="cpu", weights_only=True)
        esm = torch.load(example.esm_path, map_location="cpu", weights_only=True)
        # The path itself is not a sufficient pairing guarantee: a copied or
        # incorrectly renamed Protenix feature could otherwise be silently
        # attached to the wrong antibody.  The extraction script stores the
        # originating input ID as ``sample_name``; require that it matches the
        # CSV/manifest ID before this frozen structural representation enters
        # any train, validation, or test calculation.
        if str(protenix.get("sample_name", "")) != example.sample_id:
            raise ValueError(f"Protenix feature id mismatch in {example.protenix_path}")
        if str(esm["sample_id"]) != example.sample_id:
            raise ValueError(f"ESM feature id mismatch in {example.esm_path}")
        e_h, e_l = esm["e_h"].float(), esm["e_l"].float()
        if self.esm_normalizer is not None:
            e_h, e_l = self.esm_normalizer.transform(e_h, e_l)
        return {
            "sample_id": example.sample_id,
            "source": example.source,
            "label": torch.tensor((example.label - self.label_mean) / self.label_std, dtype=torch.float32),
            "label_raw": torch.tensor(example.label, dtype=torch.float32),
            "s": protenix["s"].float(),
            "z": protenix["z"].float(),
            # The extraction script stores softmax(distogram logits), matching c_ij in the paper.
            "p_dist": protenix["p_dist"].float(),
            "e_h": e_h,
            "e_l": e_l,
        }


def _single_item(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Variable-length Pairformer examples require batch_size=1")
    return batch[0]


class ResidualTransition(nn.Module):
    """A lightweight residual transition used inside the paper's PairCond module."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class PairCondition(nn.Module):
    """Project p_dist probabilities and inject them into the Protenix pair state."""

    def __init__(
        self,
        c_dist: int,
        c_z: int,
        transition_layers: int = 2,
        mode: str = "z_aware_transition",
    ):
        super().__init__()
        if mode not in {"additive", "z_aware_transition"}:
            raise ValueError(f"Unknown PairCond mode '{mode}'")
        self.mode = mode
        self.project = nn.Sequential(nn.LayerNorm(c_dist), nn.Linear(c_dist, c_z))
        self.transitions = nn.ModuleList(
            ResidualTransition(c_z) for _ in range(transition_layers)
        )

    def forward(self, z: torch.Tensor, p_dist: torch.Tensor) -> torch.Tensor:
        condition = self.project(p_dist)
        if self.mode == "z_aware_transition":
            # The manuscript writes PairCond(z, c_ij), rather than PairCond(c_ij).
            # Interpret that literally: first combine the frozen pair state with
            # the projected distance condition, then let both residual transition
            # layers operate on that joint state.  The old ``additive`` branch is
            # retained only to read and audit earlier reconstruction checkpoints.
            conditioned = z + condition
            for transition in self.transitions:
                conditioned = transition(conditioned)
            return conditioned
        for transition in self.transitions:
            condition = transition(condition)
        return z + condition


class FoldAbSDevelopabilityHead(nn.Module):
    """One-property FoldAbS development head with a frozen Protenix encoder."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_dist: int,
        esm_dim: int,
        use_esm: bool = True,
        structure_readout: str = "s_z",
        use_dist_conditioning: bool = True,
        use_pairformer: bool = True,
        branch_layernorm: bool = False,
        ablation_stage: str | None = None,
        paircond_mode: str = "z_aware_transition",
        fusion_normalization: str = "legacy_global",
        dropout: float = 0.1,
        pairformer_dropout: float = 0.1,
        pairformer_n_heads: int = 16,
        pairformer_c_hidden_mul: int = 128,
        pairformer_c_hidden_pair_att: int = 32,
        pairformer_no_heads_pair: int = 4,
        esm_context_dropout: float = 0.0,
        structural_component_layernorm: bool = False,
        pre_pairformer_layernorm: bool = False,
        regressor_hidden_dim: int = 512,
        structural_pool_center: torch.Tensor | None = None,
    ):
        super().__init__()
        if ablation_stage not in {None, "single_only", "distogram_pair", "pairformer", "full"}:
            raise ValueError(f"Unknown cumulative ablation stage '{ablation_stage}'")
        self.ablation_stage = ablation_stage
        # The Table S8 reconstruction adds ESM only in its final row.  Keep the
        # ordinary/diagnostic code path unchanged when no stage is selected.
        self.use_esm = use_esm and (ablation_stage is None or ablation_stage == "full")
        if structure_readout not in {"s", "z", "s_z"}:
            raise ValueError(f"Unknown structural readout '{structure_readout}'")
        self.structure_readout = structure_readout
        self.use_dist_conditioning = use_dist_conditioning
        self.use_pairformer = use_pairformer
        self.paircond_mode = paircond_mode
        if fusion_normalization not in {"legacy_global", "branch_then_global", "branch_only"}:
            raise ValueError(f"Unknown fusion normalization '{fusion_normalization}'")
        self.fusion_normalization = fusion_normalization
        if not 0.0 <= esm_context_dropout < 1.0:
            raise ValueError("esm_context_dropout must be in [0, 1).")
        # This is deliberately a *whole ESM-context* dropout: H and L are
        # dropped together for some training records.  Coordinate-wise
        # dropout would still leave a very strong ESM shortcut at batch size 1.
        # At evaluation the context is always intact, just like ordinary
        # nn.Dropout.
        self.esm_context_dropout = esm_context_dropout
        # ``branch_only`` treats Protenix as one branch relative to ESM.  The
        # stage-2 trace showed that this is still insufficient when the pooled
        # single-state vector (s) is hundreds of times larger than the pooled
        # pair/distogram vector (z or d).  This optional diagnostic normalizes
        # those two *structural* components independently before concatenation.
        self.structural_component_layernorm = structural_component_layernorm
        # The frozen Protenix tensors retain their native scale (s RMS ~245 and
        # z RMS ~43 in the Immunogenicity reconstruction).  A newly trained
        # residual Pairformer can otherwise emit updates that are numerically
        # negligible beside those bases.  This optional input normalization is
        # deliberately *before* PairCond/Pairformer; it is distinct from the
        # pooled-component normalization used immediately before the regressor.
        self.pre_pairformer_layernorm = pre_pairformer_layernorm
        self.c_s, self.c_z = c_s, c_z
        # Optional, train-input-only centering for a narrowly scoped
        # single-state diagnostic.  It removes a coordinate-wise background
        # shared across antibodies *before* LayerNorm, rather than changing
        # labels, MSA features, the regressor, or the normalization itself.
        # Do not register a buffer in the ordinary path: that keeps historic
        # checkpoint state_dict contracts intact.
        if structural_pool_center is not None:
            if structural_pool_center.ndim != 1 or int(structural_pool_center.numel()) != c_s:
                raise ValueError("structural_pool_center must be a [c_s] vector")
            if ablation_stage != "single_only":
                raise ValueError("structural_pool_center is only defined for the single_only ablation")
            self.register_buffer("_structural_pool_center", structural_pool_center.detach().float().clone())
        else:
            self._structural_pool_center: torch.Tensor | None = None
        if regressor_hidden_dim < 1:
            raise ValueError("regressor_hidden_dim must be positive")
        self.regressor_hidden_dim = regressor_hidden_dim
        # For normal/diagnostic runs retain the original construction order
        # (PairCond, then Pairformer).  Cumulative rows create *only* modules
        # used by that row, so an unused distogram projector cannot perturb the
        # fixed-seed initialization of the final/full architecture.
        self.pair_condition: PairCondition | None = None
        self.distogram_pool_project: nn.Module | None = None
        self.pairformer: PairformerBlock | None = None
        if ablation_stage is None:
            self.pair_condition = PairCondition(c_dist=c_dist, c_z=c_z, transition_layers=2, mode=paircond_mode)
            self.pairformer = PairformerBlock(
                n_heads=pairformer_n_heads,
                c_s=c_s,
                c_z=c_z,
                c_hidden_mul=pairformer_c_hidden_mul,
                c_hidden_pair_att=pairformer_c_hidden_pair_att,
                no_heads_pair=pairformer_no_heads_pair,
                dropout=pairformer_dropout,
            )
        elif ablation_stage == "distogram_pair":
            # Protenix extraction already stores softmax(distogram logits) as p_dist.
            self.distogram_pool_project = nn.Sequential(nn.LayerNorm(c_dist), nn.Linear(c_dist, c_z))
        elif ablation_stage in {"pairformer", "full"}:
            # Use the same PairCond implementation for the cumulative Pairformer
            # and full rows as for the main model.  Earlier code hard-coded the
            # legacy additive reconstruction here, even when a z-aware PairCond
            # was explicitly requested.
            self.pair_condition = PairCondition(
                c_dist=c_dist,
                c_z=c_z,
                transition_layers=2,
                mode=paircond_mode,
            )
            self.pairformer = PairformerBlock(
                n_heads=pairformer_n_heads,
                c_s=c_s,
                c_z=c_z,
                c_hidden_mul=pairformer_c_hidden_mul,
                c_hidden_pair_att=pairformer_c_hidden_pair_att,
                no_heads_pair=pairformer_no_heads_pair,
                dropout=pairformer_dropout,
            )
        if ablation_stage == "single_only":
            structural_dim = c_s
        elif ablation_stage in {"distogram_pair", "pairformer", "full"}:
            structural_dim = c_s + c_z
        else:
            structural_dim = c_s if structure_readout == "s" else c_z if structure_readout == "z" else c_s + c_z
        independent_branch_norm = fusion_normalization in {"branch_then_global", "branch_only"}
        self.structural_norm = (
            nn.Identity()
            if structural_component_layernorm
            else nn.LayerNorm(structural_dim) if independent_branch_norm else nn.Identity()
        )
        self.structural_s_norm = nn.LayerNorm(c_s) if structural_component_layernorm else nn.Identity()
        self.structural_pair_norm = nn.LayerNorm(c_z) if structural_component_layernorm else nn.Identity()
        self.pre_pairformer_s_norm = nn.LayerNorm(c_s) if pre_pairformer_layernorm else nn.Identity()
        self.pre_pairformer_z_norm = nn.LayerNorm(c_z) if pre_pairformer_layernorm else nn.Identity()
        self.heavy_norm = nn.LayerNorm(esm_dim) if independent_branch_norm else nn.Identity()
        self.light_norm = nn.LayerNorm(esm_dim) if independent_branch_norm else nn.Identity()
        regressor_input_dim = structural_dim + (2 * esm_dim if self.use_esm else 0)
        # legacy_global is the historical reconstruction. branch_then_global
        # matches the older --branch-layernorm diagnostic. branch_only is the
        # scale test: normalize each source separately but avoid calculating a
        # single mean/std over incompatible Protenix and ESM coordinate scales.
        self.fusion_norm = nn.Identity() if fusion_normalization == "branch_only" else nn.LayerNorm(regressor_input_dim)
        self.regressor = nn.Sequential(
            self.fusion_norm,
            nn.Linear(regressor_input_dim, regressor_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden_dim, regressor_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden_dim, 1),
        )

    def normalize_structural(self, structural: torch.Tensor) -> torch.Tensor:
        """Normalize pooled structural components without mixing s with z/d."""
        if not self.structural_component_layernorm:
            return self.structural_norm(structural)
        width = int(structural.shape[-1])
        if width == self.c_s:
            return self.structural_s_norm(structural)
        if width == self.c_z:
            return self.structural_pair_norm(structural)
        if width == self.c_s + self.c_z:
            return torch.cat(
                [
                    self.structural_s_norm(structural[..., : self.c_s]),
                    self.structural_pair_norm(structural[..., self.c_s :]),
                ],
                dim=-1,
            )
        raise RuntimeError(f"Unexpected pooled structural width {width}")

    def encode_structural(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        p_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one frozen Protenix record into its pooled structure vector.

        ``s``, ``z`` and ``p_dist`` are intentionally unbatched here.  The
        project currently trains variable-length records one at a time, so
        adding a singleton batch dimension at this boundary preserves the
        original forward path exactly.  Keeping this operation separate from
        ESM fusion also lets the evaluation-only modality audit replace *only*
        the pooled structural vector without changing Pairformer execution.
        """
        s, z, p_dist = s.unsqueeze(0), z.unsqueeze(0), p_dist.unsqueeze(0)
        if self.ablation_stage == "single_only":
            return s.mean(dim=1)
        if self.ablation_stage == "distogram_pair":
            s_pool = s.mean(dim=1)
            if self.distogram_pool_project is None:
                raise RuntimeError("Distogram-pair ablation is missing its projector.")
            d_pool = self.distogram_pool_project(p_dist).mean(dim=(1, 2))
            return torch.cat([s_pool, d_pool], dim=-1)

        if self.use_dist_conditioning and self.pair_condition is None:
            raise RuntimeError("Distance conditioning was requested but PairCond is unavailable.")
        # Keep frozen s/z untouched for non-Pairformer rows.  The optional
        # pre-Pairformer LayerNorm exists solely to give the trainable residual
        # PairCond/Pairformer a comparable numerical input scale.
        s_pair = self.pre_pairformer_s_norm(s) if self.use_pairformer else s
        z_pair = self.pre_pairformer_z_norm(z) if self.use_pairformer else z
        z_conditioned = self.pair_condition(z_pair, p_dist) if self.use_dist_conditioning else z_pair
        if self.use_pairformer:
            if self.pairformer is None:
                raise RuntimeError("Pairformer was requested but is unavailable.")
            s_refined, z_refined = self.pairformer(
                s=s_pair,
                z=z_conditioned,
                pair_mask=None,
                triangle_multiplicative="torch",
                triangle_attention="torch",
                inplace_safe=False,
                chunk_size=None,
            )
        else:
            s_refined, z_refined = s_pair, z_conditioned
        s_pool = s_refined.mean(dim=1)
        z_pool = z_refined.mean(dim=(1, 2))
        return (
            s_pool
            if self.structure_readout == "s"
            else z_pool
            if self.structure_readout == "z"
            else torch.cat([s_pool, z_pool], dim=-1)
        )

    def predict_from_modalities(
        self,
        structural_before_norm: torch.Tensor,
        e_h: torch.Tensor,
        e_l: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse already-pooled structure and ESM vectors, then regress Tm.

        This is the second half of :meth:`forward`.  It is public primarily so
        that ``audit_foldabs_modality_usage.py`` can perform counterfactual
        inference (zeroing or shuffling one modality) on a saved checkpoint.
        Normal training still calls :meth:`forward` and is numerically
        unchanged.
        """
        if structural_before_norm.ndim == 1:
            structural_before_norm = structural_before_norm.unsqueeze(0)
        if e_h.ndim == 1:
            e_h = e_h.unsqueeze(0)
        if e_l.ndim == 1:
            e_l = e_l.unsqueeze(0)
        if self._structural_pool_center is not None:
            structural_before_norm = structural_before_norm - self._structural_pool_center
        structural = self.normalize_structural(structural_before_norm)
        e_h_before_norm, e_l_before_norm = e_h, e_l
        if self.training and self.use_esm and self.esm_context_dropout:
            keep = (
                torch.rand((e_h.shape[0], 1), device=e_h.device) >= self.esm_context_dropout
            ).to(e_h.dtype) / (1.0 - self.esm_context_dropout)
            e_h_before_norm, e_l_before_norm = e_h * keep, e_l * keep
        e_h_after_norm, e_l_after_norm = self.heavy_norm(e_h_before_norm), self.light_norm(e_l_before_norm)
        fused = (
            torch.cat([structural, e_h_after_norm, e_l_after_norm], dim=-1)
            if self.use_esm
            else structural
        )
        prediction = self.regressor(fused).squeeze(-1)
        if not return_intermediates:
            return prediction
        return prediction, {
            "structural_before_norm": structural_before_norm,
            "structural_after_norm": structural,
            "esm_heavy_before_norm": e_h_before_norm,
            "esm_heavy_after_norm": e_h_after_norm,
            "esm_light_before_norm": e_l_before_norm,
            "esm_light_after_norm": e_l_after_norm,
            "fused_before_global_norm": fused,
            "fused_after_global_norm": self.fusion_norm(fused),
        }

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        p_dist: torch.Tensor,
        e_h: torch.Tensor,
        e_l: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structural_before_norm = self.encode_structural(s, z, p_dist)
        return self.predict_from_modalities(
            structural_before_norm,
            e_h,
            e_l,
            return_intermediates=return_intermediates,
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _examples(frame: pd.DataFrame, protenix_dir: Path, esm_dir: Path) -> list[Example]:
    examples = []
    for row in frame.itertuples(index=False):
        protenix_path = protenix_dir / f"{row.sample_id}.pt"
        esm_path = esm_dir / f"{row.sample_id}.pt"
        if not protenix_path.is_file() or not esm_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen features for {row.sample_id}: {protenix_path}, {esm_path}"
            )
        examples.append(
            Example(
                str(row.sample_id),
                float(row.label),
                str(getattr(row, "source", "unknown")),
                protenix_path,
                esm_path,
            )
        )
    return examples


def fit_single_pool_center(examples: list[Example]) -> torch.Tensor:
    """Return the train-input mean of mean-pooled frozen ``s`` vectors.

    This deliberately reads no labels.  It is only valid for the single-only
    diagnostic, where ``encode_structural`` is exactly ``mean(s, residue)``.
    Keeping the computation here makes the fitted vector explicit and avoids
    accidentally using validation/test features to form the centre.
    """
    if not examples:
        raise ValueError("Cannot fit a structural centre without training examples.")
    total: torch.Tensor | None = None
    for example in examples:
        payload = torch.load(example.protenix_path, map_location="cpu", weights_only=True)
        if str(payload.get("sample_name", "")) != example.sample_id:
            raise ValueError(f"Protenix feature id mismatch in {example.protenix_path}")
        pooled = payload["s"].float().mean(dim=0)
        total = pooled if total is None else total + pooled
    assert total is not None
    return total / len(examples)


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    """Small, JSON-safe summary for a pooled activation or gradient tensor."""
    vector = value.detach().float().reshape(-1)
    return {
        "mean": float(vector.mean().item()),
        "std": float(vector.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(vector.square())).item()),
        "l2_norm": float(torch.linalg.vector_norm(vector).item()),
        "max_abs": float(vector.abs().max().item()),
        "numel": int(vector.numel()),
    }


def _mean_activation_summaries(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    output: dict[str, dict[str, float]] = {}
    for key in keys:
        available = [row[key] for row in rows if key in row]
        output[key] = {
            metric: float(np.mean([summary[metric] for summary in available]))
            for metric in ("mean", "std", "rms", "l2_norm", "max_abs")
        }
        output[key]["samples"] = int(len(available))
    return output


def _module_norms(model: nn.Module, *, gradients: bool) -> dict[str, float]:
    """L2 norms split by the three trainable pieces relevant to the paper head."""
    totals = {"pair_condition": 0.0, "pairformer": 0.0, "regressor": 0.0, "other": 0.0}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            continue
        key = (
            "pair_condition" if name.startswith("pair_condition.")
            else "pairformer" if name.startswith("pairformer.")
            else "regressor" if name.startswith("regressor.")
            else "other"
        )
        totals[key] += float(value.detach().float().square().sum().item())
    return {key: float(np.sqrt(value)) for key, value in totals.items()}


def _make_ema_model(model: nn.Module) -> nn.Module:
    """Create a non-trainable parameter copy for exponential moving averages.

    This is an engineering training control, not a FoldAbS architecture change.
    LayerNorm has no running statistics, but non-floating buffers are copied
    exactly as well for completeness.
    """

    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update ``ema = decay * ema + (1 - decay) * current`` after one optimizer step."""

    ema_state = ema_model.state_dict()
    current_state = model.state_dict()
    if ema_state.keys() != current_state.keys():
        raise RuntimeError("EMA and online model state dictionaries do not match.")
    for name, ema_value in ema_state.items():
        current_value = current_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(current_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(current_value)


def _source_metrics(
    partition: pd.DataFrame,
    sample_ids: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    indexed = partition.set_index("sample_id").loc[sample_ids].reset_index()
    sources = indexed["source"] if "source" in indexed else pd.Series(["unknown"] * len(indexed))
    table: dict[str, dict[str, float | int]] = {}
    for source in sorted(sources.astype(str).unique()):
        mask = sources.astype(str).to_numpy() == source
        source_labels, source_predictions = labels[mask], predictions[mask]
        table[source] = {
            "n": int(mask.sum()),
            "mae_celsius": float(np.mean(np.abs(source_predictions - source_labels))),
            "rmse_celsius": float(np.sqrt(np.mean((source_predictions - source_labels) ** 2))),
            "mean_signed_error_celsius": float(np.mean(source_predictions - source_labels)),
            "label_mean_celsius": float(np.mean(source_labels)),
            "prediction_mean_celsius": float(np.mean(source_predictions)),
        }
    return table


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, normalizer, *, collect_intermediates: bool = False):
    model.eval()
    raw_labels, raw_predictions, sample_ids, normalized_losses = [], [], [], []
    activation_rows: list[dict[str, dict[str, float]]] = []
    for batch in loader:
        result = model(
            batch["s"].to(device),
            batch["z"].to(device),
            batch["p_dist"].to(device),
            batch["e_h"].to(device),
            batch["e_l"].to(device),
            return_intermediates=collect_intermediates,
        )
        if collect_intermediates:
            prediction, intermediates = result
            activation_rows.append({key: _tensor_summary(value) for key, value in intermediates.items()})
        else:
            prediction = result
        target = batch["label"].to(device).reshape_as(prediction)
        normalized_losses.append(float(F.mse_loss(prediction, target).item()))
        raw_predictions.append(float(normalizer.inverse_transform(prediction.cpu().numpy())[0]))
        raw_labels.append(float(batch["label_raw"].item()))
        sample_ids.append(str(batch["sample_id"]))
    output = (
        float(np.mean(normalized_losses)),
        np.asarray(raw_labels),
        np.asarray(raw_predictions),
        sample_ids,
    )
    return (*output, _mean_activation_summaries(activation_rows)) if collect_intermediates else output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FoldAbS developability head.")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--protenix-features-dir", required=True)
    parser.add_argument("--esm-features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional model-only checkpoint from the same architecture/split for low-LR fine-tuning.",
    )
    parser.add_argument("--heavy-column", default="heavy")
    parser.add_argument("--light-column", default="light")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument(
        "--property-name",
        choices=tuple(DEVELOPABILITY_TASKS),
        default="thermostability",
        help="Selects published row counts, label preprocessing metadata, and metric units.",
    )
    parser.add_argument(
        "--exact-paper-manifest",
        action="store_true",
        help="Declare only when train, validation, and test membership all come from the authors.",
    )
    parser.add_argument("--split-manifest", type=Path, default=None, help="Optional fixed train/validation/test manifest.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=50,
        help="Validation checks without improvement before halving LR (paper protocol: 50).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Validation checks without improvement before stopping; 0 disables it.",
    )
    parser.add_argument(
        "--train-loss-window",
        type=int,
        default=100,
        help="Optimizer steps used for the logged rolling training MSE.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "Engineering control: exponential moving average decay for parameters used in validation, "
            "checkpoint selection, and final evaluation. 0.0 disables EMA and preserves the original run."
        ),
    )
    parser.add_argument(
        "--save-online-checkpoint-with-ema",
        action="store_true",
        help=(
            "When EMA is enabled, also save the best ordinary-parameter checkpoint by its own "
            "validation MSE. This is an evaluation control and does not alter optimisation."
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help=(
            "Optional JSONL diagnostics at each validation check: gradients, parameter norms, "
            "branch activation scales, prediction distributions, and validation-source errors. "
            "This only observes training; it does not change the model or objective."
        ),
    )
    parser.add_argument("--pairformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--regressor-hidden-dim",
        type=int,
        default=512,
        help=(
            "Width of both hidden layers in the three-layer regression MLP. "
            "512 preserves the original reconstruction."
        ),
    )
    parser.add_argument(
        "--pairformer-n-heads",
        type=int,
        default=16,
        help="Pairformer single-track attention heads (Protenix default: 16).",
    )
    parser.add_argument(
        "--pairformer-c-hidden-mul",
        type=int,
        default=128,
        help="Pairformer triangle-multiplication hidden width (Protenix default: 128).",
    )
    parser.add_argument(
        "--pairformer-c-hidden-pair-att",
        type=int,
        default=32,
        help="Pairformer pair-attention hidden width (Protenix default: 32).",
    )
    parser.add_argument(
        "--pairformer-no-heads-pair",
        type=int,
        default=4,
        help="Pairformer pair-attention heads (Protenix default: 4).",
    )
    parser.add_argument(
        "--protenix-msa-source",
        default="unspecified",
        help="Input provenance only: no_msa, colabfold, compact_local, uniref50, or another explicit label.",
    )
    parser.add_argument(
        "--normalize-esm-embeddings",
        action="store_true",
        help="Diagnostic: standardize frozen ESM heavy/light embeddings with training-split statistics only.",
    )
    parser.add_argument(
        "--branch-layernorm",
        action="store_true",
        help="Diagnostic: apply independent LayerNorm to pooled structural, heavy-ESM, and light-ESM branches before fusion.",
    )
    parser.add_argument(
        "--fusion-normalization",
        choices=("legacy_global", "branch_then_global", "branch_only"),
        default="legacy_global",
        help=(
            "Fusion normalization diagnostic. branch_only normalizes each structural/ESM branch separately "
            "and removes the legacy LayerNorm over their concatenation."
        ),
    )
    parser.add_argument(
        "--structural-component-layernorm",
        action="store_true",
        help=(
            "Diagnostic: for pooled [s,z] or [s,distogram], apply separate LayerNorm "
            "to s and z/distogram before concatenation instead of one structural LayerNorm."
        ),
    )
    parser.add_argument(
        "--center-single-pool-train",
        action="store_true",
        help=(
            "Targeted single-only diagnostic: subtract the training-input mean of pooled frozen s "
            "before the existing structural LayerNorm. Labels, MSA features and the MLP are unchanged."
        ),
    )
    parser.add_argument(
        "--pre-pairformer-layernorm",
        action="store_true",
        help=(
            "Targeted structural-path repair: LayerNorm frozen per-residue s and per-pair z "
            "before PairCond/Pairformer, so their residual updates are not dwarfed by native "
            "Protenix tensor scale."
        ),
    )
    parser.add_argument(
        "--esm-context-dropout",
        type=float,
        default=0.0,
        help=(
            "Diagnostic training-only whole-context dropout for the frozen ESM heavy/light pair. "
            "0.0 preserves the original direct-fusion model exactly."
        ),
    )
    parser.add_argument(
        "--structure-readout",
        choices=("s", "z", "s_z"),
        default="s_z",
        help="Diagnostic structural readout: pooled single state s, pair state z, or both (default).",
    )
    parser.add_argument(
        "--disable-dist-conditioning",
        action="store_true",
        help="Diagnostic: bypass p_dist -> z PairCondition while retaining the remaining structure branch.",
    )
    parser.add_argument(
        "--disable-pairformer",
        action="store_true",
        help="Diagnostic: bypass the trainable Pairformer block after optional distance conditioning.",
    )
    parser.add_argument(
        "--paircond-mode",
        choices=("additive", "z_aware_transition"),
        default="z_aware_transition",
        help=(
            "PairCond implementation. 'z_aware_transition' follows the published "
            "PairCond(z, c_ij) notation by jointly refining z and the distance condition; "
            "'additive' preserves the legacy reconstruction for checkpoint audits."
        ),
    )
    parser.add_argument(
        "--disable-esm",
        action="store_true",
        help="Diagnostic ablation: retain the Protenix/Pairformer branch but remove ESM-2 from the regressor.",
    )
    parser.add_argument(
        "--cumulative-ablation-stage",
        choices=("single_only", "distogram_pair", "pairformer", "full"),
        default=None,
        help=(
            "Paper-informed Table S8 reconstruction: add one component at a time "
            "(single-only, projected distogram pooling, Pairformer, then ESM fusion)."
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    task_spec = get_developability_task(args.property_name)
    if (
        args.max_steps < 1
        or args.gradient_accumulation < 1
        or args.validate_every < 1
        or args.scheduler_patience < 0
        or args.early_stopping_patience < 0
        or args.train_loss_window < 1
        or args.regressor_hidden_dim < 1
        or not 0.0 <= args.ema_decay < 1.0
        or not 0.0 <= args.esm_context_dropout < 1.0
    ):
        raise ValueError("step counts and --gradient-accumulation must be positive")
    # The original Table-S8 reconstruction used the legacy global LayerNorm.
    # Allow an explicitly requested branch-only variant as well: it is needed
    # for a controlled structural-path diagnostic after the global-normalization
    # scale bug was found.  It is recorded as a reconstruction variant, not as
    # an exact paper implementation.
    if args.cumulative_ablation_stage is not None and any(
        (
            args.disable_esm,
            args.normalize_esm_embeddings,
            args.branch_layernorm,
            args.esm_context_dropout != 0.0,
            args.structure_readout != "s_z",
            args.disable_dist_conditioning,
            args.disable_pairformer,
            args.fusion_normalization not in {"legacy_global", "branch_only"},
        )
    ):
        raise ValueError("Cumulative ablations must run without other diagnostic flags.")
    if args.center_single_pool_train and args.cumulative_ablation_stage != "single_only":
        raise ValueError("--center-single-pool-train requires --cumulative-ablation-stage single_only")
    if args.branch_layernorm and args.fusion_normalization != "legacy_global":
        raise ValueError("Use either --branch-layernorm or --fusion-normalization, not both.")
    if args.structural_component_layernorm and args.fusion_normalization != "branch_only":
        raise ValueError("--structural-component-layernorm requires --fusion-normalization branch_only")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for Pairformer-head training")

    _set_seed(args.seed)
    device = torch.device("cuda")
    frame = load_fitness_csv(
        args.data_csv,
        schema=DatasetSchema(
            heavy_column=args.heavy_column,
            light_column=args.light_column,
            label_column=args.label_column,
        ),
    )
    splits = (
        _load_saved_manifest(args.split_manifest, frame)
        if args.split_manifest is not None
        else split_by_group(
            frame,
            group_column="heavy",
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    )
    normalizer = fit_label_normalizer(splits.train.label.to_numpy())
    protenix_dir, esm_dir = Path(args.protenix_features_dir), Path(args.esm_features_dir)
    example_splits = {
        name: _examples(partition, protenix_dir, esm_dir)
        for name, partition in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
    }
    structural_pool_center = (
        fit_single_pool_center(example_splits["train"])
        if args.center_single_pool_train
        else None
    )
    esm_normalizer = ESMEmbeddingNormalizer.fit(example_splits["train"]) if args.normalize_esm_embeddings else None
    datasets = {
        name: FeatureDataset(examples, normalizer.mean, normalizer.std, esm_normalizer=esm_normalizer)
        for name, examples in example_splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=(name == "train"), collate_fn=_single_item)
        for name, dataset in datasets.items()
    }
    first = datasets["train"][0]
    model = FoldAbSDevelopabilityHead(
        c_s=int(first["s"].shape[-1]),
        c_z=int(first["z"].shape[-1]),
        c_dist=int(first["p_dist"].shape[-1]),
        esm_dim=int(first["e_h"].shape[-1]),
        use_esm=not args.disable_esm,
        structure_readout=args.structure_readout,
        use_dist_conditioning=not args.disable_dist_conditioning,
        use_pairformer=not args.disable_pairformer,
        branch_layernorm=args.branch_layernorm,
        ablation_stage=args.cumulative_ablation_stage,
        paircond_mode=args.paircond_mode,
        fusion_normalization=("branch_then_global" if args.branch_layernorm else args.fusion_normalization),
        pairformer_dropout=args.pairformer_dropout,
        pairformer_n_heads=args.pairformer_n_heads,
        pairformer_c_hidden_mul=args.pairformer_c_hidden_mul,
        pairformer_c_hidden_pair_att=args.pairformer_c_hidden_pair_att,
        pairformer_no_heads_pair=args.pairformer_no_heads_pair,
        esm_context_dropout=args.esm_context_dropout,
        structural_component_layernorm=args.structural_component_layernorm,
        pre_pairformer_layernorm=args.pre_pairformer_layernorm,
        regressor_hidden_dim=args.regressor_hidden_dim,
        structural_pool_center=structural_pool_center,
    ).to(device)
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        resume_payload = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(resume_payload["model"])
    # The online model remains the one optimized by AdamW. EMA only provides a
    # smoother parameter snapshot for validation/checkpointing/evaluation.
    ema_model = _make_ema_model(model) if args.ema_decay else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, min_lr=1e-5
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path: Path | None = None
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = args.diagnostics_dir / "training_dynamics.jsonl"
        diagnostics_path.write_text("", encoding="utf-8")
        (args.diagnostics_dir / "README.md").write_text(
            "# FoldAbS training-dynamics audit\n\n"
            "Each JSONL line is one validation check. Gradient norms are recorded after backpropagation "
            "but before clipping. Activation summaries use validation samples only; the test split is never "
            "used during training or checkpoint selection.\n",
            encoding="utf-8",
        )
    split_manifest = pd.concat(
        [partition.assign(split=name) for name, partition in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))],
        ignore_index=True,
    )
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    if args.save_online_checkpoint_with_ema and not args.ema_decay:
        raise ValueError("--save-online-checkpoint-with-ema requires --ema-decay > 0.")
    initial_checkpoint_validation_mse: float | None = None
    online_best_validation_mse: float | None = None
    online_best_step: int | None = None
    if args.resume_checkpoint is not None:
        selection_model = ema_model if ema_model is not None else model
        initial_checkpoint_validation_mse, _, _, _ = _evaluate(selection_model, loaders["validation"], device, normalizer)
        best_validation_mse, best_step = initial_checkpoint_validation_mse, 0
        torch.save(
            {"model": selection_model.state_dict(), "step": 0, "ema_decay": args.ema_decay},
            output_dir / "best_model.pt",
        )
        print(
            json.dumps(
                {
                    "resume_checkpoint": str(args.resume_checkpoint),
                    "initial_validation_normalized_mse": initial_checkpoint_validation_mse,
                }
            ),
            flush=True,
        )
    else:
        best_validation_mse, best_step = float("inf"), 0
    if args.save_online_checkpoint_with_ema:
        online_best_validation_mse, online_best_step = float("inf"), 0
    history: list[dict[str, float | None]] = []
    rolling_train_losses: deque[float] = deque(maxlen=args.train_loss_window)
    bad_validation_checks = 0
    train_iter = iter(loaders["train"])
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses, micro_predictions, micro_targets = [], [], []
        if step <= args.warmup_steps:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * step / args.warmup_steps
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                batch = next(train_iter)
            prediction = model(
                batch["s"].to(device), batch["z"].to(device), batch["p_dist"].to(device),
                batch["e_h"].to(device), batch["e_l"].to(device),
            )
            loss = F.mse_loss(prediction, batch["label"].to(device).reshape_as(prediction))
            micro_losses.append(float(loss.item()))
            micro_predictions.append(float(prediction.detach().item()))
            micro_targets.append(float(batch["label"].item()))
            (loss / args.gradient_accumulation).backward()
        gradient_norms_before_clip = _module_norms(model, gradients=True)
        total_gradient_norm_before_clip = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        )
        optimizer.step()
        if ema_model is not None:
            _update_ema_model(ema_model, model, args.ema_decay)
        step_train_mse = float(np.mean(micro_losses))
        rolling_train_losses.append(step_train_mse)

        if step % args.validate_every == 0 or step == args.max_steps:
            # Keep the online validation loss as a diagnostic. The EMA loss is
            # the actual selection metric whenever --ema-decay is enabled.
            online_validation_mse: float | None = None
            online_improved: bool | None = None
            selection_model = ema_model if ema_model is not None else model
            if ema_model is not None:
                online_validation_mse, _, _, _ = _evaluate(
                    model,
                    loaders["validation"],
                    device,
                    normalizer,
                )
                if args.save_online_checkpoint_with_ema:
                    online_improved = online_validation_mse < float(online_best_validation_mse)
                    if online_improved:
                        online_best_validation_mse, online_best_step = online_validation_mse, step
                        torch.save(
                            {"model": model.state_dict(), "step": step, "ema_decay": args.ema_decay},
                            output_dir / "online_best_model.pt",
                        )
            validation_result = _evaluate(
                selection_model,
                loaders["validation"],
                device,
                normalizer,
                collect_intermediates=diagnostics_path is not None,
            )
            if diagnostics_path is not None:
                validation_mse, validation_labels, validation_predictions, validation_ids, activation_summaries = validation_result
            else:
                validation_mse, validation_labels, validation_predictions, validation_ids = validation_result
            scheduler.step(validation_mse)
            improved = validation_mse < best_validation_mse
            if improved:
                bad_validation_checks = 0
            else:
                bad_validation_checks += 1
            history_row = {
                "step": step,
                "train_mse": step_train_mse,
                "train_mse_rolling": float(np.mean(rolling_train_losses)),
                "validation_normalized_mse": validation_mse,
                "online_validation_normalized_mse": online_validation_mse,
                "online_improved": online_improved,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "bad_validation_checks": bad_validation_checks,
            }
            history.append(history_row)
            print(json.dumps(history_row), flush=True)
            if diagnostics_path is not None:
                diagnostics_row = {
                    "step": step,
                    "train": {
                        "optimizer_step_mse": step_train_mse,
                        "rolling_mse": float(np.mean(rolling_train_losses)),
                        "microbatch_prediction_normalized": {
                            "mean": float(np.mean(micro_predictions)),
                            "std": float(np.std(micro_predictions)),
                        },
                        "microbatch_label_normalized": {
                            "mean": float(np.mean(micro_targets)),
                            "std": float(np.std(micro_targets)),
                        },
                    },
                    "optimization": {
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "total_gradient_norm_before_clip": total_gradient_norm_before_clip,
                        "gradient_clipping_active": bool(total_gradient_norm_before_clip > 1.0),
                        "gradient_norm_by_module_before_clip": gradient_norms_before_clip,
                        "parameter_norm_by_module_after_step": _module_norms(model, gradients=False),
                    },
                    "validation": {
                        "normalized_mse": validation_mse,
                        "raw": {
                            **regression_metrics(validation_labels, validation_predictions),
                            "label_mean_celsius": float(np.mean(validation_labels)),
                            "label_std_celsius": float(np.std(validation_labels)),
                            "prediction_mean_celsius": float(np.mean(validation_predictions)),
                            "prediction_std_celsius": float(np.std(validation_predictions)),
                        },
                        "by_source": _source_metrics(
                            splits.validation,
                            validation_ids,
                            validation_labels,
                            validation_predictions,
                        ),
                        "activation_summary": activation_summaries,
                    },
                    "checkpoint": {
                        "improved": bool(improved),
                        "bad_validation_checks": int(bad_validation_checks),
                    },
                }
                with diagnostics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostics_row, ensure_ascii=False) + "\n")
            if improved:
                best_validation_mse, best_step = validation_mse, step
                torch.save(
                    {"model": selection_model.state_dict(), "step": step, "ema_decay": args.ema_decay},
                    output_dir / "best_model.pt",
                )
            if args.early_stopping_patience and bad_validation_checks >= args.early_stopping_patience:
                print(
                    f"Early stopping at step {step}: no validation improvement for "
                    f"{bad_validation_checks} checks.",
                    flush=True,
                )
                break

    def evaluate_saved_checkpoint(checkpoint_path: Path) -> tuple[dict, dict, list[pd.DataFrame]]:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        metric_rows, raw_metric_rows, frames = {}, {}, []
        for name, partition in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
            normalized_mse, labels, predictions, sample_ids = _evaluate(model, loaders[name], device, normalizer)
            metric_rows[name] = {**regression_metrics(labels, predictions), "normalized_mse": normalized_mse}
            raw_labels = task_spec.inverse_preprocessing(labels)
            raw_predictions = task_spec.inverse_preprocessing(predictions)
            raw_metric_rows[name] = regression_metrics(raw_labels, raw_predictions)
            result = partition.set_index("sample_id").loc[sample_ids].reset_index()[
                ["sample_id", "heavy", "light", "label"]
            ]
            result.insert(1, "split", name)
            result["prediction"] = predictions
            result["raw_label"] = raw_labels
            result["raw_prediction"] = raw_predictions
            frames.append(result)
        return metric_rows, raw_metric_rows, frames

    metrics, raw_metrics, prediction_frames = evaluate_saved_checkpoint(output_dir / "best_model.pt")
    online_selection_summary: dict | None = None
    if args.save_online_checkpoint_with_ema:
        online_metrics, online_raw_metrics, online_prediction_frames = evaluate_saved_checkpoint(
            output_dir / "online_best_model.pt"
        )
        online_selection_summary = {
            "selection_metric": "online validation normalized MSE",
            "best_validation_normalized_mse": online_best_validation_mse,
            "best_step": online_best_step,
            "metrics": online_metrics,
            "raw_label_metrics": online_raw_metrics,
        }
        (output_dir / "online_metrics.json").write_text(
            json.dumps(online_selection_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.concat(online_prediction_frames, ignore_index=True).to_csv(
            output_dir / "online_predictions.csv", index=False
        )
    cumulative_uses_esm = args.cumulative_ablation_stage == "full"
    effective_esm = not args.disable_esm and (args.cumulative_ablation_stage is None or cumulative_uses_esm)
    effective_pairformer = not args.disable_pairformer and args.cumulative_ablation_stage not in {"single_only", "distogram_pair"}
    effective_distance_conditioning = not args.disable_dist_conditioning and args.cumulative_ablation_stage not in {"single_only", "distogram_pair"}
    summary = {
        "experiment_name": (
            "foldabs_table_s8_cumulative_ablation"
            if args.cumulative_ablation_stage is not None
            else "foldabs_ema_engineering_control"
            if args.ema_decay != 0.0
            else "foldabs_paper_aligned_developability"
            if not any((args.disable_esm, args.normalize_esm_embeddings, args.branch_layernorm, args.fusion_normalization != "legacy_global", args.structural_component_layernorm, args.pre_pairformer_layernorm, args.esm_context_dropout != 0.0, args.structure_readout != "s_z", args.disable_dist_conditioning, args.disable_pairformer, args.paircond_mode != "z_aware_transition", args.pairformer_n_heads != 16, args.pairformer_c_hidden_mul != 128, args.pairformer_c_hidden_pair_att != 32, args.pairformer_no_heads_pair != 4, args.regressor_hidden_dim != 512))
            else "foldabs_fusion_diagnostic"
        ),
        "scope": (
            "Frozen Protenix-v1 {s,z,p_dist}; 2-transition PairCond; 1 Pairformer; frozen ESM-2 650M; 3-layer MLP"
            if effective_esm
            else "Frozen Protenix-v1 {s,z,p_dist}; 2-transition PairCond; 1 Pairformer; same training recipe; no ESM-2 input"
        ),
        "not_the_foldabs_paper_model": bool(
            args.disable_esm
            or args.normalize_esm_embeddings
            or args.branch_layernorm
            or args.fusion_normalization != "legacy_global"
            or args.structural_component_layernorm
            or args.pre_pairformer_layernorm
            or args.esm_context_dropout != 0.0
            or args.ema_decay != 0.0
            or args.structure_readout != "s_z"
            or args.disable_dist_conditioning
            or args.disable_pairformer
            or args.paircond_mode != "z_aware_transition"
            or args.pairformer_n_heads != 16
            or args.pairformer_c_hidden_mul != 128
            or args.pairformer_c_hidden_pair_att != 32
            or args.pairformer_no_heads_pair != 4
            or args.regressor_hidden_dim != 512
            or args.cumulative_ablation_stage not in {None, "full"}
            or args.resume_checkpoint is not None
        ),
        "developability_property": task_spec.name,
        "not_exact_paper_reproduction": bool(
            len(frame) != task_spec.paper_rows
            or args.protenix_msa_source != "uniref50"
            or not args.exact_paper_manifest
        ),
        "input_alignment": {
            "property": task_spec.name,
            "paper_dataset_rows": task_spec.paper_rows,
            "current_rows": int(len(frame)),
            "paper_antibody_msa_source": "UniRef50-only",
            "current_protenix_msa_source": args.protenix_msa_source,
            "exact_paper_manifest_available": bool(args.exact_paper_manifest),
            "label_preprocessing_before_train_zscore": task_spec.preprocessing,
            "model_label_unit": task_spec.model_label_unit,
            "raw_label_unit": task_spec.raw_label_unit,
        },
        "paper_alignment": {
            "frozen_protenix": True,
            "distance_conditioning": "softmax(p_dist) + 2 residual transitions" if effective_distance_conditioning else "not used in this ablation row",
            "pairformer_blocks": 1 if effective_pairformer else 0,
            "esm2_heavy_light_fusion": effective_esm,
            "selection_metric": "validation normalized MSE",
        },
        "training_control": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation": args.gradient_accumulation,
            "validate_every": args.validate_every,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience_validation_checks": args.scheduler_patience,
            "minimum_learning_rate": 1e-5,
            "early_stopping_patience_validation_checks": args.early_stopping_patience,
            "train_loss_rolling_window_steps": args.train_loss_window,
            "ema": {
                "enabled": bool(args.ema_decay),
                "decay": args.ema_decay,
                "selection_and_final_evaluation_use_ema": bool(args.ema_decay),
                "online_validation_mse_logged_for_diagnostic": bool(args.ema_decay),
                "online_best_checkpoint_saved": bool(args.save_online_checkpoint_with_ema),
            },
            "esm_context_dropout": args.esm_context_dropout,
            "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint is not None else None,
            "resume_optimizer_state": False,
            "initial_checkpoint_validation_normalized_mse": initial_checkpoint_validation_mse,
            "diagnostics_jsonl": str(diagnostics_path.resolve()) if diagnostics_path is not None else None,
        },
        "cumulative_ablation": (
            {
                "stage": args.cumulative_ablation_stage,
                "rows": {
                    "single_only": "Pool1D(s) only",
                    "distogram_pair": "Pool1D(s) + mean pooled linear projection of softmax(p_dist)",
                    "pairformer": "distance-conditioned z, one Pairformer, then pooled s and z",
                    "full": "pairformer stage plus frozen ESM-2 heavy/light embeddings",
                },
                "reconstruction_boundary": (
                    "Paper architecture is followed from the method description. Exact author code and "
                    "the unpublished distogram projection width are unavailable, so this is a documented "
                    "paper-informed reconstruction rather than a byte-for-byte reproduction."
                ),
            }
            if args.cumulative_ablation_stage is not None
            else None
        ),
        "diagnostic_configuration": {
            "train_split_only_esm_standardization": args.normalize_esm_embeddings,
            "independent_branch_layernorm": args.branch_layernorm,
            "fusion_normalization": "branch_then_global" if args.branch_layernorm else args.fusion_normalization,
            "structural_component_layernorm": args.structural_component_layernorm,
            "pre_pairformer_layernorm": args.pre_pairformer_layernorm,
            "single_pool_train_centering": {
                "enabled": args.center_single_pool_train,
                "fit_partition": "train" if args.center_single_pool_train else None,
                "vector_width": int(structural_pool_center.numel()) if structural_pool_center is not None else None,
                "vector_rms": float(structural_pool_center.square().mean().sqrt().item()) if structural_pool_center is not None else None,
            },
            "esm_context_dropout": args.esm_context_dropout,
            "structural_readout": args.structure_readout,
            "distance_conditioning_disabled": args.disable_dist_conditioning,
            "pairformer_disabled": args.disable_pairformer,
            "paircond_mode": args.paircond_mode,
            "pairformer_configuration": {
                "n_heads": args.pairformer_n_heads,
                "c_hidden_mul": args.pairformer_c_hidden_mul,
                "c_hidden_pair_att": args.pairformer_c_hidden_pair_att,
                "no_heads_pair": args.pairformer_no_heads_pair,
                "dropout": args.pairformer_dropout,
            },
            "regressor_configuration": {
                "hidden_dim": args.regressor_hidden_dim,
                "hidden_layers": 2,
                "dropout": 0.1,
            },
        },
        "assumptions": {
            "pooling": "unmasked mean Pool1D/Pool2D; exact paper pooling implementation is not published",
            "nanobody_light_embedding": "zero vector when no light chain is present",
            "paircond_internals": (
                f"{args.paircond_mode} reconstruction; exact author PairCond internals are not published"
            ),
        },
        "dataset_rows": int(len(frame)),
        "split_sizes": {name: int(len(partition)) for name, partition in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))},
        "best_validation_normalized_mse": best_validation_mse,
        "best_step": best_step,
        "metrics": metrics,
        "raw_label_metrics": raw_metrics,
        "online_selection_control": online_selection_summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
