"""A/B/C readouts with the same entity/interface masks and latent dimension."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .interfaces import (
    PAIR_INTERFACES,
    REQUIRED_ROLES,
    masked_mean_1d,
    masked_mean_2d,
)


POOLING_MODES = ("A", "B", "C")


class QueryAttentionPool(nn.Module):
    """One query per pool; zero initialization starts at uniform mean pooling.

    Layer normalization is used only for scoring. The weighted values retain
    their original scale. No affine projection, temperature, or extra MLP is
    learned, so each pool introduces exactly ``channels`` parameters.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.query = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[-1] != self.query.numel():
            raise ValueError("Expected non-empty values [M, channels]")
        if values.shape[0] == 0:
            raise ValueError("Cannot pool an empty entity or interface")
        # Score/softmax in float32 for stability with half-precision inputs.
        keys = F.layer_norm(values.float(), (values.shape[-1],))
        scores = keys @ self.query.float()
        weights = torch.softmax(scores, dim=0).to(values.dtype)
        return (weights[:, None] * values).sum(dim=0)


class InterfacePooling(nn.Module):
    """A: all mean; B: single mean/pair attention; C: all attention.

    Attention is normalized separately inside each entity/interface. Directions
    and pool ordering are exactly those of the existing Stage 1 mean readout.
    """

    def __init__(self, c_s: int, c_z: int, mode: str = "A"):
        super().__init__()
        if mode not in POOLING_MODES:
            raise ValueError(f"pooling_mode must be one of {POOLING_MODES}")
        self.mode = mode
        self.single_attention = nn.ModuleDict()
        self.pair_attention = nn.ModuleDict()
        if mode == "C":
            self.single_attention.update(
                {role: QueryAttentionPool(c_s) for role in REQUIRED_ROLES}
            )
        if mode in ("B", "C"):
            self.pair_attention.update(
                {
                    f"{left}_{right}": QueryAttentionPool(c_z)
                    for left, right in PAIR_INTERFACES
                }
            )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor, masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if s.ndim != 2 or z.ndim != 3 or z.shape[:2] != (s.shape[0], s.shape[0]):
            raise ValueError("Expected s [N,C_s] and z [N,N,C_z]")
        for role in REQUIRED_ROLES:
            mask = masks[role]
            if mask.dtype != torch.bool or mask.shape != (s.shape[0],):
                raise ValueError(f"Expected a boolean [N] mask for {role}")
        result = {}
        for role in REQUIRED_ROLES:
            result[f"s_{role}"] = (
                self.single_attention[role](s[masks[role]])
                if role in self.single_attention
                else masked_mean_1d(s, masks[role])
            )
        for left, right in PAIR_INTERFACES:
            key = f"{left}_{right}"
            if key in self.pair_attention:
                pair_mask = masks[left][:, None] & masks[right][None, :]
                pooled = self.pair_attention[key](z[pair_mask])
            else:
                pooled = masked_mean_2d(z, masks[left], masks[right])
            result[f"z_{key}"] = pooled
        return result
