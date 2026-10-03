from __future__ import annotations

from collections.abc import Sequence

import torch


REQUIRED_ROLES = ("tcra", "tcrb", "peptide", "mhc")
PAIR_INTERFACES = (
    ("tcra", "peptide"),
    ("tcrb", "peptide"),
    ("tcra", "mhc"),
    ("tcrb", "mhc"),
    ("peptide", "mhc"),
)
POOL_ORDER = tuple(f"s_{role}" for role in REQUIRED_ROLES) + tuple(
    f"z_{left}_{right}" for left, right in PAIR_INTERFACES
)


def build_role_masks(token_asym_id: torch.Tensor, roles: Sequence[str]) -> dict[str, torch.Tensor]:
    """Map Protenix chains to persisted input roles using chain order."""
    asym = token_asym_id.reshape(-1).to(dtype=torch.int64)
    unique = torch.unique(asym, sorted=True).tolist()
    if len(unique) != len(roles):
        raise ValueError(
            f"Feature has {len(unique)} chains but manifest declares {len(roles)} roles"
        )
    masks = {role: asym.eq(int(asym_id)) for role, asym_id in zip(roles, unique)}
    missing = [role for role in REQUIRED_ROLES if role not in masks]
    if missing:
        raise ValueError(f"Missing required role masks: {missing}")
    return masks


def masked_mean_1d(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or mask.ndim != 1 or value.shape[0] != mask.shape[0]:
        raise ValueError("Expected value [N,C] and mask [N]")
    if not bool(mask.any()):
        raise ValueError("Cannot pool an empty entity")
    return value[mask].mean(dim=0)


def masked_mean_2d(value: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[:2] != (left.numel(), right.numel()):
        raise ValueError("Expected value [N,N,C] and two [N] masks")
    pair_mask = left[:, None] & right[None, :]
    if not bool(pair_mask.any()):
        raise ValueError("Cannot pool an empty interface")
    return value[pair_mask].mean(dim=0)


def interface_pools(
    s: torch.Tensor,
    z: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Parameter-free entity and directed-interface summaries."""
    result = {f"s_{role}": masked_mean_1d(s, masks[role]) for role in REQUIRED_ROLES}
    for left, right in PAIR_INTERFACES:
        result[f"z_{left}_{right}"] = masked_mean_2d(z, masks[left], masks[right])
    return result
