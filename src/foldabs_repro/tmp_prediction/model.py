from __future__ import annotations

import os

# Use Protenix's portable implementation for both CPU smoke tests and GPUs
# without the optional precompiled fused LayerNorm extension.  This must be set
# before importing PairformerBlock.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch
import torch.nn as nn

from protenix.model.modules.pairformer import PairformerBlock

from .interfaces import POOL_ORDER, build_role_masks
from .pooling import InterfacePooling


class ResidualTransition(nn.Module):
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
    """Condition frozen Protenix z on its distogram probabilities."""
    def __init__(self, c_dist: int, c_z: int):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(c_dist), nn.Linear(c_dist, c_z))
        self.transitions = nn.ModuleList([ResidualTransition(c_z), ResidualTransition(c_z)])

    def forward(self, z: torch.Tensor, p_dist: torch.Tensor) -> torch.Tensor:
        value = z + self.project(p_dist)
        for transition in self.transitions:
            value = transition(value)
        return value


class TMPTrunkSpecificityHead(nn.Module):
    """Trainable TMP classifier over frozen s/z/p_dist trunk representations.

    This is a phase-1 FoldAbS-style reconstruction. It deliberately does not
    claim to be the published antibody specificity head because the current
    extractor does not provide final geometry, PAE or ipTM.
    """
    def __init__(
        self,
        *,
        c_s: int,
        c_z: int,
        c_dist: int,
        pairformer_blocks: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        n_heads: int = 16,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        pooling_mode: str = "A",
    ):
        super().__init__()
        if pairformer_blocks < 0:
            raise ValueError("pairformer_blocks cannot be negative")
        self.config = {
            "c_s": c_s,
            "c_z": c_z,
            "c_dist": c_dist,
            "pairformer_blocks": pairformer_blocks,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "n_heads": n_heads,
            "c_hidden_mul": c_hidden_mul,
            "c_hidden_pair_att": c_hidden_pair_att,
            "no_heads_pair": no_heads_pair,
            "pooling_mode": pooling_mode,
        }
        self.pair_condition = PairCondition(c_dist, c_z)
        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(
                    n_heads=n_heads,
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden_mul=c_hidden_mul,
                    c_hidden_pair_att=c_hidden_pair_att,
                    no_heads_pair=no_heads_pair,
                    dropout=dropout,
                )
                for _ in range(pairformer_blocks)
            ]
        )
        # A introduces no parameters/state keys, preserving legacy checkpoints.
        # Zero-query B/C construction consumes no RNG, preserving common weight
        # initialization for comparisons using the same seed.
        self.pooling = InterfacePooling(c_s, c_z, mode=pooling_mode)
        # Four entity-level s pools plus five biologically distinct interfaces.
        input_dim = 4 * c_s + 5 * c_z
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        p_dist: torch.Tensor,
        token_asym_id: torch.Tensor,
        roles: tuple[str, ...] | list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = {
            key: value.to(device=s.device)
            for key, value in build_role_masks(token_asym_id, roles).items()
        }
        s_batch = s.unsqueeze(0)
        z_batch = self.pair_condition(z, p_dist).unsqueeze(0)
        for block in self.pairformer:
            s_batch, z_batch = block(
                s=s_batch,
                z=z_batch,
                pair_mask=None,
                triangle_multiplicative="torch",
                triangle_attention="torch",
                inplace_safe=False,
                chunk_size=None,
            )
        pools = self.pooling(s_batch.squeeze(0), z_batch.squeeze(0), masks)
        latent = torch.cat([pools[name] for name in POOL_ORDER], dim=-1)
        return latent, pools

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        p_dist: torch.Tensor,
        token_asym_id: torch.Tensor,
        roles: tuple[str, ...] | list[str],
        *,
        return_latent: bool = False,
    ):
        latent, pools = self.encode(s, z, p_dist, token_asym_id, roles)
        logit = self.classifier(latent.unsqueeze(0)).squeeze()
        return (logit, latent, pools) if return_latent else logit
