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
from .triangle_attention import install_cueq_all_valid_mask_workaround


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
    """Pairwise conditioning: trunk distogram baseline or FoldAbS d_geom + PAE."""
    def __init__(self, c_dist: int, c_z: int, c_geom: int | None = None,
                 c_pae: int | None = None):
        super().__init__()
        self.trunk_project = nn.Sequential(nn.LayerNorm(c_dist), nn.Linear(c_dist, c_z))
        self.specificity_project = (
            nn.Sequential(nn.LayerNorm(c_geom + c_pae), nn.Linear(c_geom + c_pae, c_z))
            if c_geom is not None and c_pae is not None else None
        )
        self.transitions = nn.ModuleList([ResidualTransition(c_z), ResidualTransition(c_z)])

    def forward(self, z: torch.Tensor, p_dist: torch.Tensor | None,
                d_geom: torch.Tensor | None = None,
                p_pae: torch.Tensor | None = None) -> torch.Tensor:
        if self.specificity_project is not None:
            if d_geom is None or p_pae is None:
                raise ValueError("use_d_geom=True requires d_geom and p_pae")
            expected = (*z.shape[:2], self.specificity_project[0].normalized_shape[0] // 2)
            if d_geom.shape != expected or p_pae.shape != expected:
                raise ValueError("d_geom/p_pae must both be [N_token,N_token,64]")
            condition = torch.cat([d_geom, p_pae], dim=-1)
            value = z + self.specificity_project(condition)
        else:
            if p_dist is None:
                raise ValueError("Trunk-only PairCondition requires p_dist")
            if d_geom is not None or p_pae is not None:
                raise ValueError("d_geom/p_pae require use_d_geom=True")
            value = z + self.trunk_project(p_dist)
        for transition in self.transitions:
            value = transition(value)
        return value


class TMPTrunkSpecificityHead(nn.Module):
    """Trainable TMP classifier over frozen Protenix representations.

    Trunk-only by default. The optional specificity mode follows FoldAbS:
    concatenate diffusion-derived d_geom and PAE probabilities for PairCond,
    then fuse a 32-dimensional ipTM embedding after interface pooling.
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
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        activation_checkpoint: bool = False,
        use_d_geom: bool = False,
    ):
        super().__init__()
        if pairformer_blocks < 0:
            raise ValueError("pairformer_blocks cannot be negative")
        if triangle_multiplicative not in ('torch', 'cuequivariance') or triangle_attention not in ('torch', 'cuequivariance'):
            raise ValueError('Unsupported head triangle backend')
        if triangle_attention == 'cuequivariance':
            install_cueq_all_valid_mask_workaround()
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
            "triangle_multiplicative": triangle_multiplicative,
            "triangle_attention": triangle_attention,
            "activation_checkpoint": activation_checkpoint,
            "use_d_geom": use_d_geom,
        }
        self.pair_condition = PairCondition(
            c_dist, c_z, c_geom=64 if use_d_geom else None,
            c_pae=64 if use_d_geom else None,
        )
        self.iptm_embedding = (
            nn.Sequential(nn.Linear(1, 32), nn.GELU()) if use_d_geom else None
        )
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
        input_dim = 4 * c_s + 5 * c_z + (32 if use_d_geom else 0)
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
        *,
        d_geom: torch.Tensor | None = None,
        p_pae: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = {
            key: value.to(device=s.device)
            for key, value in build_role_masks(token_asym_id, roles).items()
        }
        # Protenix 1.0.0's cuEquivariance adapter adds its own batch axis.
        # Our public interface is one complex, so avoid adding a second axis.
        unbatched_kernel = 'cuequivariance' in (
            self.config['triangle_multiplicative'], self.config['triangle_attention'])
        s_batch = s if unbatched_kernel else s.unsqueeze(0)
        z_conditioned = self.pair_condition(z, p_dist, d_geom, p_pae)
        z_batch = z_conditioned if unbatched_kernel else z_conditioned.unsqueeze(0)
        for block in self.pairformer:
            def run_block(s_value, z_value, current_block=block):
                return current_block(
                    s=s_value, z=z_value, pair_mask=None,
                    triangle_multiplicative=self.config['triangle_multiplicative'],
                    triangle_attention=self.config['triangle_attention'],
                    inplace_safe=False, chunk_size=None)
            if self.config['activation_checkpoint'] and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                s_batch, z_batch = checkpoint(run_block, s_batch, z_batch, use_reentrant=False)
            else:
                s_batch, z_batch = run_block(s_batch, z_batch)
        pools = self.pooling(s_batch if unbatched_kernel else s_batch.squeeze(0),
                             z_batch if unbatched_kernel else z_batch.squeeze(0), masks)
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
        d_geom: torch.Tensor | None = None,
        p_pae: torch.Tensor | None = None,
        iptm: torch.Tensor | float | None = None,
        return_latent: bool = False,
    ):
        latent, pools = self.encode(
            s, z, p_dist, token_asym_id, roles, d_geom=d_geom, p_pae=p_pae
        )
        if self.iptm_embedding is not None:
            if iptm is None:
                raise ValueError("use_d_geom=True requires the Protenix ipTM scalar")
            iptm_tensor = torch.as_tensor(iptm, device=latent.device, dtype=latent.dtype).reshape(1, 1)
            iptm_embedding = self.iptm_embedding(iptm_tensor).reshape(-1)
            classifier_input = torch.cat([latent, iptm_embedding], dim=-1)
        else:
            if p_pae is not None or iptm is not None:
                raise ValueError("p_pae/iptm require use_d_geom=True")
            classifier_input = latent
        logit = self.classifier(classifier_input.unsqueeze(0)).squeeze()
        return (logit, latent, pools) if return_latent else logit
