"""Frozen Protenix diffusion geometry for the optional TMP d_geomt ablation.

The public name `d_geomt` denotes a 64-channel distribution of binned
C-alpha distances measured from sampled coordinates. It is *not* the trunk
distogram `p_dist` and is not a native Protenix output tensor.

Implementation choice (not specified exactly by FoldAbS): 64 bins over
[0, 22] Angstrom, last bin includes overflow; average one-hot histograms
over multiple diffusion samples without averaging coordinates.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

GEOMETRY_BINS = 64
GEOMETRY_MAX_DISTANCE = 22.0


def ca_atom_indices(features: dict, n_tokens: int) -> torch.Tensor:
    """Return the exact predicted C-alpha atom index for every protein token.

    Protenix 1.0.0 encodes atom names using (ASCII - 32) one-hot characters
    padded to four positions. TMP currently requires four *protein* chains.
    Reject incomplete or duplicate token-to-CA mappings rather than silently
    inventing a coordinate from another atom.
    """
    mapping = features["atom_to_token_idx"].reshape(-1).long()
    names = features["ref_atom_name_chars"]
    if names.shape != (mapping.numel(), 4, 64):
        raise ValueError("Unexpected ref_atom_name_chars layout (expected [N_atom,4,64])")
    characters = names.argmax(dim=-1)
    is_ca = ((characters[:, 0] == ord("C") - 32)
             & (characters[:, 1] == ord("A") - 32)
             & (characters[:, 2] == 0) & (characters[:, 3] == 0))
    atoms = torch.where(is_ca)[0]
    tokens = mapping[atoms]
    if (atoms.numel() != n_tokens or tokens.min().item() < 0
            or tokens.max().item() >= n_tokens
            or not torch.equal(tokens.sort().values, torch.arange(n_tokens, device=tokens.device))):
        raise ValueError("Every TMP protein token must have exactly one CA atom")
    indices = torch.empty(n_tokens, device=mapping.device, dtype=torch.long)
    indices[tokens] = atoms
    return indices


def d_geomt_from_coordinates(
    coordinates: torch.Tensor, features: dict, n_tokens: int,
    *, bins: int = GEOMETRY_BINS, max_distance: float = GEOMETRY_MAX_DISTANCE,
) -> torch.Tensor:
    """Convert [N_sample,N_atom,3] diffusion coordinates to [N,N,64].

    Averaging categorical distances over samples is invariant to each sample's
    independent rigid rotation/translation and retains ensemble variability.
    """
    if bins < 2 or max_distance <= 0:
        raise ValueError("Invalid geometric bin configuration")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or not coordinates.shape[0]:
        raise ValueError("Coordinates must have shape [N_sample,N_atom,3]")
    if coordinates.shape[-2] != features["atom_to_token_idx"].numel():
        raise ValueError("Atom count differs from Protenix features")
    if not torch.isfinite(coordinates).all():
        raise ValueError("Diffusion returned non-finite coordinates")
    indices = ca_atom_indices(features, n_tokens)
    positions = coordinates.float().index_select(-2, indices)
    distances = torch.cdist(positions, positions)
    # bins-1 thresholds: [0, step), ..., [22, infinity).
    boundaries = torch.linspace(0, max_distance, bins, device=distances.device)[1:]
    assignments = torch.bucketize(distances.contiguous(), boundaries)
    d_geomt = F.one_hot(assignments.long(), num_classes=bins).float().mean(dim=0)
    if d_geomt.shape != (n_tokens, n_tokens, bins) or not torch.isfinite(d_geomt).all():
        raise ValueError("Invalid d_geomt geometry")
    return d_geomt


@torch.no_grad()
def sample_d_geomt(
    model, features: dict, s_inputs: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
    *, steps: int = 20, samples: int = 1,
) -> torch.Tensor:
    """Call official Protenix-v1 diffusion from *existing* frozen trunk states."""
    if steps < 1 or samples < 1:
        raise ValueError("diffusion steps/samples must be positive")
    from protenix.utils.torch_utils import autocasting_disable_decorator

    cache_pair = cache_lm = cache_lig = None
    if model.enable_diffusion_shared_vars_cache:
        no_amp = autocasting_disable_decorator(model.configs.skip_amp.sample_diffusion)
        cache_pair = no_amp(model.diffusion_module.diffusion_conditioning.prepare_cache)(
            features["relp"], z, False,
        )
        cache_lm, cache_lig = no_amp(model.diffusion_module.atom_attention_encoder.prepare_cache)(
            ref_pos=features["ref_pos"],
            ref_charge=features["ref_charge"],
            ref_mask=features["ref_mask"],
            ref_element=features["ref_element"],
            ref_atom_name_chars=features["ref_atom_name_chars"],
            atom_to_token_idx=features["atom_to_token_idx"],
            d_lm=features["d_lm"],
            v_lm=features["v_lm"],
            pad_info=features["pad_info"],
            r_l=True,
            z=cache_pair,
            inplace_safe=False,
        )

    schedule = model.inference_noise_scheduler(
        N_step=steps, device=s_inputs.device, dtype=s_inputs.dtype,
    )
    coords = model.sample_diffusion(
        denoise_net=model.diffusion_module,
        input_feature_dict=features,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=None if cache_pair is not None else z,
        pair_z=cache_pair,
        p_lm=cache_lm,
        c_l=cache_lig,
        N_sample=samples,
        noise_schedule=schedule,
        inplace_safe=False,
        enable_efficient_fusion=model.enable_efficient_fusion,
    )
    return d_geomt_from_coordinates(coords, features, n_tokens=s.shape[-2])
