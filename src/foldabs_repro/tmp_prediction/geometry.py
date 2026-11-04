"""FoldAbS-style geometry, PAE and ipTM extraction from frozen Protenix."""
from __future__ import annotations

import torch

GEOMETRY_BINS = 64
GEOMETRY_MAX_DISTANCE = 22.0
PAE_BINS = 64


def representative_atom_indices(features: dict, n_tokens: int) -> torch.Tensor:
    """Map each Protenix token to its designated representative atom.

    Protenix exposes the representative-atom mask as distogram_rep_atom_mask. This
    is preferable to hard-coding atom_name == CA: it follows Protenix's own
    token-centre/representative-atom definition and its token mapping.
    """
    mapping = features["atom_to_token_idx"].reshape(-1).long()
    rep_mask = features.get("pae_rep_atom_mask")
    if rep_mask is None:
        raise ValueError("Protenix features must contain pae_rep_atom_mask")
    rep_mask = rep_mask.reshape(-1).bool()
    if rep_mask.numel() != mapping.numel():
        raise ValueError("pae_rep_atom_mask and atom_to_token_idx lengths differ")
    atoms = torch.where(rep_mask)[0]
    tokens = mapping[atoms]
    expected = torch.arange(n_tokens, device=mapping.device)
    if (atoms.numel() != n_tokens or tokens.numel() != n_tokens
            or tokens.min().item() < 0 or tokens.max().item() >= n_tokens
            or not torch.equal(tokens.sort().values, expected)):
        raise ValueError("Every Protenix token must have exactly one representative atom")
    indices = torch.empty(n_tokens, device=mapping.device, dtype=torch.long)
    indices[tokens] = atoms
    return indices


def d_geom_from_coordinates(
    coordinates: torch.Tensor, features: dict, n_tokens: int,
    *, bins: int = GEOMETRY_BINS, max_distance: float = GEOMETRY_MAX_DISTANCE,
) -> torch.Tensor:
    """Convert [N_sample,N_atom,3] to an ensemble [N_token,N_token,64] histogram."""
    if bins < 2 or max_distance <= 0:
        raise ValueError("Invalid geometry bin configuration")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3 or coordinates.shape[0] < 1:
        raise ValueError("Coordinates must have shape [N_sample,N_atom,3]")
    if coordinates.shape[-2] != features["atom_to_token_idx"].numel():
        raise ValueError("Atom count differs from Protenix features")
    if not torch.isfinite(coordinates).all():
        raise ValueError("Diffusion returned non-finite coordinates")

    atom_indices = representative_atom_indices(features, n_tokens)
    representative_xyz = coordinates.float().index_select(-2, atom_indices)
    distances = torch.cdist(representative_xyz, representative_xyz)
    boundaries = torch.linspace(0, max_distance, bins, device=distances.device)[1:]
    assignments = torch.bucketize(distances.contiguous(), boundaries)

    # Stream sample histograms to avoid allocating [samples,N,N,64].
    d_geom = torch.zeros((n_tokens, n_tokens, bins), device=distances.device, dtype=torch.float32)
    weights = torch.ones((n_tokens, n_tokens, 1), device=distances.device, dtype=torch.float32)
    for sample_bins in assignments:
        d_geom.scatter_add_(-1, sample_bins.long().unsqueeze(-1), weights)
    d_geom.div_(coordinates.shape[0])
    if (d_geom.shape != (n_tokens, n_tokens, bins)
            or not torch.isfinite(d_geom).all()
            or not torch.allclose(d_geom.sum(-1), torch.ones_like(d_geom[..., 0]), atol=1e-5)):
        raise ValueError("Invalid d_geom geometry histogram")
    return d_geom


@torch.no_grad()
def sample_foldabs_specificity_features(
    model, features: dict, s_inputs: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
    *, steps: int = 20, samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run frozen Protenix diffusion + confidence head and return d_geom, p_PAE, ipTM.

    Ensemble aggregation is mean categorical geometry, mean PAE probabilities,
    and mean per-sample ipTM. The paper specifies a 32-sample/20-step mini-rollout;
    it does not spell out the ensemble reduction, so the means are explicit
    implementation choices.
    """
    if steps < 1 or samples < 1:
        raise ValueError("diffusion steps/samples must be positive")
    from protenix.model.sample_confidence import calculate_iptm, get_bin_params
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
    coordinates = model.sample_diffusion(
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
    d_geom = d_geom_from_coordinates(coordinates, features, n_tokens=s.shape[-2])

    # The Protenix confidence head consumes the same sampled coordinates and
    # trunk states. Its PAE logits are converted to probabilities as in the
    # official main_inference_loop / sample_confidence utilities.
    _, pae_logits, _, _ = model.run_confidence_head(
        input_feature_dict=features,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_mask=None,
        x_pred_coords=coordinates,
        triangle_multiplicative=model.configs.triangle_multiplicative,
        triangle_attention=model.configs.triangle_attention,
        inplace_safe=False,
        chunk_size=model.configs.infer_setting.chunk_size,
    )
    pae_prob_samples = torch.softmax(pae_logits.float(), dim=-1)
    if pae_prob_samples.shape[-1] != PAE_BINS:
        raise ValueError(f"Expected {PAE_BINS} PAE bins, got {pae_prob_samples.shape[-1]}")
    p_pae = pae_prob_samples.mean(dim=0)

    iptm_samples = calculate_iptm(
        pae_prob_samples,
        has_frame=features["has_frame"],
        asym_id=features["asym_id"],
        **get_bin_params(model.configs.loss.pae),
    )
    iptm = iptm_samples.float().mean()
    if not torch.isfinite(p_pae).all() or not torch.isfinite(iptm):
        raise ValueError("Protenix confidence head returned non-finite PAE/ipTM")
    return d_geom, p_pae, iptm
