"""CPU regression tests for FoldAbS-aligned d_geom and PairCondition."""
import torch
import pytest

from foldabs_repro.tmp_prediction.geometry import (
    GEOMETRY_BINS, representative_atom_indices, d_geom_from_coordinates,
)


def toy_features():
    # Two tokens, each with one representative atom and one non-representative atom.
    return {
        "atom_to_token_idx": torch.tensor([0, 0, 1, 1]),
        "pae_rep_atom_mask": torch.tensor([False, True, True, False]),
    }


def test_representative_atom_selection_uses_protenix_mask():
    f = toy_features()
    assert representative_atom_indices(f, 2).tolist() == [1, 2]
    f["pae_rep_atom_mask"] = torch.tensor([False, False, True, False])
    with pytest.raises(ValueError, match="exactly one representative atom"):
        representative_atom_indices(f, 2)


def test_d_geom_is_64_bin_ensemble_histogram():
    f = toy_features()
    # Sample 1 representative-atom separation 5 A; sample 2 separation 25 A.
    coordinates = torch.tensor([
        [[0., 0., 0.], [0., 0., 0.], [3., 4., 0.], [0., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [25., 0., 0.], [0., 0., 0.]],
    ])
    geometry = d_geom_from_coordinates(coordinates, f, 2)
    assert geometry.shape == (2, 2, GEOMETRY_BINS)
    torch.testing.assert_close(geometry.sum(-1), torch.ones(2, 2))
    torch.testing.assert_close(geometry, geometry.transpose(0, 1))
    assert geometry[0, 0, 0] == 1
    assert geometry[0, 1, -1] == 0.5
    assert geometry[0, 1].count_nonzero() == 2


def test_geometry_rejects_bad_atom_count_and_nonfinite_coordinates():
    f = toy_features()
    with pytest.raises(ValueError, match="Atom count"):
        d_geom_from_coordinates(torch.zeros(1, 3, 3), f, 2)
    bad = torch.zeros(1, 4, 3)
    bad[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        d_geom_from_coordinates(bad, f, 2)


def test_pair_condition_concatenates_d_geom_and_pae_before_projection():
    pytest.importorskip("protenix")
    from foldabs_repro.tmp_prediction.model import PairCondition

    condition = PairCondition(64, 12, c_geom=64, c_pae=64)
    z = torch.randn(2, 2, 12)
    d_geom = torch.full((2, 2, 64), 1 / 64)
    p_pae = torch.full((2, 2, 64), 1 / 64)
    with pytest.raises(ValueError, match="requires d_geom and p_pae"):
        condition(z, None, d_geom, None)
    value = condition(z, None, d_geom, p_pae)
    assert value.shape == z.shape
    value.sum().backward()
    assert condition.specificity_project[1].weight.grad is not None
    with pytest.raises(ValueError, match="requires p_dist"):
        PairCondition(64, 12)(z, None)


def test_specificity_head_requires_and_fuses_iptm():
    pytest.importorskip("protenix")
    from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead

    model = TMPTrunkSpecificityHead(
        c_s=16, c_z=12, c_dist=64, pairformer_blocks=0,
        hidden_dim=16, n_heads=4, use_d_geom=True,
    )
    s, z = torch.randn(4, 16), torch.randn(4, 4, 12)
    p_dist = torch.full((4, 4, 64), 1 / 64)
    d_geom = torch.full((4, 4, 64), 1 / 64)
    p_pae = torch.full((4, 4, 64), 1 / 64)
    asym = torch.tensor([0, 1, 2, 3])
    roles = ("tcra", "tcrb", "peptide", "hla")
    with pytest.raises(ValueError, match="requires the Protenix ipTM"):
        model(s, z, p_dist, asym, roles, d_geom=d_geom, p_pae=p_pae)
    logit = model(s, z, p_dist, asym, roles, d_geom=d_geom, p_pae=p_pae,
                  iptm=torch.tensor(0.72))
    assert logit.ndim == 0
    assert model.iptm_embedding[0].out_features == 32
