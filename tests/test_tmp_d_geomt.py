"""CPU-only regression tests for diffusion-derived TMP geometry.

The expensive frozen Protenix rollout is validated on a GPU via the smoke
command in docs/plans/TMP_D_GEOM_DIFFUSION.md.
"""
import torch
import pytest

from foldabs_repro.tmp_prediction.geometry import (
    GEOMETRY_BINS, ca_atom_indices, d_geom_from_coordinates,
)


def toy_features():
    # Two protein tokens, each with N and CA; atom order need not match token order.
    names = ["N", "CA", "CA", "N"]
    token = [0, 0, 1, 1]
    chars = torch.zeros(len(names), 4, 64)
    for i, name in enumerate(names):
        for j, char in enumerate(name.ljust(4)):
            chars[i, j, ord(char) - 32] = 1
    return {"ref_atom_name_chars": chars, "atom_to_token_idx": torch.tensor(token)}


def test_ca_selection_checks_names_and_residue_mapping():
    f = toy_features()
    assert ca_atom_indices(f, 2).tolist() == [1, 2]
    f["ref_atom_name_chars"][2] = f["ref_atom_name_chars"][3]
    with pytest.raises(ValueError, match="exactly one CA"):
        ca_atom_indices(f, 2)


def test_diffusion_geometry_64_bins_probabilities_symmetry_and_overflow():
    f = toy_features()
    # Sample 1 CA separation 5 A; sample 2 separation 25 A (overflow).
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


def test_geometry_rejects_missing_or_nonfinite_coordinates():
    f = toy_features()
    with pytest.raises(ValueError, match="Atom count"):
        d_geom_from_coordinates(torch.zeros(1, 3, 3), f, 2)
    bad = torch.zeros(1, 4, 3)
    bad[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        d_geom_from_coordinates(bad, f, 2)


def test_geometry_pair_condition_does_not_silently_accept_missing_tensor():
    pytest.importorskip("protenix")
    from foldabs_repro.tmp_prediction.model import PairCondition

    condition = PairCondition(64, 12, c_geom=64)
    z = torch.randn(2, 2, 12)
    probabilities = torch.full((2, 2, 64), 1 / 64)
    with pytest.raises(ValueError, match="requires real"):
        condition(z, probabilities)
    value = condition(z, probabilities, probabilities)
    assert value.shape == z.shape
    value.sum().backward()
    assert condition.geom_project[1].weight.grad is not None
    with pytest.raises(ValueError, match="trunk-only"):
        PairCondition(64, 12)(z, probabilities, probabilities)
