import pytest

torch = pytest.importorskip("torch")

from foldabs_repro.tmp_prediction.interfaces import build_role_masks, interface_pools


def test_interface_masks_and_pooling():
    asym = torch.tensor([1, 1, 2, 2, 3, 4])
    roles = ["tcra", "tcrb", "peptide", "mhc"]
    masks = build_role_masks(asym, roles)
    s = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    z = torch.arange(36, dtype=torch.float32).reshape(6, 6, 1)
    pools = interface_pools(s, z, masks)
    assert torch.equal(pools["s_peptide"], s[4])
    assert torch.equal(pools["z_tcra_peptide"], z[:2, 4].mean(dim=0))
    assert torch.equal(pools["z_tcrb_mhc"], z[2:4, 5].mean(dim=0))
