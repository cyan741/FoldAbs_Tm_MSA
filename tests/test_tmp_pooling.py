"""Pooling behavior/compatibility checks; no GPU or data cache required."""

import pytest

torch = pytest.importorskip("torch")

from foldabs_repro.tmp_prediction.interfaces import POOL_ORDER, build_role_masks, interface_pools
from foldabs_repro.tmp_prediction.pooling import InterfacePooling, QueryAttentionPool


def inputs():
    torch.manual_seed(7)
    s = torch.randn(8, 6)
    z = torch.randn(8, 8, 4)
    asym = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    masks = build_role_masks(asym, ("tcra", "tcrb", "peptide", "mhc"))
    return s, z, asym, masks


@pytest.mark.parametrize("mode,parameters", [("A", 0), ("B", 20), ("C", 44)])
def test_zero_query_reproduces_original_mean_and_dimension(mode, parameters):
    s, z, _, masks = inputs()
    pooling = InterfacePooling(6, 4, mode)
    result = pooling(s, z, masks)
    expected = interface_pools(s, z, masks)
    assert tuple(result) == POOL_ORDER
    assert sum(p.numel() for p in pooling.parameters()) == parameters
    assert torch.cat(list(result.values())).numel() == 4 * 6 + 5 * 4
    for name in POOL_ORDER:
        torch.testing.assert_close(result[name], expected[name])
    if mode == "A":
        assert not pooling.state_dict()


def test_attention_can_focus_and_is_permutation_invariant():
    pool = QueryAttentionPool(2)
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        pool.query.copy_(torch.tensor([5.0, -5.0]))
    result = pool(values)
    assert result[0] > 0.99
    torch.testing.assert_close(result, pool(values.flip(0)))


def test_attention_masks_direction_and_learns_from_nonuniform_values():
    s, z, _, masks = inputs()
    z.requires_grad_()
    pooling = InterfacePooling(6, 4, "B")
    value = pooling(s, z, masks)["z_tcra_peptide"]
    value[0].backward()
    query = pooling.pair_attention["tcra_peptide"].query
    assert query.grad is not None and query.grad.abs().sum() > 0
    selected = masks["tcra"][:, None] & masks["peptide"][None, :]
    assert z.grad[~selected].abs().sum() == 0
    assert z.grad[selected].abs().sum() > 0


def test_invalid_and_empty_inputs_fail():
    with pytest.raises(ValueError, match="pooling_mode"):
        InterfacePooling(6, 4, "D")
    with pytest.raises(ValueError, match="empty"):
        QueryAttentionPool(4)(torch.empty(0, 4))


def test_old_mean_checkpoint_config_and_common_initialization():
    pytest.importorskip("protenix")
    from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead

    config = dict(c_s=6, c_z=4, c_dist=3, pairformer_blocks=0, hidden_dim=8, dropout=0.0)
    torch.manual_seed(42)
    legacy = TMPTrunkSpecificityHead(**config).eval()
    legacy_config = {k: v for k, v in legacy.config.items() if k != "pooling_mode"}
    restored = TMPTrunkSpecificityHead(**legacy_config).eval()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    s, z, asym, masks = inputs()
    p_dist = torch.softmax(torch.randn(8, 8, 3), dim=-1)
    roles = ("tcra", "tcrb", "peptide", "mhc")
    expected_pools = interface_pools(s, legacy.pair_condition(z, p_dist), masks)
    expected_latent = torch.cat([expected_pools[name] for name in POOL_ORDER])
    expected_logit = legacy.classifier(expected_latent.unsqueeze(0)).squeeze()
    torch.testing.assert_close(restored(s, z, p_dist, asym, roles), expected_logit)
    for mode in ("B", "C"):
        torch.manual_seed(42)
        model = TMPTrunkSpecificityHead(**config, pooling_mode=mode).eval()
        for name, tensor in legacy.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], tensor)
        torch.testing.assert_close(model(s, z, p_dist, asym, roles), expected_logit)
        reloaded = TMPTrunkSpecificityHead(**model.config).eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(reloaded(s, z, p_dist, asym, roles), expected_logit)
