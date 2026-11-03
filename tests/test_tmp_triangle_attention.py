import pytest
import torch

from foldabs_repro.tmp_prediction import model
from foldabs_repro.tmp_prediction.triangle_attention import install_cueq_all_valid_mask_workaround


@pytest.mark.parametrize("kind", ["all_valid", "partial", "none", "float"])
def test_workaround_preserves_nonredundant_masks_and_arguments(monkeypatch, kind):
    import protenix.model.triangular.layers as layers

    received = []
    result = object()

    def original(*args):
        received.append(args)
        return result

    monkeypatch.setattr(layers, "cuequivariance_triangular_attn", original)
    masks = {
        "all_valid": torch.ones(2, 2, dtype=torch.bool),
        "partial": torch.tensor([[True, False], [True, True]]),
        "none": None,
        "float": torch.ones(2, 2),
    }
    mask = masks[kind]
    query, key, value, bias = [object() for _ in range(4)]
    install_cueq_all_valid_mask_workaround()
    wrapped = layers.cuequivariance_triangular_attn
    install_cueq_all_valid_mask_workaround()
    assert layers.cuequivariance_triangular_attn is wrapped
    assert wrapped(query, key, value, bias, mask, 0.5) is result
    actual = received.pop()
    assert actual[:4] == (query, key, value, bias)
    assert actual[4] is (None if kind == "all_valid" else mask)
    assert actual[5] == 0.5


def test_head_installs_workaround_only_for_cueq_attention(monkeypatch):
    installed = []
    monkeypatch.setattr(model, "install_cueq_all_valid_mask_workaround", lambda: installed.append(True))
    for backend in ("torch", "cuequivariance"):
        model.TMPTrunkSpecificityHead(c_s=16, c_z=16, c_dist=4, pairformer_blocks=0,
                                     triangle_attention=backend)
        assert len(installed) == int(backend == "cuequivariance")


@pytest.mark.parametrize("partial", [False, True])
def test_mask_normalization_preserves_attention_outputs_and_gradients(monkeypatch, partial):
    import protenix.model.triangular.layers as layers

    def reference(query, key, value, bias, mask, scale):
        effective_bias = bias if mask is None else bias.masked_fill(~mask, float("-inf"))
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=effective_bias, scale=scale)

    monkeypatch.setattr(layers, "cuequivariance_triangular_attn", reference)
    install_cueq_all_valid_mask_workaround()
    torch.manual_seed(42)
    inputs = [torch.randn(1, 2, 5, 4, requires_grad=True) for _ in range(3)]
    inputs.append(torch.randn(1, 2, 5, 5, requires_grad=True))
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    if partial:
        mask[..., -1] = False
    expected = reference(*inputs, mask, 0.5)
    actual = layers.cuequivariance_triangular_attn(*inputs, mask, 0.5)
    torch.testing.assert_close(actual, expected)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    actual_gradients = torch.autograd.grad(actual.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
