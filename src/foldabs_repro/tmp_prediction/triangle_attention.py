"""Avoid the cuEq 0.5.1 masked-backward overread for unpadded complexes."""

from functools import wraps

import torch


def install_cueq_all_valid_mask_workaround():
    """Omit only redundant boolean masks; retain real padding and bias unchanged."""
    import protenix.model.triangular.layers as layers

    original = layers.cuequivariance_triangular_attn
    if getattr(original, "_foldabs_all_valid_mask_workaround", False):
        return

    @wraps(original)
    def attention(query, key, value, bias, mask, scale):
        if mask is not None and mask.dtype == torch.bool and bool(mask.all()):
            mask = None
        return original(query, key, value, bias, mask, scale)

    attention._foldabs_all_valid_mask_workaround = True
    layers.cuequivariance_triangular_attn = attention
