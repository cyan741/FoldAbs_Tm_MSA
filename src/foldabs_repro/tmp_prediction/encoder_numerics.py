"""Opt-in deterministic aggregation for the frozen, single-complex encoder probe."""
import torch


def deterministic_atom_mean(x_atom, atom_to_token_idx, n_token=None, reduce='mean'):
    """Mean contiguous atom segments in FP32, then restore the feature dtype.

    This intentionally changes accumulation precision from BF16 atomic scatter.
    It supports the unbatched, contiguous token mapping used by our encoder.
    Unsupported layouts fail explicitly instead of silently using atomic scatter.
    """
    if x_atom.ndim != 2 or atom_to_token_idx.ndim != 1 or reduce != 'mean':
        raise ValueError('Expected unbatched atom features and mean reduction')
    if len(atom_to_token_idx) != len(x_atom) or not len(x_atom):
        raise ValueError('Invalid atom-to-token mapping length')
    unique, counts = torch.unique_consecutive(atom_to_token_idx, return_counts=True)
    expected = torch.arange(len(unique), device=unique.device, dtype=unique.dtype)
    if not torch.equal(unique, expected):
        raise ValueError('Token indices must be contiguous and sorted, starting at zero')
    if n_token is not None and n_token != len(unique):
        raise ValueError('Explicit token count differs from atom mapping')
    return torch.segment_reduce(x_atom.float(), reduce='mean', lengths=counts, axis=0).to(x_atom.dtype)


def install_deterministic_atom_mean():
    """Install only in an explicitly opted-in encoder worker process."""
    import protenix.model.modules.transformer as transformer
    transformer.aggregate_atom_to_token = deterministic_atom_mean
