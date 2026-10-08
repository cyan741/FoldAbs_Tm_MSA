"""Stream cache validation without retaining feature tensors in memory."""
from pathlib import Path
import torch


def validate_feature(payload, sample_id, sequences=None):
    if payload.get('sample_name') != sample_id:
        raise ValueError('sample_name mismatch')
    required = {'s', 'z', 'p_dist', 'token_asym_id'}
    if not required <= set(payload):
        raise ValueError(f'Missing tensors: {sorted(required - set(payload))}')
    asym = payload['token_asym_id']
    s, z, p = (payload[key] for key in ('s', 'z', 'p_dist'))
    if not all(isinstance(value, torch.Tensor) for value in (asym, s, z, p)):
        raise ValueError('Feature values must be tensors')
    n = asym.numel()
    if asym.ndim != 1 or s.ndim != 2 or s.shape[0] != n or z.ndim != 3 or p.ndim != 3:
        raise ValueError('Invalid tensor dimensions')
    if z.shape[:2] != (n, n) or p.shape[:2] != (n, n) or min(s.shape[-1], z.shape[-1], p.shape[-1]) < 1:
        raise ValueError('Inconsistent feature shapes')
    if not all(bool(torch.isfinite(value).all()) for value in (asym, s, z, p)):
        raise ValueError('Non-finite feature values')
    if not bool(asym.eq(asym.long()).all()) or len(torch.unique(asym)) != 4:
        raise ValueError('Expected four integer chain identities')
    if sequences is not None:
        if payload.get('input_sequences') != list(sequences):
            raise ValueError('Four-chain sequence identity missing or mismatched; re-extract legacy cache')
        counts = [int(asym.eq(chain).sum()) for chain in torch.unique(asym, sorted=True)]
        if counts != list(map(len, sequences)):
            raise ValueError('Token counts do not match four-chain sequences')


def audit_cache(examples, sequences_by_id=None, *, require_extraction_config=False):
    failures, sizes = [], []
    expected_channels, expected_config = None, None
    for example in examples:
        path = Path(example.feature_path)
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            sequences = None if sequences_by_id is None else sequences_by_id[example.sample_id]
            validate_feature(payload, example.sample_id, sequences)
            channels = tuple(payload[key].shape[-1] for key in ('s', 'z', 'p_dist'))
            config = payload.get('extraction_config')
            if require_extraction_config:
                required_config = {'model','package_version','cycles','use_msa','inference_precision','cache_precision'}
                if not isinstance(config, dict) or not required_config <= set(config):
                    raise ValueError('Missing extraction configuration; cache is not protocol-ready')
                if config['model'] != 'protenix-v1' or not config['use_msa']:
                    raise ValueError('Protocol requires frozen Protenix-v1 with role-routed MSA')
            if expected_channels is not None and channels != expected_channels:
                raise ValueError('Inconsistent feature channel dimensions across cache')
            if expected_config is not None and config != expected_config:
                raise ValueError('Inconsistent extraction configuration across cache')
            expected_channels = channels
            if config is not None:
                expected_config = config
            sizes.append(path.stat().st_size)
            del payload
        except Exception as error:
            failures.append({'sample_id': example.sample_id, 'error': str(error)})
    return {'ready': not failures, 'checked': len(examples), 'valid': len(sizes),
            'total_bytes': sum(sizes), 'mean_bytes': sum(sizes) / len(sizes) if sizes else 0,
            'failures': failures}
