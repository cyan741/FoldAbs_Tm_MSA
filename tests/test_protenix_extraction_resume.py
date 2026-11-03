"""Both encoder cache formats survive migration to the shared extraction entry."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import extract_protenix_embeddings as extraction


def cached_example(mini):
    record = {'name': 'sample', 'sequences': [
        {'proteinChain': {'sequence': 'AC', 'count': 1}} for _ in range(4 if mini else 2)]}
    spec = {'model': extraction.MINI_MODEL if mini else 'protenix-v1',
            'cycles': 4, 'seed': 42 if mini else 101, 'use_msa': False,
            'cache_precision': 'fp16', 'inference_precision': 'bf16'}
    if not mini:
        spec.update(checkpoint='/old-host/model.pt', checkpoint_bytes=123,
                    sampling_policy='fixed_per_record', deterministic_algorithms=True)
    n = len(record['sequences']) * 2
    payload = {'sample_name': 'sample', 'input_sequences': ['AC'] * (n // 2),
               'extraction_config': spec,
               's': torch.zeros(n, 384, dtype=torch.float16),
               'z': torch.zeros(n, n, 128, dtype=torch.float16),
               'p_dist': torch.full((n, n, 64), 1 / 64, dtype=torch.float16),
               'token_asym_id': torch.arange(n // 2).repeat_interleave(2),
               'token_residue_index': torch.tensor([1, 2] * (n // 2))}
    if not mini:
        for key in ('token_asym_id', 'token_residue_index'):
            payload[key] = payload[key].half()
    return record, spec, payload


@pytest.mark.parametrize('mini', [False, True])
def test_resume_reuses_cache_without_rewriting(tmp_path, mini):
    record, spec, payload = cached_example(mini)
    path = tmp_path / 'sample.pt'
    torch.save(payload, path)
    original = path.read_bytes(), path.stat().st_mtime_ns
    expected = dict(spec)
    if not mini:
        expected['checkpoint'] = '/new-host/model.pt'
    source = tmp_path / 'input.json'
    source.write_text(json.dumps([record]))
    pending, indices, rows = extraction._prepare_worker_input(
        source, tmp_path, tmp_path, num_shards=1, shard_index=0, limit=0,
        resume=True, run_spec=expected)
    assert json.loads(pending.read_text()) == [] and indices == []
    assert rows[0]['sample_id'] == 'sample'
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original


@pytest.mark.parametrize('mini', [False, True])
@pytest.mark.parametrize('mismatch', ['seed', 'cycles', 'sequence', 'sample', 'model'])
def test_resume_rejects_wrong_cache_identity(tmp_path, mini, mismatch):
    record, spec, payload = cached_example(mini)
    payload = copy.deepcopy(payload)
    if mismatch in ('seed', 'cycles'):
        payload['extraction_config'][mismatch] += 1
    elif mismatch == 'sequence':
        payload['input_sequences'][0] = 'AA'
    elif mismatch == 'sample':
        payload['sample_name'] = 'other'
    else:
        payload['extraction_config']['model'] = 'wrong-model'
    path = tmp_path / 'sample.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError):
        extraction._cached_index_row(path, sample_name='sample', input_index=0,
                                     record=record, run_spec=spec)


def test_base_input_rejects_msa_before_preprocessing(tmp_path):
    record, spec, _ = cached_example(False)
    record['sequences'][0]['proteinChain']['unpairedMsaPath'] = '/old-cache/query.a3m'
    source = tmp_path / 'input.json'
    source.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match='forbidden field'):
        extraction._prepare_worker_input(source, tmp_path, tmp_path, num_shards=1,
                                         shard_index=0, limit=0, resume=False, run_spec=spec)
