"""Mini-ESM cache boundaries: identity, probabilities and no hidden MSA inputs."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from protenix_embedding_profile import atomic_save, reject_msa_templates, validate_joint


def example():
    record = {'name':'example','sequences':[{'proteinChain':{'sequence':'AC','count':1}} for _ in range(4)]}
    spec = {'cache_precision':'fp16','model':'protenix_mini_esm_v0.5.0','cycles':4,
            'kernel_backend':'cuequivariance/cuequivariance'}
    payload = {'sample_name':'example','input_sequences':['AC']*4,'extraction_config':spec,
               's':torch.zeros(8,384,dtype=torch.float16),'z':torch.zeros(8,8,128,dtype=torch.float16),
               'p_dist':torch.full((8,8,64),1/64,dtype=torch.float16),
               'token_asym_id':torch.arange(4).repeat_interleave(2),
               'token_residue_index':torch.arange(1,3).repeat(4)}
    return record,spec,payload


def test_roundtrip_atomic_cache(tmp_path):
    record,spec,payload = example()
    path = tmp_path / 'sample.pt'
    atomic_save(payload,path)
    validate_joint(torch.load(path,weights_only=True),record,spec)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('change',['model','cycles','kernel_backend','sequence','nan','probability','integer','length','legacy'])
def test_reject_cache_mismatch(change):
    record,spec,payload = example()
    expected = copy.deepcopy(spec)
    if change in ('model','cycles','kernel_backend'):
        expected[change] = 'different'
    elif change == 'sequence':
        record['sequences'][0]['proteinChain']['sequence'] = 'AD'
    elif change == 'nan':
        payload['s'][0,0] = float('nan')
    elif change == 'probability':
        payload['p_dist'][0,0] = 0
    elif change == 'integer':
        payload['token_asym_id'] = payload['token_asym_id'].float()
    elif change == 'length':
        payload['token_residue_index'][0] = 7
    elif change == 'legacy':
        del payload['extraction_config']
    with pytest.raises(ValueError):
        validate_joint(payload,record,expected)


@pytest.mark.parametrize('key',['unpairedMsaPath','pairedMsa','templatePath','use_msa'])
def test_no_msa_rejects_hidden_input(key):
    with pytest.raises(ValueError):
        reject_msa_templates({'sequences':[{'proteinChain':{key:''}}]})


@pytest.mark.parametrize('validate_tensors', [True, False])
def test_official_esm_cache_survives_cwd_restore(tmp_path, validate_tensors):
    import json
    import os
    from types import SimpleNamespace
    from protenix.data.esm.esm_featurizer import ESMFeaturizer
    from protenix_embedding_profile import (cached_esm_dataloader, chain_key,
        esm_spec, stabilize_esm_paths)

    root = tmp_path / 'cache'
    work = tmp_path / 'work'
    (root / 'metadata').mkdir(parents=True)
    (root / 'embeddings' / 'chains').mkdir(parents=True)
    work.mkdir()
    spec = esm_spec(42,'bf16')
    (root / 'run_spec.json').write_text(json.dumps(spec))
    key = chain_key('AC')
    expected = torch.ones(2,2560)
    torch.save(expected,root / 'embeddings' / 'chains' / (key+'.pt'))
    (root / 'metadata' / (key+'.json')).write_text(json.dumps({'sequence':'AC','spec':spec}))
    config = SimpleNamespace(esm=SimpleNamespace(model_name='esm2-3b'))
    original_cwd = Path.cwd()
    record = example()[0]
    with cached_esm_dataloader(config,root,work,validate_tensors=validate_tensors):
        ESMFeaturizer.precompute_esm_embedding([record],'esm2-3b',
            './esm_embeddings/esm2-3b','./esm_embeddings/chains.csv','unused')
        featurizer = ESMFeaturizer('./esm_embeddings/esm2-3b','./esm_embeddings/chains.csv',2560)
        loader = SimpleNamespace(dataset=SimpleNamespace(esm_featurizer=featurizer))
        stabilize_esm_paths(loader,config)
    assert Path.cwd() == original_cwd
    assert Path(featurizer.embedding_dir).is_absolute()
    assert torch.equal(featurizer.load_esm_embedding('AC'),expected)
