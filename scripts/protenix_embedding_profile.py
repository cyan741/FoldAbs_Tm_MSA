"""Identity checks and the adapter for Protenix 1.0.0's ESM cache layout."""
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def versions():
    return {name: importlib.metadata.version(name) for name in ('protenix', 'torch', 'fair-esm')}


def atomic_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sequences(record):
    chains = record.get('sequences', [])
    if len(chains) != 4:
        raise ValueError('Require tcra/tcrb/peptide/mhc, exactly four chains')
    result = []
    for wrapper in chains:
        chain = wrapper['proteinChain']
        if chain.get('count', 1) != 1 or chain.get('modifications'):
            raise ValueError('Require unmodified protein chains with count=1')
        sequence = chain['sequence']
        if not sequence or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWYX'):
            raise ValueError('Invalid protein sequence')
        result.append(sequence)
    return result


def reject_msa_templates(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in key.lower() for word in ('msa', 'template')):
                raise ValueError(f'No-MSA input contains forbidden field: {key}')
            reject_msa_templates(item)
    elif isinstance(value, list):
        for item in value:
            reject_msa_templates(item)


def chain_key(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def esm_spec(seed, dtype):
    return {'model': 'esm2_t36_3B_UR50D', 'layer': 36, 'embedding_dim': 2560,
            'special_tokens': 'BOS/EOS removed', 'seed': seed, 'inference_precision': dtype,
            'cache_precision': 'fp32', 'package_versions': versions()}


def load_chain(root, sequence, spec=None):
    root = Path(root)
    key = chain_key(sequence)
    metadata = json.loads((root / 'metadata' / (key + '.json')).read_text())
    if metadata['sequence'] != sequence or (spec is not None and metadata['spec'] != spec):
        raise ValueError('ESM sequence or profile mismatch')
    value = torch.load(root / 'embeddings' / 'chains' / (key + '.pt'), map_location='cpu', weights_only=True)
    if value.shape != (len(sequence), 2560) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Invalid ESM tensor')
    return value


def validate_joint(payload, record, spec):
    seqs = sequences(record)
    if payload.get('sample_name') != record['name'] or payload.get('input_sequences') != seqs:
        raise ValueError('Sample or sequence identity mismatch')
    if payload.get('extraction_config') != spec:
        raise ValueError('Extraction profile mismatch; legacy/unidentified caches cannot resume')
    n = sum(map(len, seqs))
    for key, shape in [('s', (n, 384)), ('z', (n, n, 128)), ('p_dist', (n, n, 64))]:
        value = payload[key]
        if value.shape != shape or value.dtype != DTYPES[spec['cache_precision']] or not torch.isfinite(value).all():
            raise ValueError(f'Invalid {key} shape, dtype or values')
    asym, residue = payload['token_asym_id'], payload['token_residue_index']
    if asym.shape != (n,) or residue.shape != (n,) or asym.dtype != torch.int64 or residue.dtype != torch.int64:
        raise ValueError('Token indices must be int64 vectors')
    unique = torch.unique_consecutive(asym)
    if len(unique) != 4 or len(torch.unique(asym)) != 4:
        raise ValueError('Expected four contiguous chains')
    for chain, sequence in zip(unique, seqs):
        indices = residue[asym.eq(chain)]
        if len(indices) != len(sequence) or not torch.equal(indices, torch.arange(1, len(sequence)+1)):
            raise ValueError('Residue indices/chain lengths differ from input')
    probability = payload['p_dist'].float()
    if probability.min() < 0 or probability.max() > 1 or not torch.allclose(probability.sum(-1), torch.ones((n,n)), atol=0.003, rtol=0):
        raise ValueError('Invalid distogram probabilities')


@contextmanager
def cached_esm_dataloader(configs, root, work_dir, *, validate_tensors=True):
    """Replace only the installed package's eager ESM precompute, then restore it.

    Protenix 1.0.0 overwrites ESM paths in InferenceDataset.__init__. A private
    worker cwd and a symlink provide its expected layout without changing packages.
    The replacement validates every required chain and writes the official CSV.
    """
    from protenix.data.esm.esm_featurizer import ESMFeaturizer
    root, work_dir = Path(root).resolve(), Path(work_dir).resolve()
    spec = json.loads((root / 'run_spec.json').read_text())
    original = ESMFeaturizer.__dict__['precompute_esm_embedding']
    cwd = Path.cwd()
    layout = work_dir / 'esm_embeddings'
    layout.mkdir(exist_ok=True)
    link = layout / configs.esm.model_name
    if not link.exists():
        link.symlink_to(root / 'embeddings', target_is_directory=True)
    elif link.resolve() != (root / 'embeddings').resolve():
        raise ValueError('ESM cache link points to another profile')
    def precompute(inputs, model_name, embedding_dir, sequence_fpath, checkpoint_dir):
        if model_name != 'esm2-3b':
            raise ValueError('Wrong ESM model')
        rows = []
        for sequence in sorted({s for record in inputs for s in sequences(record)}):
            key = chain_key(sequence)
            if validate_tensors:
                load_chain(root, sequence, spec)
            else:
                # The extractor validates the complete tensor again immediately
                # before each sample's trunk inference. Avoid reading all chains
                # twice at startup on a remote filesystem.
                meta = json.loads((root / 'metadata' / (key + '.json')).read_text())
                if meta != {'sequence': sequence, 'spec': spec}:
                    raise ValueError('ESM sequence or profile mismatch')
                if not (root / 'embeddings' / 'chains' / (key + '.pt')).is_file():
                    raise FileNotFoundError(key)
            rows.append({'seq': sequence, 'pdb_entity_id': key, 'seq_label': key, 'part_id': 'chains'})
        with open(sequence_fpath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['seq','pdb_entity_id','seq_label','part_id'])
            writer.writeheader()
            writer.writerows(rows)
    ESMFeaturizer.precompute_esm_embedding = staticmethod(precompute)
    try:
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(cwd)
        ESMFeaturizer.precompute_esm_embedding = original


def stabilize_esm_paths(dataloader, configs):
    """Resolve featurizer paths while still inside its private initialization cwd."""
    featurizer = dataloader.dataset.esm_featurizer
    for key in ('embedding_dir', 'sequence_fpath', 'error_dir'):
        value = getattr(featurizer, key, None)
        if value is not None:
            setattr(featurizer, key, str(Path(value).resolve()))
    configs.esm.embedding_dir = featurizer.embedding_dir
    configs.esm.sequence_fpath = featurizer.sequence_fpath
