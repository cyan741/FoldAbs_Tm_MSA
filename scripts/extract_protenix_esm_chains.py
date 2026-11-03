#!/usr/bin/env python3
"""Precompute deduplicated ESM2-3B residue tensors in a separate GPU process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from protenix_embedding_profile import (DTYPES, atomic_save, chain_key, esm_spec,
    load_chain, reject_msa_templates, seed_rng, sequences)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-json', required=True, nargs='+')
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', choices=DTYPES, default='bf16')
    args = p.parse_args()
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec = esm_spec(args.seed, args.dtype)
        spec_path = root / 'run_spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('ESM cache profile differs')
        spec_path.write_text(json.dumps(spec, indent=2))
        (root / 'embeddings' / 'chains').mkdir(parents=True, exist_ok=True)
        (root / 'metadata').mkdir(exist_ok=True)
        unique = set()
        for path in args.input_json:
            records = json.loads(Path(path).read_text())
            reject_msa_templates(records)
            unique.update(s for record in records for s in sequences(record))
        pending = []
        for sequence in sorted(unique):
            path = root / 'embeddings' / 'chains' / (chain_key(sequence) + '.pt')
            if path.exists():
                load_chain(root, sequence, spec)
            else:
                pending.append(sequence)
        print(f'Unique chains={len(unique)}, reused={len(unique)-len(pending)}, pending={len(pending)}', flush=True)
        if not pending:
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        from argparse import Namespace
        torch.serialization.add_safe_globals([Namespace])
        from protenix.data.esm.compute_esm import load_esm_model
        for name in ['esm2_t36_3B_UR50D.pt', 'esm2_t36_3B_UR50D-contact-regression.pt']:
            if not (Path(args.checkpoint_dir) / name).is_file():
                raise FileNotFoundError(name)
        seed_rng(args.seed)
        model, alphabet = load_esm_model('esm2-3b', args.checkpoint_dir)
        model.eval().requires_grad_(False)
        converter = alphabet.get_batch_converter()
        started = time.perf_counter()
        for i, sequence in enumerate(pending, 1):
            if len(sequence) > 4094:
                raise ValueError('Refusing to truncate a chain')
            _, _, tokens = converter([('chain', sequence)])
            with torch.no_grad(), torch.autocast('cuda', dtype=DTYPES[args.dtype], enabled=args.dtype != 'fp32'):
                output = model(tokens.cuda(), repr_layers=[36], return_contacts=False)
            tensor = output['representations'][36][0, 1:len(sequence)+1].float().cpu().contiguous()
            if tensor.shape != (len(sequence),2560) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid ESM output')
            key = chain_key(sequence)
            # Publish metadata first; tensor appearance commits this cache entry.
            metadata = root / 'metadata' / (key + '.json')
            fd, temporary = tempfile.mkstemp(dir=metadata.parent)
            with os.fdopen(fd, 'w') as f:
                json.dump({'sequence':sequence, 'spec':spec},f)
            os.replace(temporary, metadata)
            atomic_save(tensor, root / 'embeddings' / 'chains' / (key + '.pt'))
            del output, tensor, tokens
            if i % 20 == 0 or i == len(pending):
                print(f'ESM {i}/{len(pending)} ({time.perf_counter()-started:.1f}s)', flush=True)
        (root / 'stats.json').write_text(json.dumps({'unique_chains':len(unique), 'generated':len(pending),
            'seconds':time.perf_counter()-started, 'gpu_peak_bytes':torch.cuda.max_memory_allocated()},indent=2))

if __name__ == '__main__':
    main()
