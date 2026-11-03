#!/usr/bin/env python3
"""Measure mini-ESM feature write, cached head training and resident GPU training.

Each invocation exposes one GPU. Launch eight invocations concurrently for a
single-host throughput test. Reports retain startup and per-sample measurements.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
import torch.nn.functional as F
from extract_protenix_embeddings import _load_checkpointed_runner, _install_empty_template_fast_path
from benchmark_hitph_online import enable_activation_checkpoint, timed_gpu, summarize
from protenix_embedding_profile import cached_esm_dataloader, stabilize_esm_paths, seed_rng, atomic_save
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import update_input_feature_dict
from protenix.utils.torch_utils import autocasting_disable_decorator, to_device
from runner.batch_inference import preprocess_input
from runner.inference import update_inference_configs
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES


class SeededFeatures(torch.utils.data.Dataset):
    """Keep CPU featurization RNG independent of prefetch worker assignment."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        random.seed(42)
        np.random.seed(42)
        torch.random.default_generator.manual_seed(42)
        return self.dataset[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-json', type=Path, required=True)
    p.add_argument('--esm-cache-dir', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--mode', choices=['online', 'write', 'read'], required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--label', type=float, default=1.)
    p.add_argument('--labels-json', type=Path)
    p.add_argument('--cycles', type=int, default=4)
    p.add_argument('--head-blocks', type=int, default=4)
    p.add_argument('--head-kernel', choices=['torch', 'cuequivariance'], default='torch')
    p.add_argument('--activation-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cpu-workers', type=int, default=0)
    p.add_argument('--warmup-samples', type=int, default=1)
    p.add_argument('--head-input-dtype', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--head-gpu', type=int, default=0,
                   help='Logical GPU for head; 0 keeps features on the encoder GPU')
    args = p.parse_args()
    labels = json.loads(args.labels_json.read_text()) if args.labels_json else {}
    started = time.perf_counter()
    torch.set_num_threads(4)
    seed_rng(42)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.input_json.read_text())[args.offset::args.stride][:args.samples]
    if len(records) != args.samples:
        raise ValueError('Insufficient records for requested disjoint shard')
    report = {'mode': args.mode, 'hostname': os.uname().nodename,
              'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': torch.cuda.get_device_name(), 'samples': [],
              'kernel_backend': 'cuequivariance/cuequivariance',
              'cycles': args.cycles, 'head_blocks': args.head_blocks,
              'head_kernel': args.head_kernel, 'activation_checkpoint': args.activation_checkpoint,
              'cpu_workers': args.cpu_workers, 'warmup_samples': args.warmup_samples,
              'head_input_dtype': args.head_input_dtype,
              'precision': {'encoder_autocast': 'bf16', 'joint_feature': 'fp16',
                            'head_parameters': 'fp32', 'head_autocast': 'bf16'},
              'head_ddp': False, 'feature_transfer': 'same-device tensors' if args.head_gpu == 0 else 'cross-GPU copy',
              'peer_access': torch.cuda.can_device_access_peer(0, args.head_gpu) if args.head_gpu else None,
              'initial_free_mib': torch.cuda.mem_get_info()[0] / 2**20}
    head = optimizer = None
    if args.mode != 'write':
        head = TMPTrunkSpecificityHead(c_s=384, c_z=128, c_dist=64,
            pairformer_blocks=args.head_blocks, pooling_mode='C', hidden_dim=512, dropout=0.1,
            triangle_multiplicative=args.head_kernel, triangle_attention=args.head_kernel).to(f'cuda:{args.head_gpu}')
        if args.activation_checkpoint:
            enable_activation_checkpoint(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
        head.train()
    if args.mode != 'read':
        import cuequivariance_torch
        from configs.configs_model_type import model_configs
        name = 'protenix_mini_esm_v0.5.0'
        model_configs[name]['load_strict'] = True
        source = args.work_dir / 'sample_input.json'
        source.write_text(json.dumps(records))
        prepared = preprocess_input(str(source), out_dir=str(args.work_dir), use_msa=False,
                                    use_template=False, msa_server_mode='protenix')
        runner = _load_checkpointed_runner(model_name=name, seeds=[42], n_cycle=args.cycles,
            n_step=1, n_sample=1, dtype='bf16', use_msa=False, use_template=False,
            use_rna_msa=False, trimul_kernel='cuequivariance', triatt_kernel='cuequivariance',
            enable_cache=True, enable_fusion=False)
        runner.model.eval().requires_grad_(False)
        runner.configs.input_json_path = prepared
        runner.configs.num_workers = 0
        _install_empty_template_fast_path()
        with cached_esm_dataloader(runner.configs, args.esm_cache_dir, args.work_dir):
            loader = get_inference_dataloader(configs=runner.configs)
            stabilize_esm_paths(loader, runner.configs)
        loader = torch.utils.data.DataLoader(SeededFeatures(loader.dataset), batch_size=1,
            collate_fn=loader.collate_fn, num_workers=args.cpu_workers,
            **({'prefetch_factor':2, 'persistent_workers':True} if args.cpu_workers else {}))
        iterator = iter(loader)
    report['startup_seconds'] = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    for index, record in enumerate(records):
        torch.cuda.set_device(0)
        row_start = time.perf_counter()
        row = {'id': record['name'], 'warmup': index < args.warmup_samples,
               'label': labels.get(record['name'], args.label)}
        seed_rng(42)
        if args.mode == 'read':
            read_start = time.perf_counter()
            # Do not call this a cold read: page/SSHFS/server cache is uncontrolled.
            cpu = torch.load(args.features_dir / (record['name'] + '.pt'), map_location='cpu', weights_only=True)
            row['read_seconds'] = time.perf_counter() - read_start
            feature, row['h2d_seconds'] = timed_gpu(lambda: {k: cpu[k].cuda() for k in
                ('s', 'z', 'p_dist', 'token_asym_id')})
            del cpu
        else:
            cpu_start = time.perf_counter()
            data, _, error = next(iterator)[0]
            if error:
                raise RuntimeError(error)
            row['featurize_seconds'] = time.perf_counter() - cpu_start
            data, row['h2d_seconds'] = timed_gpu(lambda: to_device(data, runner.device))
            runner.update_model_configs(update_inference_configs(runner.configs, int(data['N_token'].item())))
            def infer():
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    features = update_input_feature_dict(runner.model.relative_position_encoding.generate_relp(data['input_feature_dict']))
                    _, s, z = runner.model.get_pairformer_output(input_feature_dict=features,
                        N_cycle=args.cycles, inplace_safe=True, chunk_size=runner.model.configs.infer_setting.chunk_size)
                    p_dist = autocasting_disable_decorator(True)(torch.softmax)(runner.model.distogram_head(z), dim=-1)
                return {'s': s.detach().half(), 'z': z.detach().half(), 'p_dist': p_dist.detach().half(),
                        'token_asym_id': features['asym_id'].detach().long(),
                        'token_residue_index': features['residue_index'].detach().long()}
            feature, row['infer_seconds'] = timed_gpu(infer)
            del data
        row['feature_dtypes'] = {k: str(v.dtype) for k,v in feature.items()}
        row['feature_bytes'] = sum(v.numel()*v.element_size() for v in feature.values())
        if args.mode == 'write':
            cpu, row['d2h_seconds'] = timed_gpu(lambda: {k:v.cpu() for k,v in feature.items()})
            args.features_dir.mkdir(parents=True, exist_ok=True)
            path = args.features_dir / (record['name'] + '.pt')
            write_start = time.perf_counter()
            atomic_save(cpu, path)
            # Request flush through the mount; remote media persistence is not claimed.
            with path.open('rb') as f:
                os.fsync(f.fileno())
            row['save_fsync_seconds'] = time.perf_counter() - write_start
            row['file_bytes'] = path.stat().st_size
            del cpu
        else:
            if args.head_gpu:
                transfer_start = time.perf_counter()
                feature = {k:v.to(f'cuda:{args.head_gpu}') for k,v in feature.items()}
                torch.cuda.synchronize(args.head_gpu)
                row['peer_transfer_seconds'] = time.perf_counter() - transfer_start
                torch.cuda.set_device(args.head_gpu)
            optimizer.zero_grad(set_to_none=True)
            def forward():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    dtype = torch.float32 if args.head_input_dtype == 'fp32' else torch.bfloat16
                    return head(feature['s'].to(dtype), feature['z'].to(dtype), feature['p_dist'].to(dtype),
                                feature['token_asym_id'], REQUIRED_ROLES).float()
            logit, row['head_forward_seconds'] = timed_gpu(forward)
            row['route_to_forward_seconds'] = time.perf_counter() - row_start
            if index == 0:
                report['startup_to_first_forward_seconds'] = time.perf_counter() - started
            loss = F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, row['label']))
            def backward():
                loss.backward()
                optimizer.step()
            _, row['backward_step_seconds'] = timed_gpu(backward)
            row['loss'] = loss.item()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            del logit, loss
        row['wall_seconds'] = time.perf_counter() - row_start
        row['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        row['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        report['samples'].append(row)
        if index == len(records)-1 and head is not None:
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            report['gradient_check'] = {
                'all_finite': all(bool(torch.isfinite(g).all()) for g in grads),
                'nonzero_tensors': sum(bool(torch.count_nonzero(g)) for g in grads),
                'gradient_tensors': len(grads)}
            if not report['gradient_check']['all_finite'] or not report['gradient_check']['nonzero_tensors']:
                raise ValueError('Invalid head gradients')
        print(json.dumps(row), flush=True)
        del feature
        (args.work_dir / 'progress.json').write_text(json.dumps(report, indent=2))
    report['run_seconds'] = time.perf_counter() - started
    warm = report['samples'][args.warmup_samples:]
    report['summary'] = {k:summarize([r[k] for r in warm if k in r]) for k in
        ('featurize_seconds','read_seconds','h2d_seconds','infer_seconds','d2h_seconds',
         'save_fsync_seconds','head_forward_seconds','backward_step_seconds',
         'peer_transfer_seconds','route_to_forward_seconds','wall_seconds')}
    (args.work_dir / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
