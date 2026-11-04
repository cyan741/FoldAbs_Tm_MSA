"""Group-rank mode for the online mini-ESM entrypoint (BCE remains separate)."""
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from foldabs_repro.tmp_prediction.checkpoints import archive_epoch, nominal_batch_size
from foldabs_repro.tmp_prediction.group_rank import (
    GroupForward, fixed_group_metrics, group_rank_loss, sample_groups, shard_groups,
)
from foldabs_repro.tmp_prediction.metrics import binary_metrics, macro_pmhc_metrics
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from train_tmp_specificity import PeptideMismatchSampling, capture_rng, restore_rng, set_seed
from train_mini_esm_online_ddp import OnlineFeatures, atomic_checkpoint, atomic_json, forward, record_execution_change


def evaluation_manifest(sampling, split):
    frame = sampling.manifest(sampling.frames[split], split)
    frame['hla_context'] = sampling.frames[split].hla_alleles.to_numpy()
    return frame


def smoke_evaluation_frame(frame, seed, epoch, limit):
    """Keep existing records, ensuring smoke covers both ranking metric axes."""
    selected = []
    for keys in (['pmhc_id'], ['tcra', 'tcrb', 'hla_sequence', 'hla_context']):
        for _, group in frame.groupby(keys, sort=True):
            if group.label.nunique() == 2:
                selected.extend(group.loc[group.label.eq(label)].index[0] for label in (1, 0))
                break
        else:
            raise ValueError(f'No fixed evaluation group with both labels for {keys}')
    lengths = sum(frame[column].str.len() for column in ('tcra', 'tcrb', 'peptide', 'hla_sequence'))
    selected.extend(lengths.loc[frame.label.eq(label)].idxmax() for label in (1, 0))
    selected = list(dict.fromkeys(selected))
    if limit < len(selected):
        raise ValueError('Group smoke evaluation requires at least six records')
    remaining = frame.drop(selected).sample(n=min(limit - len(selected), len(frame) - len(selected)),
                                            random_state=seed + epoch).index.tolist()
    return frame.loc[selected + remaining].sample(frac=1, random_state=seed + epoch).reset_index(drop=True)


@torch.no_grad()
def evaluate(head, provider, frame, args, rank, world_size, split):
    head.eval()
    local = frame.iloc[rank::world_size].reset_index(drop=True)
    rows = []
    for start in range(0, len(local), args.chunk_rounds):
        chunk = local.iloc[start:start + args.chunk_rounds]
        for row, batch in zip(chunk.itertuples(index=False), provider.batches(chunk), strict=True):
            features = provider.infer(batch, row.sample_id)
            score = forward(head, features).item()
            rows.append({'sample_id': row.sample_id, 'pmhc_id': row.pmhc_id,
                         'label': row.label, 'logit': score, 'score': score})
            del features
        print(json.dumps({'event': 'evaluation_progress', 'rank': rank, 'split': split,
                          'done': len(rows), 'total': len(local)}), flush=True)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, rows)
    result = pd.DataFrame([row for shard in gathered for row in shard])
    if len(result) != len(frame) or result.sample_id.duplicated().any():
        raise ValueError('Evaluation sharding lost or duplicated records')
    if not np.isfinite(result.score).all():
        raise FloatingPointError('Non-finite evaluation scores')
    metrics = {**binary_metrics(result.label, result.logit), **macro_pmhc_metrics(result),
               **fixed_group_metrics(frame, result, args.group_margin)}
    if split.endswith('validation') and not np.isfinite(metrics['macro_auc_0_1']):
        raise ValueError('Validation macro partial AUROC is undefined; model selection stopped')
    if rank == 0:
        if split.endswith('test'):
            result.to_csv(args.output_dir / f'{split}_predictions.csv', index=False)
        atomic_json(metrics, args.output_dir / f'{split}_metrics.json')
    head.train()
    return metrics


def write_curves(history, directory):
    # CSV is useful even on servers without matplotlib.
    pd.DataFrame(history).to_csv(directory / 'learning_curves.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ('train_group_loss', 'valid_group_loss'):
        axes[0].plot([h['epoch'] for h in history], [h[key] for h in history], label=key)
    for key in ('valid_macro_pauc_0.1', 'valid_macro_auroc', 'pairwise_accuracy'):
        axes[1].plot([h['epoch'] for h in history], [h[key] for h in history], label=key)
    for ax in axes:
        ax.legend()
        ax.set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png')
    plt.close(fig)


def train(args):
    rank, local_rank, world_size = (int(os.environ.get(name, default)) for name, default in
                                   (('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1)))
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16-capable GPUs are required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        run_lock = (args.output_dir / 'run.lock').open('a')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and (args.output_dir / 'last.pt').exists():
        raise FileExistsError('Use a fresh output directory or resume')
    set_seed(args.seed)
    sampling = PeptideMismatchSampling(args)
    fingerprint = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(args.splits_dir.iterdir()) if path.suffix in ('.npy', '.csv')}
    fingerprint['esm_spec'] = hashlib.sha256((args.esm_cache_dir / 'run_spec.json').read_bytes()).hexdigest()
    config = {**vars(args), 'world_size': world_size, 'input_fingerprint': fingerprint,
              'checkpoint_policy': 'retain_every_epoch',
              'epoch_checkpoint_template': 'bs{batch_size}_lr_{learning_rate}sd_{seed}_ep{epoch}.pt',
              'selection_metric': 'valid_macro_pauc_0.1', 'selection_mode': 'max',
              'effective_groups_per_update': world_size * args.group_accumulation_steps,
              'max_samples_per_update': world_size * args.group_accumulation_steps * (1 + args.group_negatives),
              'encoder_cycles': 4, 'head_blocks': 4, 'head_pooling': 'C', 'precision': 'bf16',
              'diffusion_feature': 'd_geom' if args.use_d_geom else None,
              'geometry_bins': 64 if args.use_d_geom else None,
              'geometry_max_angstrom': 22.0 if args.use_d_geom else None,
              'pae_bins': 64 if args.use_d_geom else None,
              'iptm_embedding_dim': 32 if args.use_d_geom else None,
              'confidence_executed': bool(args.use_d_geom),
              'joint_feature_disk_cache': False, 'feature_handoff': 'native',
              'ordering': 'seed+epoch groups; drop shared DDP tail',
              'hostname': os.uname().nodename, 'visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}
    provider = OnlineFeatures(args, rank, device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    try:
        head = TMPTrunkSpecificityHead(c_s=384, c_z=128, c_dist=64, pairformer_blocks=4,
            pooling_mode='C', hidden_dim=512, dropout=0.1, triangle_multiplicative='cuequivariance',
            triangle_attention='cuequivariance', activation_checkpoint=args.activation_checkpoint,
            use_d_geom=args.use_d_geom).to(device)
        ddp = DistributedDataParallel(GroupForward(head, forward), device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
        epoch, cursor, step, best, bad_epochs, history = 1, 0, 0, -float('inf'), 0, []
        # loss, groups, mean gaps, mean accuracies, sample count
        running = [0.0] * 5
        resumed_rng = None
        if args.resume:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            record_execution_change(saved, config, args.allow_execution_config_change)
            for key in ('loss_type', 'selection_metric', 'world_size', 'input_fingerprint', 'seed',
                        'negative_sampling_seed', 'group_negatives', 'group_margin', 'group_require_full',
                        'group_accumulation_steps', 'learning_rate', 'weight_decay', 'smoke_samples',
                        'early_stop_patience'):
                if saved['run_config'].get(key) != config[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            if args.resume.parent != args.output_dir:
                raise ValueError('Resume in original output directory to retain best.pt')
            for key, default in (('use_d_geom', False), ('diffusion_steps', 20), ('diffusion_samples', 32)):
                if saved['run_config'].get(key, default) != config[key]:
                    raise ValueError(f'Resume diffusion configuration mismatch: {key}')
            head.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            epoch, cursor, step = saved['epoch'], saved['cursor'], saved['step']
            best, bad_epochs, history = saved['best_selection_metric'], saved['bad_epochs'], saved['history']
            running, resumed_rng = saved['running_by_rank'][rank], saved['rng_by_rank'][rank]
            if args.max_steps and args.max_steps <= step:
                raise ValueError('Resume max-steps must exceed saved step')
        set_seed(args.seed + rank)
        if resumed_rng is not None:
            restore_rng(resumed_rng)
        if rank == 0:
            atomic_json(config, args.output_dir / 'run_config.json')
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        session_groups = session_samples = 0

        def save(kind=None, completed_epoch=None):
            states = [None] * world_size
            dist.all_gather_object(states, {'rng': capture_rng(), 'running': running})
            if rank == 0:
                value = {'model': head.state_dict(), 'model_config': head.config, 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'run_config': config, 'epoch': epoch, 'cursor': cursor,
                         'step': step, 'best_selection_metric': best, 'bad_epochs': bad_epochs, 'history': history,
                         'rng_by_rank': [state['rng'] for state in states],
                         'running_by_rank': [state['running'] for state in states]}
                if completed_epoch is not None:
                    value['checkpoint_stage'] = 'training_complete_validation_pending'
                    archive_epoch(value, args.output_dir, batch_size=nominal_batch_size(args, world_size),
                                  learning_rate=args.learning_rate, seed=args.seed, epoch=completed_epoch)
                atomic_checkpoint(value, args.output_dir / 'last.pt')
                if kind:
                    atomic_checkpoint(value, args.output_dir / f'{kind}.pt')
                    # Retain familiar hyperparameter naming alongside stable resume paths.
                    stem = (f'lr_{args.learning_rate:g}_bs_{world_size * args.group_accumulation_steps}'
                            f'_wd_{args.weight_decay:g}_drop_0.1_group_rank_k_{args.group_negatives}_{kind}.pt')
                    alias = args.output_dir / stem
                    if alias.is_symlink():
                        alias.unlink()
                    alias.symlink_to(f'{kind}.pt')
                atomic_json({'epoch': epoch, 'next_group_round': cursor, 'step': step,
                             'elapsed_seconds': time.monotonic() - started}, args.output_dir / 'progress.json')
            dist.barrier()

        while epoch <= args.epochs and bad_epochs < args.early_stop_patience:
            groups, report = sample_groups(sampling, epoch, args.group_negatives, args.group_require_full)
            if args.smoke_samples:
                # Include longest complexes to exercise peak memory, then restore sampled order.
                count = max(world_size, args.smoke_samples // (1 + args.group_negatives))
                indices = sorted(range(len(groups)), key=lambda i: max(
                    len(r.tcra) + len(r.tcrb) + len(r.peptide) + len(r.hla_sequence)
                    for r in groups[i].itertuples()), reverse=True)[:count]
                groups = [groups[i] for i in sorted(indices)]
            local, dropped = shard_groups(groups, world_size, rank)
            if not local:
                raise ValueError('No complete DDP group round after filtering')
            if rank == 0:
                executed = groups[:len(groups) - dropped] if dropped else groups
                pd.concat(executed, ignore_index=True).to_csv(args.output_dir / f'epoch_{epoch:03d}_groups.csv', index=False)
                atomic_json({**report, 'ddp_dropped_groups': dropped, 'executed_groups': len(executed),
                             'executed_negatives': sum(len(g) - 1 for g in executed), 'smoke': bool(args.smoke_samples)},
                            args.output_dir / f'epoch_{epoch:03d}_sampling.json')
            head.train()
            while cursor < len(local):
                chunk_groups = local[cursor:cursor + args.chunk_rounds]
                iterator = iter(provider.batches(pd.concat(chunk_groups, ignore_index=True)))
                chunk_start = cursor
                for offset, group in enumerate(chunk_groups):
                    index = chunk_start + offset
                    start = index // args.group_accumulation_steps * args.group_accumulation_steps
                    stop = min(start + args.group_accumulation_steps, len(local))
                    sync = index + 1 == stop
                    if index == start:
                        optimizer.zero_grad(set_to_none=True)
                    # Generator keeps encoder inference sequential inside ONE DDP forward.
                    features = (provider.infer(next(iterator), row.sample_id) for row in group.itertuples(index=False))
                    with nullcontext() if sync else ddp.no_sync():
                        logits = ddp(features)
                        loss = group_rank_loss(logits[0], logits[1:], args.group_margin)
                        (loss / (stop - start)).backward()
                    gap = logits[0].detach() - logits[1:].detach()
                    running = [a + b for a, b in zip(running, [loss.item(), 1, gap.mean().item(),
                                                             (gap > 0).float().mean().item(), len(group)])]
                    session_groups += 1
                    session_samples += len(group)
                    del logits, loss, gap, features
                    if sync:
                        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
                        optimizer.step()
                        step += 1
                        cursor = index + 1
                        if step == 1 or step % args.log_every == 0:
                            totals = torch.tensor(running, dtype=torch.float64, device=device)
                            dist.all_reduce(totals)
                            checksum = torch.stack([p.detach().double().sum() for p in head.parameters()]).sum()
                            checks = [torch.zeros_like(checksum) for _ in range(world_size)]
                            dist.all_gather(checks, checksum)
                            if not all(torch.equal(checks[0], c) for c in checks):
                                raise RuntimeError('DDP head parameter checksums differ')
                            seconds = time.monotonic() - started
                            record = {'event': 'train', 'rank': rank, 'epoch': epoch, 'step': step,
                                'group_round': cursor, 'train_group_loss': (totals[0] / totals[1]).item(),
                                'mean_score_gap': (totals[2] / totals[1]).item(),
                                'pairwise_accuracy': (totals[3] / totals[1]).item(), 'num_groups': int(totals[1]),
                                'skipped_anchors': report['skipped_anchors'], 'lr': optimizer.param_groups[0]['lr'],
                                'gradient_norm': norm.item(), 'ddp_checksum_equal': True,
                                'encoder_has_grad': any(p.grad is not None for p in provider.runner.model.parameters()),
                                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                                'seconds_per_group_per_rank': seconds / session_groups,
                                'samples_per_second_per_rank': session_samples / seconds, 'elapsed_seconds': seconds}
                            print(json.dumps(record), flush=True)
                            if rank == 0:
                                atomic_json(record, args.output_dir / 'benchmark.json')
                        if step == 1 or step % args.checkpoint_every == 0 or (args.max_steps and step >= args.max_steps):
                            save(completed_epoch=epoch if cursor == len(local) else None)
                        if args.max_steps and step >= args.max_steps:
                            return
                del iterator
            save(completed_epoch=epoch)
            if args.smoke_samples:
                save('final')
                if args.smoke_evaluation_samples:
                    for split, name in (('val', 'smoke_validation'), ('test', 'smoke_test')):
                        frame = smoke_evaluation_frame(evaluation_manifest(sampling, split), args.seed,
                                            epoch, args.smoke_evaluation_samples)
                        evaluate(head, provider, frame, args, rank, world_size, name)
                if rank == 0:
                    atomic_json({'status': 'smoke_complete', 'step': step}, args.output_dir / 'completion.json')
                return
            metrics = evaluate(head, provider, evaluation_manifest(sampling, 'val'),
                               args, rank, world_size, 'validation')
            totals = torch.tensor(running, dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            history.append({'epoch': epoch, 'train_group_loss': (totals[0] / totals[1]).item(),
                'valid_group_loss': metrics['group_loss'], 'valid_macro_pauc_0.1': metrics['macro_auc_0_1'],
                'valid_macro_auroc': metrics['macro_auroc'], 'mean_score_gap': metrics['mean_score_gap'],
                'pairwise_accuracy': metrics['pairwise_accuracy'], 'num_groups': int(totals[1]),
                'skipped_anchors': report['skipped_anchors'], 'lr': optimizer.param_groups[0]['lr']})
            improved = metrics['macro_auc_0_1'] > best
            best, bad_epochs = (metrics['macro_auc_0_1'], 0) if improved else (best, bad_epochs + 1)
            scheduler.step(metrics['macro_auc_0_1'])
            epoch, cursor, running = epoch + 1, 0, [0.0] * 5
            save('best' if improved else None)
            if rank == 0:
                atomic_json(history, args.output_dir / 'history.json')
                write_curves(history, args.output_dir)
        save('final')
        head.load_state_dict(torch.load(args.output_dir / 'best.pt', map_location='cpu', weights_only=False)['model'])
        evaluate(head, provider, evaluation_manifest(sampling, 'test'), args, rank, world_size, 'test')
        if rank == 0:
            atomic_json({'status': 'complete', 'step': step, 'epochs': epoch - 1,
                         'best_selection_metric': best, 'selection_metric': config['selection_metric']},
                        args.output_dir / 'completion.json')
    finally:
        dist.destroy_process_group()
