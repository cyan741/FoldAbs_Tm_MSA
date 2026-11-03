#!/usr/bin/env python3
"""Isolated synchronized stage benchmark; never writes training checkpoints."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch
from train_mini_esm_online_ddp import OnlineFeatures, forward, atomic_json
from train_tmp_specificity import PeptideMismatchSampling, set_seed
from foldabs_repro.tmp_prediction.group_rank import sample_groups, group_rank_loss
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activation-checkpoint', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--verify-recomputation', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    set_seed(42)
    cfg = json.loads((args.run / 'run_config.json').read_text())
    for key in ('output_dir', 'esm_cache_dir', 'local_esm_cache', 'splits_dir'):
        cfg[key] = Path(cfg[key])
    cfg.update(output_dir=args.output, cpu_workers=1)
    ns = SimpleNamespace(**cfg)
    sampling = PeptideMismatchSampling(ns)
    groups, _ = sample_groups(sampling, 6, 2, False)
    # Two warmups, ten evenly spaced length quantiles, plus longest group.
    ordered = sorted(groups, key=lambda g: sum(len(g.iloc[0][c]) for c in
                     ('tcra', 'tcrb', 'peptide', 'hla_sequence')))
    selected = groups[:2] + [ordered[int(i)] for i in np.linspace(0, len(ordered)-1, 11)]
    provider = OnlineFeatures(ns, 0, torch.device('cuda', 0))
    saved = torch.load(args.run / 'bs6_lr_0.0001sd_42_ep6.pt', map_location='cpu', weights_only=False)
    head = TMPTrunkSpecificityHead(c_s=384, c_z=128, c_dist=64, pairformer_blocks=4,
        pooling_mode='C', hidden_dim=512, dropout=0.1, triangle_multiplicative='cuequivariance',
        triangle_attention='cuequivariance', activation_checkpoint=args.activation_checkpoint).cuda().train()
    head.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    if args.verify_recomputation:
        # Isolate execution equivalence from stochastic dropout and optimizer updates.
        for module in head.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        frame = selected[-1]  # Longest selected group.
        cached = [provider.infer(batch, row.sample_id) for row, batch in
                  zip(frame.itertuples(index=False), provider.batches(frame), strict=True)]
        reference = None
        report = {}
        for enabled in (True, False):
            head.config['activation_checkpoint'] = enabled
            optimizer.zero_grad(set_to_none=True)
            set_seed(123)
            logits = torch.stack([forward(head, x) for x in cached])
            loss = group_rank_loss(logits[0], logits[1:], 1.0)
            loss.backward()
            gradients = {n: p.grad.detach().cpu().clone() for n, p in head.named_parameters()
                         if p.grad is not None}
            if not all(torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError('Non-finite comparison gradient')
            if reference is None:
                reference = (logits.detach().cpu(), gradients)
            else:
                torch.testing.assert_close(logits.detach().cpu(), reference[0], rtol=.01, atol=1e-4)
                assert gradients.keys() == reference[1].keys()
                failures = []
                for n, g in gradients.items():
                    if not torch.allclose(g, reference[1][n], rtol=.01, atol=1e-4):
                        failures.append(n)
                report.update(gradient_max_abs=max((g-reference[1][n]).abs().max().item()
                                                   for n,g in gradients.items()),
                              compared_parameters=len(gradients), elementwise_tolerance_failures=failures)
                flat = torch.cat([g.reshape(-1) for g in gradients.values()]).double()
                ref = torch.cat([reference[1][n].reshape(-1) for n in gradients]).double()
                relative_l2 = ((flat-ref).norm()/ref.norm().clamp_min(1e-12)).item()
                cosine = torch.nn.functional.cosine_similarity(flat, ref, dim=0).item()
                report.update(gradient_relative_l2=relative_l2, gradient_cosine=cosine,
                              # Near-zero elements make relative elementwise tolerances
                              # unsuitable for BF16 kernels; bound global error/direction.
                              passed=relative_l2 < .01 and cosine > .9999)
            report[str(enabled)] = {'loss':loss.item(), 'logits':logits.detach().cpu().tolist()}
            del logits, loss
        atomic_json(report, args.output/'verification.json')
        if not report['passed']:
            raise ValueError(f"Gradient comparison failed: {report}")
        print(json.dumps(report), flush=True)
        return
    rows = []
    context = {}

    def measure(name, fn, *a, **kw):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        result = fn(*a, **kw)
        torch.cuda.synchronize()
        rows.append({**context, 'stage': name, 'seconds': time.perf_counter()-t,
            'start_MiB': before/2**20, 'end_MiB': torch.cuda.memory_allocated()/2**20,
            'peak_MiB': torch.cuda.max_memory_allocated()/2**20,
            'extra_peak_MiB': (torch.cuda.max_memory_allocated()-before)/2**20,
            'reserved_MiB': torch.cuda.max_memory_reserved()/2**20})
        return result

    model = provider.runner.model
    original_encoder = model.get_pairformer_output
    model.get_pairformer_output = lambda *a, **k: measure('encoder_4cycles', original_encoder, *a, **k)
    import protenix.utils.torch_utils as utils
    original_transfer = utils.to_device
    # Original helper can recurse through the module global; guard recursive calls.
    transferring = False
    def transfer(*a, **kw):
        nonlocal transferring
        if transferring:
            return original_transfer(*a, **kw)
        transferring = True
        try:
            return measure('host_to_device', original_transfer, *a, **kw)
        finally:
            transferring = False
    utils.to_device = transfer
    iterator = iter(provider.batches(pd.concat(selected, ignore_index=True)))
    totals = []
    for gi, group in enumerate(selected):
        optimizer.zero_grad(set_to_none=True)
        logits = []
        tgroup = time.perf_counter()
        for si, row in enumerate(group.itertuples(index=False)):
            context.update(group=gi, sample=si, warmup=gi < 2,
                           tokens=sum(len(getattr(row,c)) for c in ('tcra','tcrb','peptide','hla_sequence')))
            t = time.perf_counter()
            batch = next(iterator)
            rows.append({**context, 'stage':'loader_wait', 'seconds':time.perf_counter()-t})
            torch.cuda.synchronize()
            t = time.perf_counter()
            features = provider.infer(batch, row.sample_id)
            if gi == 0 and si == 0:
                atomic_json({'activation_checkpoint': args.activation_checkpoint,
                             'feature_handoff': 'native',
                             'feature_dtypes': [str(x.dtype) for x in features]}, args.output/'profile_config.json')
            torch.cuda.synchronize()
            rows.append({**context, 'stage':'infer_total', 'seconds':time.perf_counter()-t})
            logits.append(measure('head_forward', forward, head, features))
            del features, batch
        context.update(sample=-1)
        z = torch.stack(logits)
        loss = measure('loss', group_rank_loss, z[0], z[1:], 1.0)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite profiling loss')
        measure('head_backward', loss.backward)
        def update():
            torch.nn.utils.clip_grad_norm_(head.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
        measure('optimizer', update)
        totals.append({'group':gi, 'warmup':gi<2, 'seconds':time.perf_counter()-tgroup,
                       'loss':loss.item()})
        del z, logits, loss
        print(json.dumps(totals[-1]), flush=True)
    summary = {}
    for stage in sorted({r['stage'] for r in rows}):
        r = [r for r in rows if r['stage']==stage and not r['warmup']]
        summary[stage] = {'n':len(r)}
        for key in ('seconds','start_MiB','end_MiB','peak_MiB','extra_peak_MiB','reserved_MiB'):
            v = [x[key] for x in r if key in x]
            if v:
                summary[stage][key] = dict(zip(('mean','p50','p95','max'),
                    map(float, (np.mean(v),np.percentile(v,50),np.percentile(v,95),max(v)))))
    atomic_json({'summary':summary,'rows':rows,'groups':totals,
        'notes':'Single GPU, no DDP communication; two warmup groups excluded; length-stratified sample, not population weighted; training processes suspended externally.'}, args.output/'result.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
