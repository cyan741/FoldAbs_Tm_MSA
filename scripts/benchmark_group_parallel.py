#!/usr/bin/env python3
"""Isolated fixed-work online training benchmark; never updates the real run."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import torch
from train_mini_esm_online_ddp import OnlineFeatures, forward, atomic_json
from train_tmp_specificity import PeptideMismatchSampling, set_seed
from foldabs_repro.tmp_prediction.group_rank import sample_groups, group_rank_loss
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--groups', type=int, default=24)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--activation-checkpoint', action='store_true')
    ap.add_argument('--mode', choices=['serial', 'streams'], default='serial')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--processes', type=int, default=1)
    ap.add_argument('--gate', type=Path)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--dropout-zero', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    set_seed(42)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['run_config'].copy()
    for k in ('output_dir','splits_dir','esm_cache_dir','local_esm_cache'):
        cfg[k] = Path(cfg[k])
    cfg.update(output_dir=args.output, cpu_workers=args.workers)
    ns = SimpleNamespace(**cfg)
    sampling = PeptideMismatchSampling(ns)
    all_groups, _ = sample_groups(sampling, 9, 2, False)
    ordered = sorted(all_groups, key=lambda g: max(sum(len(getattr(r,c)) for c in
        ('tcra','tcrb','peptide','hla_sequence')) for r in g.itertuples()))
    selected = [ordered[int(i)] for i in np.linspace(0,len(ordered)-1,args.groups)]
    ids = [g.iloc[0].group_id for g in selected]
    assert len(ids)==len(set(ids))
    global_hash = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    local = selected[args.rank::args.processes]
    warm = [ordered[len(ordered)//2], ordered[-1]]
    frames = warm + local
    pd.concat(local, ignore_index=True).to_csv(args.output/'manifest.csv',index=False)
    provider = OnlineFeatures(ns, 0, torch.device('cuda',0))
    head = TMPTrunkSpecificityHead(c_s=384,c_z=128,c_dist=64,pairformer_blocks=4,
        pooling_mode='C',hidden_dim=512,dropout=.1,triangle_multiplicative='cuequivariance',
        triangle_attention='cuequivariance',activation_checkpoint=args.activation_checkpoint).cuda().train()
    head.load_state_dict(saved['model'])
    if args.dropout_zero:
        for module in head.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    t = time.perf_counter()
    batches = iter(provider.batches(pd.concat(frames,ignore_index=True)))
    loader_setup_seconds = time.perf_counter()-t
    waits = []
    records = []
    record = torch.profiler.record_function if args.profile else lambda _: nullcontext()
    encoder = provider.runner.model.get_pairformer_output
    def traced_encoder(*a,**kw):
        with record('stage/encoder_4cycles'):
            return encoder(*a,**kw)
    provider.runner.model.get_pairformer_output = traced_encoder

    def infer_group(group):
        out=[]
        for row in group.itertuples(index=False):
            with record('stage/loader_wait'):
                t=time.perf_counter();batch=next(batches);waits.append(time.perf_counter()-t)
            with record('stage/features_with_transfer'):
                out.append(provider.infer(batch,row.sample_id))
        return out

    def train_group(group, features=None):
        optimizer.zero_grad(set_to_none=True)
        logits=[]
        if features is None:
            for row in group.itertuples(index=False):
                with record('stage/loader_wait'):
                    t=time.perf_counter();batch=next(batches);waits.append(time.perf_counter()-t)
                with record('stage/features_with_transfer'):
                    feature=provider.infer(batch,row.sample_id)
                with record('stage/head_forward'):
                    logits.append(forward(head,feature))
                del feature,batch
        else:
            for feature in features:
                for tensor in feature:
                    tensor.record_stream(torch.cuda.current_stream())
                with record('stage/head_forward'):
                    logits.append(forward(head,feature))
        logits=torch.stack(logits)
        loss=group_rank_loss(logits[0],logits[1:],1.)
        with record('stage/head_backward'):
            loss.backward()
        with record('stage/optimizer'):
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
        done=torch.cuda.Event();done.record();done.synchronize()
        return {'loss':loss.item(),'logits':logits.detach().cpu().tolist(),'gradient_norm':norm.item()}

    for group in warm:
        train_group(group)
    # Compare identical initial weights/states after warming kernels/allocator.
    head.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    optimizer.zero_grad(set_to_none=True)
    set_seed(123)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    waits.clear()
    atomic_json({'ready':True,'rank':args.rank,'global_hash':global_hash},args.output/'ready.json')
    if args.gate:
        deadline=time.monotonic()+180
        while not args.gate.exists():
            if time.monotonic()>deadline:raise TimeoutError('benchmark start gate')
            time.sleep(.1)
    def trace_ready(prof):
        prof.export_chrome_trace(str(args.output/'trace.json'))
        (args.output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        atomic_json([{'name':e.key,'cpu_us':e.cpu_time_total,
                     'device_us':getattr(e,'device_time_total',0),'count':e.count}
                    for e in prof.key_averages() if e.key.startswith('stage/')],args.output/'stage_profile.json')
    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],schedule=torch.profiler.schedule(wait=0,warmup=0,active=2,repeat=1),
        on_trace_ready=trace_ready) if args.profile else nullcontext()
    start_epoch=time.time();start=time.perf_counter()
    stream=torch.cuda.Stream() if args.mode=='streams' else None
    with profile as prof:
        current=None
        for i,group in enumerate(local):
            t=time.perf_counter()
            if stream is not None:
                if current is None:
                    with torch.cuda.stream(stream):
                        current=infer_group(group)
                        ready=torch.cuda.Event();ready.record()
                torch.cuda.current_stream().wait_event(ready)
                future=None
                if i+1<len(local):
                    with torch.cuda.stream(stream):
                        future=infer_group(local[i+1])
                        next_ready=torch.cuda.Event();next_ready.record()
                result=train_group(group,current)
                current=future
                if future is not None:ready=next_ready
            else:
                result=train_group(group)
            result.update(group_id=group.iloc[0].group_id,seconds=time.perf_counter()-t)
            records.append(result)
            if prof is not None:prof.step()
            print(json.dumps({'rank':args.rank,'done':i+1,'seconds':result['seconds'],'loss':result['loss']}),flush=True)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    report={'config':vars(args),'global_groups_hash':global_hash,'group_ids':[g.iloc[0].group_id for g in local],
        'groups':len(local),'samples':sum(len(g) for g in local),'seconds':seconds,
        'samples_per_second':sum(len(g) for g in local)/seconds,'start_epoch':start_epoch,'end_epoch':time.time(),
        'group_seconds_p50':float(np.percentile([r['seconds'] for r in records],50)),
        'group_seconds_p95':float(np.percentile([r['seconds'] for r in records],95)),
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'loader_setup_seconds':loader_setup_seconds,'loader_wait_seconds':sum(waits),
        'encoder_has_grad':any(p.grad is not None for p in provider.runner.model.parameters()),
        'records':records,'note':'Multi-process mode measures independent replicas total throughput, not synchronized same-model training.'}
    atomic_json(report,args.output/'result.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','config']}),flush=True)


if __name__=='__main__':main()
