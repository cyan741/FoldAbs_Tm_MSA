#!/usr/bin/env python3
"""Bounded CUDA IPC encoder producers feeding ONE trainable head; isolated benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from train_mini_esm_online_ddp import OnlineFeatures, forward, atomic_json
from train_tmp_specificity import PeptideMismatchSampling, set_seed
from foldabs_repro.tmp_prediction.group_rank import sample_groups, group_rank_loss
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead


def setup():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    set_seed(42)


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(tensors):
    return hashlib.sha256(json.dumps([
        (name, tensor_digest(value)) for name, value in tensors
    ], sort_keys=True).encode()).hexdigest()


def producer(rank,cfg,groups,warm,out,ack,gate,finish,directory):
    setup()
    cfg=dict(cfg,output_dir=Path(directory)/f'encoder{rank}',cpu_workers=1)
    provider=OnlineFeatures(SimpleNamespace(**cfg),rank,torch.device('cuda',0))
    if cfg.get('stable_aggregate',False):
        from foldabs_repro.tmp_prediction.encoder_numerics import install_deterministic_atom_mean
        install_deterministic_atom_mean()
    assigned=list(range(rank,len(groups),cfg['pool_size']))
    iterator=iter(provider.batches(pd.concat([warm]+[groups[i] for i in assigned],ignore_index=True)))
    records=[]
    for index in [-1]+assigned:
        if index!=-1 and not gate.wait(300):raise TimeoutError('start gate')
        group=warm if index==-1 else groups[index]
        start=time.perf_counter();features=[]
        for row in group.itertuples(index=False):
            features.append(provider.infer(next(iterator),row.sample_id))
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        # Warmup checksum verifies all tensor values after IPC, outside timed section.
        checksum=[[t.double().sum().item() for t in f] for f in features] if index==-1 else None
        out.put((index,features,checksum))
        if ack.get(timeout=300)!=index:raise RuntimeError('ack index mismatch')
        del features
        if index==-1:
            torch.cuda.reset_peak_memory_stats()
        else:records.append({'index':index,'produce_seconds':elapsed})
    atomic_json({'rank':rank,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'encoder_has_grad':any(p.grad is not None for p in provider.runner.model.parameters()),
        'records':records},Path(directory)/f'encoder{rank}.json')
    # IPC exporting processes must outlive all consumer references.
    if not finish.wait(300):raise TimeoutError('consumer finish')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--encoders',type=int,default=4)
    ap.add_argument('--groups',type=int,default=24)
    ap.add_argument('--activation-checkpoint',action='store_true')
    ap.add_argument('--verify',action='store_true')
    ap.add_argument('--stable-aggregate',action='store_true')
    ap.add_argument('--deterministic-head',action='store_true')
    ap.add_argument('--trace-numerics',action='store_true',help='Hash all features, head states and gradients; timings include diagnostic overhead')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not 1<=args.encoders<=args.groups:raise ValueError('encoder count')
    setup();saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=saved['run_config'].copy()
    for k in ('output_dir','splits_dir','esm_cache_dir','local_esm_cache'):cfg[k]=Path(cfg[k])
    cfg['pool_size']=args.encoders
    cfg['stable_aggregate']=args.stable_aggregate
    all_groups,_=sample_groups(PeptideMismatchSampling(SimpleNamespace(**cfg)),9,2,False)
    ordered=sorted(all_groups,key=lambda g:max(sum(len(getattr(r,c)) for c in ('tcra','tcrb','peptide','hla_sequence')) for r in g.itertuples()))
    groups=[ordered[int(i)] for i in np.linspace(0,len(ordered)-1,args.groups)]
    ids=[g.iloc[0].group_id for g in groups]
    ctx=mp.get_context('spawn');gate=ctx.Event();finish=ctx.Event()
    outputs=[ctx.Queue(maxsize=1) for _ in range(args.encoders)]
    acks=[ctx.Queue(maxsize=1) for _ in range(args.encoders)]
    children=[]
    try:
        for r in range(args.encoders):
            p=ctx.Process(target=producer,args=(r,cfg,groups,ordered[-1],outputs[r],acks[r],gate,finish,str(args.output)))
            p.start();children.append(p)
        if args.deterministic_head:torch.use_deterministic_algorithms(True)
        head=TMPTrunkSpecificityHead(c_s=384,c_z=128,c_dist=64,pairformer_blocks=4,
            pooling_mode='C',hidden_dim=512,dropout=.1,triangle_multiplicative='torch' if args.deterministic_head else 'cuequivariance',
            triangle_attention='torch' if args.deterministic_head else 'cuequivariance',activation_checkpoint=args.activation_checkpoint).cuda().train()
        head.load_state_dict(saved['model'])
        if args.verify:
            for module in head.modules():
                if isinstance(module,torch.nn.Dropout):module.p=0
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
        optimizer.load_state_dict(saved['optimizer'])
        def update(features):
            trace = {}
            if args.trace_numerics:
                trace['features_sha256'] = [[tensor_digest(t) for t in f] for f in features]
                trace['head_before_sha256'] = state_digest(head.state_dict().items())
                trace['optimizer_before_sha256'] = state_digest(
                    (f'{i}/{key}', value) for i, state in optimizer.state_dict()['state'].items()
                    for key, value in sorted(state.items()) if isinstance(value, torch.Tensor))
            optimizer.zero_grad(set_to_none=True)
            logits=torch.stack([forward(head,f) for f in features])
            loss=group_rank_loss(logits[0],logits[1:],1.)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True)
            if args.trace_numerics:
                trace['gradients_sha256'] = state_digest((n, p.grad) for n, p in head.named_parameters() if p.grad is not None)
            optimizer.step();torch.cuda.synchronize()
            if args.trace_numerics:
                trace['head_after_sha256'] = state_digest(head.state_dict().items())
            return {'loss':loss.item(),'gradient_norm':norm.item(),'logits':logits.detach().cpu().tolist(), **trace}
        for r in range(args.encoders):
            index,features,checksum=outputs[r].get(timeout=300)
            assert index==-1
            received=[[t.double().sum().item() for t in f] for f in features]
            assert received==checksum,'IPC checksum differs'
            if r==0:update(features)
            del features
            acks[r].put(-1)
        head.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        del saved
        optimizer.zero_grad(set_to_none=True);set_seed(123)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        records=[];start_epoch=time.time();start=time.perf_counter();gate.set()
        for i,g in enumerate(groups):
            r=i%args.encoders;t=time.perf_counter()
            index,features,_=outputs[r].get(timeout=300)
            wait=time.perf_counter()-t
            assert index==i
            if args.verify and i==0:
                torch.save([[t.cpu() for t in f] for f in features],args.output/'first_features.pt')
            t=time.perf_counter();result=update(features);compute=time.perf_counter()-t
            del features
            acks[r].put(i)
            result.update(index=i,group_id=ids[i],wait_seconds=wait,head_seconds=compute)
            records.append(result)
            print(json.dumps(result),flush=True)
        elapsed=time.perf_counter()-start
        end_epoch=time.time()
        finish.set()
        for p in children:
            p.join(60)
            if p.exitcode!=0:raise RuntimeError(f'producer exit {p.exitcode}')
        reports=[json.loads((args.output/f'encoder{r}.json').read_text()) for r in range(args.encoders)]
        atomic_json({'encoders':args.encoders,'activation_checkpoint':args.activation_checkpoint,'verification_run':args.verify,'stable_aggregate':args.stable_aggregate,'deterministic_head':args.deterministic_head,
            'groups':len(groups),'samples':sum(len(g) for g in groups),'seconds':elapsed,
            'start_epoch':start_epoch,'end_epoch':end_epoch,
            'samples_per_second':sum(len(g) for g in groups)/elapsed,
            'wait_seconds':sum(r['wait_seconds'] for r in records),
            'head_seconds':sum(r['head_seconds'] for r in records),
            'head_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'head_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
            'global_groups_hash':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'ipc_warmup_checksums_equal':True,'trainable_heads':1,'optimizer_updates':len(records),
            'encoder_reports':reports,'records':records},args.output/'result.json')
    finally:
        finish.set()
        for p in children:
            if p.is_alive():p.terminate()
        for p in children:p.join(20)
        for q in outputs+acks:q.close();q.cancel_join_thread()

if __name__=='__main__':main()
