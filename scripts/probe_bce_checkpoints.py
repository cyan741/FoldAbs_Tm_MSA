#!/usr/bin/env python3
"""Read-only matched-input comparison of early/late BCE heads."""
import argparse,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import torch
from train_mini_esm_online_ddp import OnlineFeatures,forward,atomic_json
from train_tmp_specificity import PeptideMismatchSampling,set_seed
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from diagnose_bce_predictions import summary,loss


def select(frame,n):
    return pd.concat([frame[frame.label.eq(y)].sample(n=n//3 if y else 2*n//3,random_state=1942+y)
                      for y in (0,1)]).sample(frac=1,random_state=42).reset_index(drop=True)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/'paired_probe';out.mkdir(exist_ok=True)
    if (out/'completion.json').exists():raise FileExistsError('Already completed')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20,0)
    device=torch.device('cuda',0);set_seed(42)
    checkpoints={name:torch.load(args.root/file,map_location='cpu',weights_only=False,mmap=True)
                 for name,file in [('early','best_epoch1.pt'),('late','latest_snapshot.pt')]}
    cfg=checkpoints['late']['run_config'].copy()
    for key in ('output_dir','esm_cache_dir','local_esm_cache','splits_dir'):
        cfg[key]=Path(cfg[key])
    cfg.update(output_dir=out,cpu_workers=1,chunk_rounds=120)
    ns=SimpleNamespace(**cfg)
    sampling=PeptideMismatchSampling(ns)
    epoch=checkpoints['late']['epoch']
    training,_=sampling.epoch(epoch)
    validation=sampling.manifest(sampling.frames['val'],'val')
    frames={'train':select(training,120),'validation':select(validation,240)}
    for split,frame in frames.items():frame.to_csv(out/f'{split}_manifest.csv',index=False)
    metadata={name:{'saved_epoch':c['epoch'],'step':c['step'],'cursor':c['cursor'],
                   'last_completed_epoch':c['history'][-1]['epoch'] if c['history'] else None}
              for name,c in checkpoints.items()}
    atomic_json(metadata,out/'checkpoint_metadata.json')
    provider=OnlineFeatures(ns,0,device)
    heads={}
    for name,c in checkpoints.items():
        head=TMPTrunkSpecificityHead(**c['model_config']).to(device)
        head.load_state_dict(c['model']);head.eval().requires_grad_(False);heads[name]=head
    rows=[];started=time.monotonic()
    for split,frame in frames.items():
        for start in range(0,len(frame),120):
            chunk=frame.iloc[start:start+120]
            for row,batch in zip(chunk.itertuples(index=False),provider.batches(chunk),strict=True):
                features=provider.infer(batch,row.sample_id)
                for name,head in heads.items():
                    z=forward(head,features).item()
                    rows.append({'split':split,'checkpoint':name,'sample_id':row.sample_id,
                                 'pmhc_id':row.pmhc_id,'label':row.label,'logit':z})
                del features
                if len(rows)%40==0:
                    print(json.dumps({'event':'probe','complexes':len(rows)//2,'seconds':time.monotonic()-started}),flush=True)
            pd.DataFrame(rows).to_csv(out/'paired_predictions.csv',index=False)
    result={}
    table=pd.DataFrame(rows)
    for split in frames:
        result[split]={name:summary(table[(table.split==split)&(table.checkpoint==name)]) for name in heads}
        early=table[(table.split==split)&(table.checkpoint=='early')].set_index('sample_id')
        late=table[(table.split==split)&(table.checkpoint=='late')].set_index('sample_id').loc[early.index]
        delta=np.asarray(loss(late.label,late.logit)-loss(early.label,early.logit))
        rng=np.random.default_rng(42)
        means=np.mean(delta[rng.integers(0,len(delta),size=(2000,len(delta)))],axis=1)
        result[split]['paired_bce_change']={'late_minus_early':delta.mean(),
            'sample_bootstrap_95_interval':np.quantile(means,[.025,.975]).tolist()}
    result['checkpoint_metadata']=metadata
    result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    atomic_json(result,out/'diagnosis.json')
    atomic_json({'status':'complete','complexes':len(rows)//2,'seconds':time.monotonic()-started},out/'completion.json')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
