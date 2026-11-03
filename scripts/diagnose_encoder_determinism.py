#!/usr/bin/env python3
"""Separate featurization, frozen-weight and inference reproducibility."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from train_mini_esm_online_ddp import OnlineFeatures,atomic_json
from train_tmp_specificity import PeptideMismatchSampling,set_seed
from foldabs_repro.tmp_prediction.group_rank import sample_groups


def fingerprint(x):
    if isinstance(x,torch.Tensor):
        t=x.detach().cpu().contiguous()
        return {'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(x,dict):return {str(k):fingerprint(v) for k,v in sorted(x.items())}
    if isinstance(x,(list,tuple)):return [fingerprint(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    return str(type(x))


def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:out.append(prefix+'/'+k)
            else:out+=differences(a[k],b[k],prefix+'/'+k)
        return out
    return [] if a==b else [prefix]


def stats(a,b):
    result=[]
    for name,x,y in zip(('single','pair','probability','asym'),a,b):
        d=x.float()-y.float()
        result.append({'tensor':name,'equal':torch.equal(x,y),'max_abs':d.abs().max().item(),'rms_diff':d.square().mean().sqrt().item()})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--torch-kernels',action='store_true');p.add_argument('--deterministic',action='store_true');p.add_argument('--stable-aggregate',action='store_true');p.add_argument('--reuse-input',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.cuda.set_device(0);set_seed(42)
    if a.deterministic:torch.use_deterministic_algorithms(True)
    ck='/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2/resume_before_encoder_pool_sweep_20261031.pt'
    saved=torch.load(ck,map_location='cpu',weights_only=False);cfg=saved['run_config'];del saved
    for k in ('output_dir','splits_dir','esm_cache_dir','local_esm_cache'):cfg[k]=Path(cfg[k])
    cfg.update(output_dir=a.output,cpu_workers=1)
    ns=SimpleNamespace(**cfg);groups,_=sample_groups(PeptideMismatchSampling(ns),9,2,False)
    groups.sort(key=lambda g:max(sum(len(getattr(r,c)) for c in ('tcra','tcrb','peptide','hla_sequence')) for r in g.itertuples()))
    frame=groups[0].iloc[:1];sample=frame.iloc[0].sample_id
    provider=OnlineFeatures(ns,0,torch.device('cuda',0));model=provider.runner.model
    if a.torch_kernels:
        model.configs.triangle_attention='torch';model.configs.triangle_multiplicative='torch'
    weights=fingerprint(model.state_dict());atomic_json(weights,a.output/'weights.json')
    batches=[]
    for i in range(2):
        if a.reuse_input:
            batch=torch.load(a.reference/'batch.pt',map_location='cpu',weights_only=False)
        else:
            iterator=iter(provider.batches(frame));batch=next(iterator);del iterator
        batches.append(batch)
        atomic_json(fingerprint(batch[0][0]['input_feature_dict']),a.output/f'input{i}.json')
    torch.save(batches[0],a.output/'batch.pt')
    stages=[]
    import protenix.model.modules.transformer as transformer
    original_aggregate=transformer.aggregate_atom_to_token
    def traced_aggregate(x_atom,atom_to_token_idx,n_token=None,reduce='mean'):
        stages.append({'name':'aggregate_source','output':fingerprint(x_atom)})
        if a.stable_aggregate:
            assert x_atom.ndim==2 and atom_to_token_idx.ndim==1 and reduce=='mean'
            unique,counts=torch.unique_consecutive(atom_to_token_idx,return_counts=True)
            assert torch.equal(unique,torch.arange(len(unique),device=unique.device))
            result=torch.segment_reduce(x_atom.float(),reduce='mean',lengths=counts,axis=0).to(x_atom.dtype)
        else:
            result=original_aggregate(x_atom,atom_to_token_idx,n_token,reduce)
        stages.append({'name':'aggregate_result','output':fingerprint(result)})
        return result
    transformer.aggregate_atom_to_token=traced_aggregate
    def hook(name):
        def record(module,inputs,output):stages.append({'name':name,'output':fingerprint(output)})
        return record
    handles=[getattr(model,n).register_forward_hook(hook(n)) for n in ('input_embedder','linear_no_bias_sinit','pairformer_stack')]
    outputs=[]
    cases=[('same_input_a',batches[0]),('same_input_b',batches[0]),('new_input',batches[1])]
    if a.reference:cases.append(('reference_input',torch.load(a.reference/'batch.pt',map_location='cpu',weights_only=False)))
    for label,batch in cases:
        stages.clear()
        result=[t.detach().cpu() for t in provider.infer(copy.deepcopy(batch),sample)]
        torch.save(result,a.output/f'{label}.pt');atomic_json(stages,a.output/f'{label}_stages.json');outputs.append(result)
    report={'sample':sample,'training':model.training,'encoder_has_grad':any(p.requires_grad for p in model.parameters()),'input_repeat_differences':differences(fingerprint(batches[0][0][0]['input_feature_dict']),fingerprint(batches[1][0][0]['input_feature_dict'])),'same_input_repeat':stats(outputs[0],outputs[1]),'new_input_repeat':stats(outputs[0],outputs[2]),'torch_kernels':a.torch_kernels,'deterministic':a.deterministic,'stable_aggregate':a.stable_aggregate}
    if a.reference:
        ref=torch.load(a.reference/'same_input_a.pt',map_location='cpu',weights_only=True)
        report.update(weights_differences=differences(weights,json.loads((a.reference/'weights.json').read_text())),fresh_process_same_input=stats(ref,outputs[-1]),fresh_process_new_input=stats(ref,outputs[0]),fresh_process_input_differences=differences(fingerprint(batches[0][0][0]['input_feature_dict']),json.loads((a.reference/'input0.json').read_text())))
    atomic_json(report,a.output/'report.json');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
