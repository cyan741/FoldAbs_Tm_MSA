#!/usr/bin/env python3
"""Repeat backward with identical features/weights; no optimizer updates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from train_mini_esm_online_ddp import forward,atomic_json
from train_tmp_specificity import set_seed
from foldabs_repro.tmp_prediction.group_rank import group_rank_loss
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--torch-kernels',action='store_true');ap.add_argument('--deterministic',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(0);set_seed(123)
    if a.deterministic:torch.use_deterministic_algorithms(True)
    checkpoint='/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2/resume_before_parallel_eval_step21400.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    head=TMPTrunkSpecificityHead(c_s=384,c_z=128,c_dist=64,pairformer_blocks=4,pooling_mode='C',hidden_dim=512,dropout=0.,triangle_multiplicative='torch' if a.torch_kernels else 'cuequivariance',triangle_attention='torch' if a.torch_kernels else 'cuequivariance',activation_checkpoint=True).cuda().train()
    head.load_state_dict(saved['model']);del saved
    features=[[t.cuda() for t in f] for f in torch.load('/NAS/luyq/foldabs_parallel_eval_20261031/encoder_pool_stable_v1/enc1_verify/first_features.pt',map_location='cpu',weights_only=True)]
    reports=[];reference=None
    for i in range(3):
        head.zero_grad(set_to_none=True);set_seed(123)
        logits=torch.stack([forward(head,f) for f in features]);loss=group_rank_loss(logits[0],logits[1:],1.);loss.backward();torch.cuda.synchronize()
        gradients={n:p.grad.detach().cpu().clone() for n,p in head.named_parameters() if p.grad is not None}
        norm=sum(g.double().square().sum().item() for g in gradients.values())**.5
        row={'iteration':i,'logits':logits.detach().cpu().tolist(),'loss':loss.item(),'gradient_norm':norm}
        if reference is None:reference=gradients
        else:
            differences=[];squared=0.;dot=0.;refnorm=0.
            for n,g in gradients.items():
                r=reference[n];d=(g-r).double();err=d.square().sum().item();squared+=err;dot+=(g.double()*r.double()).sum().item();refnorm+=r.double().square().sum().item()
                if err:differences.append({'name':n,'max_abs':d.abs().max().item(),'squared_diff':err})
            row.update(relative_gradient_L2=(squared/refnorm)**.5,gradient_cosine=dot/(norm*refnorm**.5),different_parameters=len(differences),top_differences=sorted(differences,key=lambda d:d['squared_diff'],reverse=True)[:10])
        reports.append(row)
    atomic_json(reports,a.output/'report.json');print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
