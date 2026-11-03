#!/usr/bin/env python3
"""Compare training gradients for torch and cuEquivariance on one real feature."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import torch.nn.functional as F
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES
from foldabs_repro.tmp_prediction.triangle_attention import install_cueq_all_valid_mask_workaround
from benchmark_hitph_online import enable_activation_checkpoint

parser=argparse.ArgumentParser()
parser.add_argument('--bf16-fast-input',action='store_true')
parser.add_argument('--output-dir',type=Path,default=Path('/data/luyq/mini_esm_optimization_20261017'))
parser.add_argument('--checkpoint',type=Path)
args=parser.parse_args()
torch.set_num_threads(4);torch.manual_seed(123)
path=Path('/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_routes_bench_20261017/single/L4_0000001.pt')
p=torch.load(path,map_location='cpu',weights_only=True)
x=[p[k].cuda().float() for k in ['s','z','p_dist']]+[p['token_asym_id'].cuda(),REQUIRED_ROLES]
ref=TMPTrunkSpecificityHead(c_s=384,c_z=128,c_dist=64,pairformer_blocks=4,pooling_mode='C',dropout=0).cuda()
if args.checkpoint:
    ref.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model'])
fast=copy.deepcopy(ref)
fast.config['triangle_multiplicative']='cuequivariance';fast.config['triangle_attention']='cuequivariance'
install_cueq_all_valid_mask_workaround()
for head in (ref,fast):enable_activation_checkpoint(head);head.train()
optim=[torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=.01) for h in (ref,fast)]
rows=[]
for step in range(3):
    outputs=[];grads=[]
    for head,opt in zip((ref,fast),optim):
        opt.zero_grad(set_to_none=True)
        head_x=[v.to(torch.bfloat16) for v in x[:3]]+x[3:] if head is fast and args.bf16_fast_input else x
        with torch.autocast('cuda',dtype=torch.bfloat16):y=head(*head_x).float()
        loss=F.binary_cross_entropy_with_logits(y,torch.full_like(y,float(step%2)))
        loss.backward()
        outputs.append((y.item(),loss.item()))
        grads.append({k:v.grad.detach().float().cpu() for k,v in head.named_parameters() if v.grad is not None})
        opt.step()
        del y,loss
    common=set(grads[0])&set(grads[1]);norm=sum(float(grads[0][k].square().sum()) for k in common)
    diff=sum(float((grads[0][k]-grads[1][k]).square().sum()) for k in common)
    finite=all(bool(torch.isfinite(g).all()) for group in grads for g in group.values())
    row={'step':step,'torch_logit':outputs[0][0],'cueq_logit':outputs[1][0],
         'logit_abs_difference':abs(outputs[0][0]-outputs[1][0]),'gradient_relative_l2':(diff/max(norm,1e-30))**.5,
         'all_gradients_finite':finite,'matched_gradient_tensors':len(common),
         'nonzero_cueq_gradient_tensors':sum(bool(torch.count_nonzero(g)) for g in grads[1].values())}
    if not finite or not row['nonzero_cueq_gradient_tensors']:raise ValueError('Invalid gradients')
    rows.append(row);print(json.dumps(row),flush=True)
name='head_bf16_gradient_check.json' if args.bf16_fast_input else 'head_kernel_gradient_check.json'
args.output_dir.mkdir(parents=True,exist_ok=True)
(args.output_dir/name).write_text(json.dumps(rows,indent=2))
