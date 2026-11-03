"""Activation recomputation must preserve the head's training outputs/gradients."""
import copy
import torch
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES


def test_checkpoint_preserves_training_gradients():
    torch.manual_seed(123)
    direct = TMPTrunkSpecificityHead(c_s=16,c_z=16,c_dist=4,pairformer_blocks=1,
        hidden_dim=16,n_heads=4,c_hidden_mul=16,c_hidden_pair_att=4,no_heads_pair=4,
        dropout=0,pooling_mode='C')
    recompute = copy.deepcopy(direct)
    recompute.config['activation_checkpoint'] = True
    x=(torch.randn(8,16),torch.randn(8,8,16),torch.softmax(torch.randn(8,8,4),-1),
       torch.arange(4).repeat_interleave(2),REQUIRED_ROLES)
    outputs=[]
    for model in (direct,recompute):
        model.train()
        out=model(*x)
        out.square().backward()
        outputs.append(out.detach())
    torch.testing.assert_close(outputs[0],outputs[1])
    for (name,p),(other,q) in zip(direct.named_parameters(),recompute.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad)
