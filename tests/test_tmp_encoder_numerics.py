import pytest
import torch
from foldabs_repro.tmp_prediction.encoder_numerics import deterministic_atom_mean

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_segment_mean_preserves_mapping_and_dtype(dtype):
    x=torch.tensor([[1,3],[3,5],[7,9],[2,4],[4,6],[6,8]],dtype=dtype)
    idx=torch.tensor([0,0,1,2,2,2])
    expected=torch.tensor([[2,4],[7,9],[4,6]],dtype=dtype)
    actual=deterministic_atom_mean(x,idx,3)
    assert actual.dtype==dtype
    assert torch.equal(actual,expected)

@pytest.mark.parametrize('idx',[[0,1,0],[0,2,2],[1,1,2]])
def test_rejects_unsupported_mapping(idx):
    with pytest.raises(ValueError):
        deterministic_atom_mean(torch.ones(3,2),torch.tensor(idx))
