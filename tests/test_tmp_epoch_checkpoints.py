from types import SimpleNamespace
import pytest
import torch
from foldabs_repro.tmp_prediction.checkpoints import archive_epoch, epoch_checkpoint_name, nominal_batch_size


def test_names_use_initial_lr_and_global_nominal_samples():
    bce = SimpleNamespace(loss_type='bce', gradient_accumulation=3)
    group = SimpleNamespace(loss_type='group_rank', group_accumulation_steps=1, group_negatives=2)
    assert nominal_batch_size(bce, 2) == nominal_batch_size(group, 2) == 6
    assert epoch_checkpoint_name(6, 1e-4, 42, 1) == 'bs6_lr_0.0001sd_42_ep1.pt'
    group.group_negatives = 3
    assert nominal_batch_size(group, 2) == 8


def test_epoch_archives_survive_new_epochs_and_validation_resume(tmp_path):
    model = {'weight': torch.tensor([1., 2.])}
    value = {'model': model, 'step': 2480, 'epoch': 1, 'cursor': 2480,
             'optimizer': {'lr': 5e-5}, 'checkpoint_stage': 'training_complete_validation_pending'}
    options = dict(batch_size=6, learning_rate=1e-4, seed=42, epoch=1)
    path = archive_epoch(value, tmp_path, **options)
    original = path.read_bytes()
    assert archive_epoch(value, tmp_path, **options) == path
    assert path.read_bytes() == original
    second = archive_epoch({**value, 'model': {'weight': torch.tensor([3., 4.])}, 'step': 4960},
                           tmp_path, **{**options, 'epoch': 2})
    assert second != path and path.read_bytes() == original
    saved = torch.load(path, weights_only=True)
    assert saved['completed_epoch'] == 1 and saved['cursor'] == 2480
    assert saved['optimizer']['lr'] == 5e-5
    with pytest.raises(FileExistsError):
        archive_epoch({**value, 'model': {'weight': torch.tensor([9., 9.])}}, tmp_path, **options)
    with pytest.raises(FileExistsError):
        archive_epoch({**value, 'step': 2500}, tmp_path, **options)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


def test_legacy_archive_accepts_only_completed_epochs(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from archive_legacy_online_epochs import archive_completed
    source = tmp_path/'last.pt'
    value = {'model': {'w':torch.tensor([1.])}, 'step':2480, 'epoch':2, 'cursor':0,
             'history':[{'epoch':1}], 'run_config':{'world_size':2,'gradient_accumulation':3,'learning_rate':1e-4,'seed':42}}
    torch.save(value,source)
    path = archive_completed(source,tmp_path)
    assert path.name == 'bs6_lr_0.0001sd_42_ep1.pt'
    assert torch.load(path,weights_only=True)['completed_epoch'] == 1
    torch.save({**value,'cursor':300,'step':2580},source)
    assert archive_completed(source,tmp_path) is None
    assert len(list(tmp_path.glob('bs*.pt'))) == 1
