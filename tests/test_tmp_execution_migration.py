from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_mini_esm_online_ddp import record_execution_change


def test_old_precision_resume_requires_explicit_migration():
    saved = {'run_config': {'activation_checkpoint': True}, 'step': 19840,
             'epoch': 8, 'cursor': 2480}
    config = {'activation_checkpoint': False, 'feature_handoff': 'native'}
    with pytest.raises(ValueError, match='allow-execution-config-change'):
        record_execution_change(saved, config, False)
    record_execution_change(saved, config, True)
    assert config['execution_changes'] == [{'step': 19840, 'epoch': 8, 'cursor': 2480,
        'changes': {'feature_handoff': {'before': 'fp16', 'after': 'native'},
                    'activation_checkpoint': {'before': True, 'after': False}}}]


def test_unchanged_resume_retains_migration_history_without_new_entry():
    history = [{'step': 19840, 'changes': {'feature_handoff': {'before': 'fp16', 'after': 'native'}}}]
    config = {'activation_checkpoint': False, 'feature_handoff': 'native'}
    saved = {'run_config': {**config, 'execution_changes': history}}
    record_execution_change(saved, config, False)
    assert config['execution_changes'] == history
    assert config['execution_changes'] is not history
