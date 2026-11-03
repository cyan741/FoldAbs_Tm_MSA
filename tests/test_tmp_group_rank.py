import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from foldabs_repro.tmp_prediction.group_rank import (
    GroupForward, fixed_group_metrics, group_rank_loss, sample_groups, shard_groups,
)
from prepare_immrep25_subset_training import prepare
from train_tmp_specificity import PeptideMismatchSampling
from test_immrep25_subset_training import make_sources
import train_mini_esm_group_rank as training


def test_loss_order_shift_fp32_and_both_gradients():
    pos = torch.tensor(0., dtype=torch.bfloat16, requires_grad=True)
    neg = torch.tensor([1., 2.], dtype=torch.bfloat16, requires_grad=True)
    loss = group_rank_loss(pos, neg)
    assert loss.dtype == torch.float32
    assert group_rank_loss(pos + 1, neg) < loss
    assert group_rank_loss(pos, neg - 1) < loss
    torch.testing.assert_close(group_rank_loss(pos + 10, neg + 10), loss)
    loss.backward()
    assert pos.grad < 0 and (neg.grad > 0).all()
    with pytest.raises(ValueError):
        group_rank_loss(pos, torch.empty(0))


def test_group_sampling_audit_and_short_candidates(tmp_path):
    source, known = make_sources(tmp_path)
    prepare(source, 'subset', known, tmp_path / 'prepared')
    sampler = PeptideMismatchSampling(SimpleNamespace(splits_dir=tmp_path / 'prepared',
        negatives_per_positive=2, negative_sampling_seed=42))
    # Covers a zero-candidate anchor and one with K_g < K.
    sampler.blacklist['alpha_train_1', 'beta_train_1'] = set(sampler.pools['train']['HLA-A*02'])
    groups, report = sample_groups(sampler, 1, negatives=3)
    again, report2 = sample_groups(sampler, 1, negatives=3)
    assert report == report2
    assert report['skipped_zero_candidates'] == 1
    assert report['negative_count_histogram'] == {2: 1, 3: 2}
    for group, repeated in zip(groups, again):
        pd.testing.assert_frame_equal(group, repeated)
        assert group[['tcra', 'tcrb', 'hla_sequence']].drop_duplicates().shape[0] == 1
        assert group.label.tolist() == [1] + [0] * (len(group) - 1)
        assert group.peptide.is_unique
        positive = group.iloc[0]
        for peptide in group.iloc[1:].peptide:
            assert peptide in sampler.pools['train']['HLA-A*02']
            assert peptide not in sampler.blacklist[positive.tcra, positive.tcrb]
    full, report = sample_groups(sampler, 1, negatives=3, require_full=True)
    assert len(full) == 2 and report['skipped_incomplete'] == 1
    left, dropped = shard_groups(groups, 2, 0)
    right, _ = shard_groups(groups, 2, 1)
    assert len(left) == len(right) == 1 and dropped == 1
    assert left[0].group_id.iloc[0] != right[0].group_id.iloc[0]


def test_fixed_metrics_are_group_weighted_and_context_matched():
    frame = pd.DataFrame({'sample_id': list('abcdefg'), 'tcra': ['a'] * 3 + ['b'] * 2 + ['c'] * 2,
        'tcrb': ['b'] * 7, 'hla_sequence': ['h'] * 6 + ['other'], 'label': [1, 0, 0, 1, 0, 1, 0]})
    scores = pd.DataFrame({'sample_id': list('abcdefg'), 'logit': [3., 0., 1., 0., 2., 1., -1.]})
    result = fixed_group_metrics(frame, scores)
    expected = (group_rank_loss(torch.tensor(3.), torch.tensor([0., 1.])) +
                group_rank_loss(torch.tensor(0.), torch.tensor([2.]))) / 2
    assert result['group_loss'] == pytest.approx(expected.item())
    assert result['num_groups'] == 2 and result['skipped_anchors'] == 1
    assert result['pairwise_accuracy'] == .5


def forward_one(head, feature):
    from torch.utils.checkpoint import checkpoint
    return checkpoint(head, feature, use_reentrant=False)


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(2)
        head = torch.nn.Linear(2, 1)
        reference = torch.nn.Linear(2, 1)
        reference.load_state_dict(head.state_dict())
        ddp = DistributedDataParallel(GroupForward(head, forward_one))
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        groups = [[torch.tensor([float(i + j), float(j - i)]) for j in range(2 + i % 3)] for i in range(7)]
        local, dropped = shard_groups(groups, 2, rank)
        assert dropped == 1
        for start in (0, 2):
            stop = min(start + 2, len(local))
            optimizer.zero_grad()
            for index in range(start, stop):
                with nullcontext() if index + 1 == stop else ddp.no_sync():
                    logits = ddp(iter(local[index]))
                    (group_rank_loss(logits[0], logits[1:]) / (stop - start)).backward()
            optimizer.step()
            ref_optimizer.zero_grad()
            losses = []
            for group in groups[start * 2:stop * 2]:
                scores = torch.stack([reference(x).reshape(()) for x in group])
                losses.append(group_rank_loss(scores[0], scores[1:]))
            torch.stack(losses).mean().backward()
            ref_optimizer.step()
            for actual, expected in zip(head.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    finally:
        dist.destroy_process_group()


def test_real_two_process_ddp_matches_serial_variable_k_and_partial_window(tmp_path):
    mp.spawn(ddp_worker, args=('file://' + str(tmp_path / 'rendezvous'),), nprocs=2, join=True)


class Provider:
    def batches(self, frame):
        return frame.sample_id.tolist()

    def infer(self, batch, expected_id):
        assert batch == expected_id
        return torch.tensor(float(batch))


def test_evaluation_exports_only_test_scores_and_rejects_undefined_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(training.dist, 'all_gather_object', lambda output, rows: output.__setitem__(0, rows))
    monkeypatch.setattr(training, 'forward', lambda head, features: features)
    args = SimpleNamespace(chunk_rounds=2, output_dir=tmp_path, group_margin=1.)
    frame = pd.DataFrame({'sample_id': ['0', '1', '2'], 'pmhc_id': ['p'] * 3, 'label': [0, 0, 1],
                          'tcra': ['a'] * 3, 'tcrb': ['b'] * 3, 'hla_sequence': ['h'] * 3})
    head = torch.nn.Identity()
    metrics = training.evaluate(head, Provider(), frame, args, 0, 1, 'validation')
    assert metrics['macro_auc_0_1'] == 1
    assert head.training
    assert not (tmp_path / 'validation_predictions.csv').exists()
    training.evaluate(head, Provider(), frame, args, 0, 1, 'test')
    result = pd.read_csv(tmp_path / 'test_predictions.csv')
    assert result.score.tolist() == [0, 1, 2]
    assert 'prediction' not in result
    frame.label = 1
    with pytest.raises(ValueError, match='model selection stopped'):
        training.evaluate(head, Provider(), frame, args, 0, 1, 'validation')


def test_fixed_groups_do_not_cross_hla_contexts_with_same_sequence():
    frame = pd.DataFrame({'sample_id': ['p', 'n'], 'tcra': ['a', 'a'], 'tcrb': ['b', 'b'],
                          'hla_sequence': ['same', 'same'], 'hla_context': ['A', 'B'], 'label': [1, 0]})
    scores = pd.DataFrame({'sample_id': ['p', 'n'], 'logit': [1., 0.]})
    result = fixed_group_metrics(frame, scores)
    assert result['num_groups'] == 0 and result['skipped_anchors'] == 1


def test_cli_group_accumulation_and_bce_defaults(monkeypatch, tmp_path):
    import train_mini_esm_online_ddp as entry
    base = ['train', '--splits-dir', str(tmp_path), '--esm-cache-dir', str(tmp_path / 'source'),
            '--local-esm-cache', str(tmp_path / 'local'), '--output-dir', str(tmp_path / 'out')]
    monkeypatch.setattr(sys, 'argv', base)
    args = entry.parse_args()
    assert args.loss_type == 'bce' and args.gradient_accumulation == 3
    assert args.activation_checkpoint is False
    monkeypatch.setattr(sys, 'argv', base + ['--loss-type', 'group_rank', '--group-accumulation-steps', '2'])
    args = entry.parse_args()
    assert args.gradient_accumulation == 2 and args.activation_checkpoint is False
    monkeypatch.setattr(sys, 'argv', base + ['--loss-type', 'group_rank', '--activation-checkpoint'])
    assert entry.parse_args().activation_checkpoint is True
    monkeypatch.setattr(sys, 'argv', base + ['--group-negatives', '0'])
    with pytest.raises(SystemExit):
        entry.parse_args()


def test_smoke_evaluation_keeps_metric_and_tcr_groups():
    frame = pd.DataFrame({'sample_id': [str(i) for i in range(8)],
        'pmhc_id': ['p', 'p', 'x', 'y', 'z', 'w', 'q', 'r'],
        'tcra': ['a', 'b', 'c', 'c', 'e', 'f', 'g', 'h'], 'tcrb': ['b'] * 8,
        'hla_sequence': ['h'] * 8, 'hla_context': ['A'] * 8,
        'peptide': ['P' * (i + 1) for i in range(8)], 'label': [1, 0, 1, 0, 1, 0, 1, 0]})
    selected = training.smoke_evaluation_frame(frame, 42, 1, 6)
    assert set(selected.sample_id) == {'0', '1', '2', '3', '6', '7'}
    assert selected.sample_id.is_unique
    pd.testing.assert_frame_equal(selected, training.smoke_evaluation_frame(frame, 42, 1, 6))
