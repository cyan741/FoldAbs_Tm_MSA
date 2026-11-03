#!/usr/bin/env python3
"""Audit real training groups without encoder execution or GPU use."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from foldabs_repro.tmp_prediction.group_rank import fixed_group_metrics, sample_groups, shard_groups
from train_tmp_specificity import PeptideMismatchSampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sampler = PeptideMismatchSampling(SimpleNamespace(splits_dir=args.splits_dir,
        negatives_per_positive=2, negative_sampling_seed=42))
    reports = []
    for epoch in (1, 2):
        groups, report = sample_groups(sampler, epoch)
        repeated, report2 = sample_groups(sampler, epoch)
        assert report == report2
        for group, other in zip(groups, repeated, strict=True):
            pd.testing.assert_frame_equal(group, other)
            assert group[['tcra', 'tcrb', 'hla_sequence']].drop_duplicates().shape[0] == 1
            assert group.peptide.is_unique
            anchor = group.iloc[0]
            assert anchor.label == 1 and group.iloc[1:].label.eq(0).all()
            for peptide in group.iloc[1:].peptide:
                assert peptide not in sampler.blacklist.get((anchor.tcra, anchor.tcrb), set())
        shards = [shard_groups(groups, 2, rank)[0] for rank in range(2)]
        assert len(shards[0]) == len(shards[1])
        assert not ({g.group_id.iloc[0] for g in shards[0]} & {g.group_id.iloc[0] for g in shards[1]})
        reports.append({'epoch': epoch, **report, 'ddp_dropped_groups': len(groups) % 2})
    evaluation = {}
    for split in ('val', 'test'):
        frame = sampler.manifest(sampler.frames[split], split)
        frame['hla_context'] = sampler.frames[split].hla_alleles.to_numpy()
        dummy = frame[['sample_id']].assign(logit=0.0)
        groups = fixed_group_metrics(frame, dummy)
        eligible = int(frame.groupby('pmhc_id').label.nunique().eq(2).sum())
        evaluation[split] = {'records': len(frame), 'eligible_pmhc': eligible,
                             'fixed_groups': groups['num_groups'], 'skipped_anchors': groups['skipped_anchors']}
        assert eligible > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'passed', 'epochs': reports, 'evaluation': evaluation}, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'evaluation': evaluation, 'summaries': [
        {k: v for k, v in r.items() if k != 'phla_counts'} for r in reports]}))


if __name__ == '__main__':
    main()
