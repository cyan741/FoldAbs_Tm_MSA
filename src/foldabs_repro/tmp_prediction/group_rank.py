"""Deterministic peptide groups and pure, equally weighted group ranking loss."""
from collections import Counter
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def group_rank_loss(positive, negatives, margin=1.0):
    if negatives.numel() == 0:
        raise ValueError("A ranking group needs at least one negative")
    return F.softplus(margin + negatives.float().reshape(-1) - positive.float().reshape(())).mean()


def sample_groups(sampling, epoch, negatives=2, require_full=False):
    if negatives < 1:
        raise ValueError("group_negatives must be positive")
    rng = random.Random(sampling.args.negative_sampling_seed + epoch)
    groups, skipped_zero, skipped_short = [], 0, 0
    for anchor_index, anchor in enumerate(sampling.frames['train'].to_dict('records')):
        candidates = sorted(sampling.pools['train'][anchor['hla_alleles']] -
                            sampling.blacklist.get((anchor['tcra_v'], anchor['tcrb_v']), set()) -
                            {anchor['peptide']})
        if not candidates:
            skipped_zero += 1
            continue
        if require_full and len(candidates) < negatives:
            skipped_short += 1
            continue
        chosen = rng.sample(candidates, min(negatives, len(candidates)))
        rows = [anchor] + [{**anchor, 'peptide': peptide, 'label': 0} for peptide in chosen]
        for row in rows[1:]:
            if (row['tcra_v'], row['tcrb_v'], row['peptide'], row['hla_clip']) in sampling.fixed_negative_inputs:
                raise ValueError('Dynamic training negative overlaps fixed evaluation input')
        frame = sampling.manifest(pd.DataFrame(rows), 'train')
        # A biological complex may occur in several groups. Unique occurrence IDs
        # prevent preprocessing from conflating those independent forwards.
        frame['hla_context'] = anchor['hla_alleles']
        frame['original_id'] = frame.sample_id
        frame['sample_id'] = [f'group_{anchor_index:07d}_{i}_{sid}' for i, sid in enumerate(frame.sample_id)]
        frame['group_id'] = f'group_{anchor_index:07d}'
        groups.append(frame)
    rng.shuffle(groups)
    counts = Counter((row.pmhc_id, int(row.label)) for group in groups for row in group.itertuples())
    report = {'num_groups': len(groups), 'skipped_anchors': skipped_zero + skipped_short,
              'skipped_zero_candidates': skipped_zero, 'skipped_incomplete': skipped_short,
              'actual_negatives': sum(len(group) - 1 for group in groups),
              'negative_count_histogram': dict(Counter(len(group) - 1 for group in groups)),
              'phla_counts': {key: {'positive': counts[key, 1], 'negative': counts[key, 0]}
                              for key in sorted({key for key, _ in counts})}}
    return groups, report


def shard_groups(groups, world_size, rank):
    """Drop a shared tail, never duplicate groups to equalize DDP ranks."""
    usable = len(groups) // world_size * world_size
    return groups[:usable][rank::world_size], len(groups) - usable


def fixed_group_metrics(frame, scores, margin=1.0):
    """Only existing negatives with identical natural TCR pair and HLA context."""
    data = frame.merge(scores[['sample_id', 'logit']], on='sample_id', validate='one_to_one')
    losses, gaps, accuracies = [], [], []
    skipped = 0
    keys = ['tcra', 'tcrb', 'hla_sequence']
    if 'hla_context' in data:
        keys.append('hla_context')
    for _, context in data.groupby(keys, sort=True):
        negatives = context.loc[context.label.eq(0), 'logit'].to_numpy(dtype=np.float64)
        for pos in context.loc[context.label.eq(1), 'logit']:
            if not len(negatives):
                skipped += 1
                continue
            difference = pos - negatives
            losses.append(float(np.logaddexp(0, margin - difference).mean()))
            gaps.append(float(difference.mean()))
            accuracies.append(float((difference > 0).mean()))
    return {'group_loss': float(np.mean(losses)) if losses else float('nan'),
            'mean_score_gap': float(np.mean(gaps)) if gaps else float('nan'),
            'pairwise_accuracy': float(np.mean(accuracies)) if accuracies else float('nan'),
            'num_groups': len(losses), 'skipped_anchors': skipped}


class GroupForward(torch.nn.Module):
    """One DDP forward owns every head graph in a complete group."""
    def __init__(self, head, forward_one):
        super().__init__()
        self.head = head
        self.forward_one = forward_one

    def forward(self, features):
        return torch.stack([self.forward_one(self.head, feature).reshape(()) for feature in features])
