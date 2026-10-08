#!/usr/bin/env python3
"""Peptide mismatch with exact natural TCR pairs, shared exclusion and seed 42.

No file hashes, manifests, reports or auxiliary CSVs are written. Numeric IDs
are not required: identities are exact sequence tuples. The requested NumPy
files contain trusted local dictionaries/sets (load with allow_pickle=True).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['tcra_v','tcrb_v','peptide','hla_alleles_raw','hla_alleles','hla_clip','label','neg_count']
AA = set('ACDEFGHIKLMNPQRSTVWY')
SPLITS = ('train','val','test')


def common_allele_prefix(names):
    # Preserve complete colon-delimited allele fields, not a character prefix
    # ending halfway through a field (e.g. 02:01 vs 02:09 -> 02).
    fields = [name.split(':') for name in sorted(set(names))]
    prefix = []
    for parts in zip(*fields):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    if not prefix:
        raise ValueError(f'No allele field prefix for one clip group: {names}')
    return ':'.join(prefix)


def mapping_contexts(mapping):
    by_clip = defaultdict(set)
    import re
    for name, item in mapping.items():
        if not re.fullmatch(r'HLA-[AB]\*\d{2}(?::\d+)+', name):
            continue
        sequence = item.get('clip sequence','')
        if sequence and set(sequence) <= AA:
            # The cleaned grouping uses protein/two-field names for prefixes.
            by_clip[sequence].add(':'.join(name.split(':')[:2]))
    return {clip:common_allele_prefix(names) for clip,names in by_clip.items()}


def load_input(path, mapping_path):
    path = Path(path)
    if not path.exists() and path.with_suffix('.csv').exists():
        path = path.with_suffix('.csv')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'tcra_v','tcrb_v','peptide','hla_alleles_raw','hla_alleles','hla_clip','label','split','mapping_key'}
    if required - set(frame):
        raise ValueError(f'Missing source fields: {sorted(required - set(frame))}')
    frame['split'] = frame.split.str.strip().str.lower().replace({'validation':'val'})
    if set(frame.split) != set(SPLITS) or not pd.to_numeric(frame.label).eq(1).all():
        raise ValueError('Frozen positive input must contain train/val/test and label=1 only')
    for column in ('tcra_v','tcrb_v','peptide','hla_clip'):
        frame[column] = frame[column].map(lambda value: ''.join(value.split()).upper())
        if any(not sequence or not set(sequence) <= AA for sequence in frame[column]):
            raise ValueError(f'Invalid exact sequence in {column}')
    mapping = json.loads(Path(mapping_path).read_text())
    contexts = mapping_contexts(mapping)
    for row in frame.itertuples(index=False):
        if mapping[row.mapping_key]['clip sequence'] != row.hla_clip:
            raise ValueError(f'JSON clip mismatch for {row.mapping_key}')
        if contexts[row.hla_clip] != row.hla_alleles:
            raise ValueError(f'Cleaned HLA prefix disagrees with JSON clip grouping: {row.hla_alleles}')
    if frame.duplicated(['tcra_v','tcrb_v','peptide','hla_clip']).any():
        raise ValueError('Duplicate positive four-chain inputs; refusing implicit weighting')
    frame['label'] = 1
    frame['neg_count'] = 0
    return frame.reset_index(drop=True)


def build_blacklist(positives):
    result = defaultdict(set)
    for row in positives.itertuples(index=False):
        result[row.tcra_v,row.tcrb_v].add(row.peptide)
    return dict(result)


def build_peptide_pool(positives):
    if positives.groupby('hla_alleles').hla_clip.nunique().gt(1).any():
        raise ValueError('One normalized HLA context must have exactly one clip')
    result = defaultdict(set)
    for row in positives.itertuples(index=False):
        result[row.hla_alleles].add(row.peptide)
    return dict(result)


def load_numpy_dictionary(path):
    data = np.load(path,allow_pickle=True).item()
    if not isinstance(data,dict):
        raise ValueError(f'Expected local dictionary in {path}')
    return data


def prepare_groups(positives, pool, blacklist):
    grouped = defaultdict(list)
    for i,row in enumerate(positives.itertuples(index=False)):
        grouped[row.hla_alleles,row.hla_clip,row.tcra_v,row.tcrb_v].append(i)
    groups = []
    for key,indices in sorted(grouped.items()):
        hla,clip,alpha,beta = key
        legal_count = len(pool[hla] - blacklist.get((alpha,beta),set()))
        groups.append((key,indices,legal_count))
    return groups


def capped_quotas(capacities, weights, budget):
    quotas = [0]*len(capacities)
    remaining = min(budget,sum(capacities))
    while remaining:
        active = [i for i,c in enumerate(capacities) if quotas[i] < c]
        weight_sum = sum(weights[i] for i in active)
        additions = {i:min(capacities[i]-quotas[i],remaining*weights[i]//weight_sum) for i in active}
        if not sum(additions.values()):
            for i in sorted(active,key=lambda i:(-weights[i],i))[:remaining]:
                additions[i] = 1
        for i,amount in additions.items():
            quotas[i] += amount
        remaining -= sum(additions.values())
    return quotas


def sample_epoch(positives, pool, blacklist, *, seed=42, epoch=1, n=2,
                 materialize=True, groups=None):
    """Exactly n*P unique negatives when global capacity permits.

    Start with at most n per original positive. Redistribute deficits only to
    components with remaining legal capacity. Per-source counts remain explicit.
    Different (TCR,HLA-clip) components cannot share a negative four-chain input.
    """
    if epoch < 1 or n < 0 or int(n) != n:
        raise ValueError('epoch>=1 and a nonnegative integer negative ratio required')
    n = int(n)
    positives = positives.reset_index(drop=True)
    groups = prepare_groups(positives,pool,blacklist) if groups is None else groups
    capacities = [item[2] for item in groups]
    weights = [len(item[1]) for item in groups]
    baseline = [min(n*w,c) for w,c in zip(weights,capacities)]
    target = n*len(positives)
    if sum(capacities) < target:
        raise ValueError(f'Unique capacity {sum(capacities)} < required {target}; cannot produce fixed 1:{n}')
    extras = capped_quotas([c-q for c,q in zip(capacities,baseline)],weights,target-sum(baseline))
    quotas = [q+extra for q,extra in zip(baseline,extras)]
    result = positives.copy()
    counts = np.zeros(len(result),dtype=np.int64)
    negatives = []
    rows = list(positives.itertuples(index=False))
    for group_index,((hla,clip,alpha,beta),indices,capacity) in enumerate(groups):
        rng = random.Random(f'{seed}:pep_mismatch:{epoch}:{group_index}')
        order = sorted(indices,key=lambda i:(rows[i].peptide,rows[i].hla_alleles_raw))
        rng.shuffle(order)
        quota = quotas[group_index]
        for j,index in enumerate(order):
            counts[index] = quota//len(order) + int(j < quota%len(order))
        if not materialize or not quota:
            continue
        legal = sorted(pool[hla] - blacklist.get((alpha,beta),set()))
        selected = rng.sample(legal,quota)
        offset = 0
        for index in order:
            source = rows[index]
            for peptide in selected[offset:offset+counts[index]]:
                negatives.append({'tcra_v':alpha,'tcrb_v':beta,'peptide':peptide,
                    'hla_alleles_raw':source.hla_alleles_raw,'hla_alleles':hla,'hla_clip':clip,'label':0,'neg_count':0})
            offset += counts[index]
    result['neg_count'] = counts
    if materialize:
        result = pd.concat([result,pd.DataFrame(negatives,columns=COLUMNS)],ignore_index=True)
        result = result.iloc[random.Random(f'{seed}:order:{epoch}').sample(range(len(result)),len(result))].reset_index(drop=True)
        if len(negatives) != target or result.duplicated(['tcra_v','tcrb_v','peptide','hla_clip']).any():
            raise AssertionError('Sampling produced an incorrect budget or duplicate input')
    zero = sum(len(indices) for _,indices,capacity in groups if capacity==0)
    low = sum(len(indices) for _,indices,capacity in groups if capacity<n)
    summary = {'positives':len(positives),'unique_negative_capacity':sum(capacities),
               'negatives':target if materialize else 0,'planned_negatives':int(counts.sum()),
               'zero_candidate_positives':zero,'fewer_than_n_candidate_positives':low,
               'baseline_shortfall_redistributed':target-sum(baseline),
               'neg_count_histogram':{int(k):int(v) for k,v in pd.Series(counts).value_counts().sort_index().items()}}
    return result[COLUMNS],summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positive-splits',default='/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits/positive_splits')
    parser.add_argument('--hla-map',default='/NAS/luyq/TMP_prediction/data/hitph_l4_v1/raw/Hi-TpH-hla_allele2seq.json')
    parser.add_argument('--output-dir',default='/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--negatives-per-positive',type=int,default=2)
    args=parser.parse_args()
    positive=load_input(args.positive_splits,args.hla_map)
    blacklist=build_blacklist(positive)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    target_files=[output/f'{split}.csv' for split in SPLITS]+[output/f'{split}_pep2candidates.npy' for split in SPLITS]+[output/'blacklilst.npy']
    if any(path.exists() for path in target_files):
        raise FileExistsError('Final outputs already exist; inspect them rather than overwriting')
    summaries={}; selected={}; pools={}
    # Finish all checks before publishing any final CSV.
    for split in SPLITS:
        part=positive[positive.split==split].reset_index(drop=True)
        pools[split]=build_peptide_pool(part)
        selected[split],summaries[split]=sample_epoch(part,pools[split],blacklist,seed=args.seed,n=args.negatives_per_positive,
                                                   materialize=split!='train')
        for row in selected[split].query('label == 0').itertuples(index=False):
            if row.peptide in blacklist[row.tcra_v,row.tcrb_v]:
                raise AssertionError('Generated negative intersects global positive blacklist')
    all_negative=pd.concat([frame.query('label == 0') for frame in selected.values()],ignore_index=True)
    if all_negative.duplicated(['tcra_v','tcrb_v','peptide','hla_clip']).any():
        raise ValueError('Cross-split duplicate negative inputs')
    for split in SPLITS:
        selected[split].to_csv(output/f'{split}.csv',index=False)
        np.save(output/f'{split}_pep2candidates.npy',pools[split],allow_pickle=True)
    # Filename follows the user-requested spelling; no second alias is written.
    np.save(output/'blacklilst.npy',blacklist,allow_pickle=True)
    print(json.dumps({'seed':args.seed,'columns':COLUMNS,'blacklist_tcrs':len(blacklist),
                      'blacklist_pairs':sum(map(len,blacklist.values())),'splits':summaries},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
