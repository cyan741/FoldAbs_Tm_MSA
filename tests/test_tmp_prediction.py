import pandas as pd
import pytest

from foldabs_repro.tmp_prediction.metrics import macro_pmhc_metrics
from scripts.make_protenix_input.data import canonical_hla, load_tmp_csv
from scripts.make_protenix_input.make_tmp_protenix_input import build_protenix_inputs


def test_hla_canonicalization():
    assert canonical_hla("A0201") == "HLA-A*02:01"
    assert canonical_hla("HLA-B*40:01") == "HLA-B*40:01"


def test_load_and_build_tmp_input(tmp_path):
    path = tmp_path / "tmp.csv"
    pd.DataFrame(
        {
            "id": ["pair-1"],
            "tcra": [" CASSL "],
            "tcrb": ["CASSP"],
            "peptide": ["GILGFVFTL"],
            "hla": ["A0201"],
            "hla_sequence": ["GSHSMRYFYTAMSRPGR"],
            "label": [1],
            "split": ["train"],
        }
    ).to_csv(path, index=False)
    frame = load_tmp_csv(path)
    records = build_protenix_inputs(frame)
    assert records[0]["name"] == "pair-1"
    assert len(records[0]["sequences"]) == 4
    assert records[0]["sequences"][3]["proteinChain"]["sequence"] == "GSHSMRYFYTAMSRPGR"


def test_missing_alpha_is_rejected(tmp_path):
    path = tmp_path / "tmp.csv"
    pd.DataFrame(
        {
            "id": ["pair-1"],
            "tcra": [""],
            "tcrb": ["CASSP"],
            "peptide": ["GILGFVFTL"],
            "hla": ["A0201"],
            "hla_sequence": ["GSHSMRYFYTAMSRPGR"],
            "label": [0],
            "split": ["test"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="tcra is empty"):
        load_tmp_csv(path)


def test_macro_pmhc_skips_single_class_groups():
    frame = pd.DataFrame(
        {
            "pmhc_id": ["p1", "p1", "p2", "p2"],
            "label": [0, 1, 1, 1],
            "prediction": [0.1, 0.9, 0.2, 0.3],
        }
    )
    result = macro_pmhc_metrics(frame)
    assert result["eligible_pmhc"] == 1
    assert result["macro_auroc"] == 1.0



@pytest.fixture
def peptide_dataset(tmp_path):
    """Small current-protocol dataset with a shared TCR and a zero-capacity HLA."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    from scripts.negative_sampling import peptide_mismatch

    rows = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    for split, peptides in [('train', ['AC', 'CD', 'DE', 'EF', 'FG', 'GH', 'NP']),
                            ('val', ['CD', 'HI', 'IK', 'KL']), ('test', ['LM', 'MN', 'PQ', 'QR'])]:
        for i, peptide in enumerate(peptides):
            isolated = split == 'train' and i == 6
            alpha = 'AA' if split == 'val' and i == 0 else alphabet[i + {'train': 0, 'val': 8, 'test': 12}[split]] * 2
            raw = 'HLA-B*07:02' if isolated else 'HLA-A*02:01'
            rows.append(dict(record_id=f'pos_{split}_{i}', split=split, tcra_v=alpha, tcrb_v='CC',
                             peptide=peptide, hla_alleles_raw=raw, mapping_key=raw,
                             hla_alleles='HLA-B*07:02' if isolated else 'HLA-A*02',
                             hla_clip='C' if isolated else 'A', label=1))
    source = pd.DataFrame(rows)
    source_path = tmp_path / 'positives.csv'
    source.to_csv(source_path, index=False)
    mapping_path = tmp_path / 'hla.json'
    mapping_path.write_text(json.dumps({'HLA-A*02:01': {'clip sequence': 'A'},
                                       'HLA-A*02:09': {'clip sequence': 'A'},
                                       'HLA-B*07:02': {'clip sequence': 'C'}}))

    def build(name, seed=42):
        output = tmp_path / name
        subprocess.run([sys.executable, str(Path(peptide_mismatch.__file__)),
                        '--positive-splits', str(source_path), '--hla-map', str(mapping_path),
                        '--output-dir', str(output), '--seed', str(seed)], check=True, capture_output=True)
        return output

    return source, build, peptide_mismatch


def test_peptide_mismatch_cli_seed_replay_and_exact_outputs(peptide_dataset):
    source, build, sampling = peptide_dataset
    first, replay, changed = build('first'), build('replay'), build('changed', seed=43)
    expected = {'train.csv', 'val.csv', 'test.csv', 'blacklilst.npy',
                'train_pep2candidates.npy', 'val_pep2candidates.npy', 'test_pep2candidates.npy'}
    assert {path.name for path in first.iterdir()} == expected
    blacklist = sampling.load_numpy_dictionary(first / 'blacklilst.npy')
    assert blacklist == sampling.build_blacklist(source)
    assert blacklist['AA', 'CC'] == {'AC', 'CD'}
    all_negatives = []
    for split in sampling.SPLITS:
        frame = pd.read_csv(first / f'{split}.csv')
        pd.testing.assert_frame_equal(frame, pd.read_csv(replay / f'{split}.csv'))
        assert list(frame.columns) == sampling.COLUMNS
        positives = frame[frame.label.eq(1)]
        assert len(positives) == len(source[source.split.eq(split)])
        assert positives.neg_count.sum() == 2 * len(positives)
        assert sampling.load_numpy_dictionary(first / f'{split}_pep2candidates.npy') == sampling.build_peptide_pool(positives)
        negatives = frame[frame.label.eq(0)]
        assert len(negatives) == (0 if split == 'train' else 2 * len(positives))
        pool = sampling.load_numpy_dictionary(first / f'{split}_pep2candidates.npy')
        for row in negatives.itertuples(index=False):
            assert row.peptide in pool[row.hla_alleles]
            assert row.peptide not in blacklist[row.tcra_v, row.tcrb_v]
        all_negatives.append(negatives)
    assert not pd.concat(all_negatives).duplicated(['tcra_v', 'tcrb_v', 'peptide', 'hla_clip']).any()
    assert not pd.read_csv(first / 'val.csv').equals(pd.read_csv(changed / 'val.csv'))
    assert pd.read_csv(first / 'train.csv').query("hla_clip == 'C'").neg_count.tolist() == [0]


def test_peptide_dynamic_training_exact_resume(peptide_dataset, tmp_path, monkeypatch):
    """Exercise current dynamic mode, joint cache identity and epoch-boundary resume."""
    import json
    import sys
    torch = pytest.importorskip('torch')
    pytest.importorskip('protenix')
    from types import SimpleNamespace
    from scripts import train_tmp_specificity as training

    torch.set_num_threads(1)
    source, build, _ = peptide_dataset
    splits = build('splits')
    source.to_csv(splits / 'positive_splits.csv', index=False)
    sampler = training.PeptideMismatchSampling(SimpleNamespace(
        splits_dir=str(splits), negatives_per_positive=2, negative_sampling_seed=42))
    epoch1, report = sampler.epoch(1)
    epoch2, _ = sampler.epoch(2)
    replay, _ = sampler.epoch(1)
    pd.testing.assert_frame_equal(epoch1, replay)
    assert report['negatives'] == 14 and report['zero_candidate_positives'] == 1
    assert set(epoch1.query('label == 0').sample_id) != set(epoch2.query('label == 0').sample_id)
    assert not epoch1.sample_id.duplicated().any()
    local0, padding = training.shard_epoch_ids(epoch1.sample_id.tolist(), 2, 0)
    local1, _ = training.shard_epoch_ids(epoch1.sample_id.tolist(), 2, 1)
    assert len(local0) == len(local1) == 11 and padding == 1
    assert set(local0 + local1) == set(epoch1.sample_id)
    features = tmp_path / 'features'
    features.mkdir()
    for split, frame in sampler.frames.items():
        if split == 'train':
            negatives = []
            for (hla, clip, alpha, beta), indices, _ in sampler.groups[split]:
                for peptide in sampler.pools[split][hla] - sampler.blacklist[alpha, beta]:
                    negatives.append(dict(tcra_v=alpha, tcrb_v=beta, peptide=peptide,
                                          hla_alleles_raw=frame.iloc[indices[0]].hla_alleles_raw,
                                          hla_alleles=hla, hla_clip=clip, label=0, neg_count=0))
            frame = pd.concat([frame, pd.DataFrame(negatives)], ignore_index=True)
        for row in sampler.manifest(frame, split).itertuples(index=False):
            sequences = [row.tcra, row.tcrb, row.peptide, row.hla_sequence]
            tokens = sum(map(len, sequences))
            torch.save({'sample_name': row.sample_id, 'input_sequences': sequences,
                        'token_asym_id': torch.tensor([i + 1 for i, seq in enumerate(sequences) for _ in seq]),
                        's': torch.randn(tokens, 6), 'z': torch.randn(tokens, tokens, 4),
                        'p_dist': torch.softmax(torch.randn(tokens, tokens, 3), dim=-1)}, features / f'{row.sample_id}.pt')

    def run(name, epochs, resume=None):
        output = tmp_path / name
        argv = ['train', '--negative-sampling', 'peptide_mismatch', '--splits-dir', str(splits),
                '--features-dir', str(features), '--output-dir', str(output), '--epochs', str(epochs),
                '--gradient-accumulation', '4', '--pairformer-blocks', '0', '--hidden-dim', '8',
                '--dropout', '0.2', '--log-every', '100']
        if resume:
            argv += ['--resume', str(resume)]
        monkeypatch.setattr(sys, 'argv', argv)
        training.set_seed(42)
        training.run_training(0, 1, training.parse_args(), [], torch.device('cpu'))
        metrics = json.loads((output / 'metrics.json').read_text())
        checkpoint = output / metrics['final_checkpoint']
        return torch.load(checkpoint, weights_only=True), checkpoint

    full, _ = run('full', 3)
    _, checkpoint = run('partial', 1)
    resumed, _ = run('resumed', 3, checkpoint)
    assert full['history'] == resumed['history'] and full['step'] == resumed['step']
    assert all(torch.equal(full['model'][key], resumed['model'][key]) for key in full['model'])
    assert len(pd.read_csv(tmp_path / 'full' / 'predictions.csv')) == 12
