#!/usr/bin/env python3
"""Audit frozen four-chain caches and write a retry list (one file at a time)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from foldabs_repro.tmp_prediction.cache_audit import audit_cache
from foldabs_repro.tmp_prediction.dataset import TMPFeatureExample
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES
from foldabs_repro.tmp_prediction.canonical import load_tmp_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--records-csv', required=True, help='Canonical four-chain inputs for identity checks')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest.sample_id.duplicated().any():
        raise ValueError('Duplicate manifest sample IDs')
    records = load_tmp_csv(args.records_csv)
    sequences = {r.id: [r.tcra, r.tcrb, r.peptide, r.hla_sequence] for r in records.itertuples(index=False)}
    examples = [TMPFeatureExample(str(r.sample_id), str(r.original_id), str(r.pmhc_id),
                REQUIRED_ROLES, Path(args.features_dir) / f'{r.sample_id}.pt', int(r.label))
                for r in manifest.itertuples(index=False)]
    report = audit_cache(examples, sequences)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'feature_cache_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(report['failures'], columns=['sample_id', 'error']).to_csv(out / 'feature_cache_retry.csv', index=False)
    print(f"Cache audit: {report['valid']}/{report['checked']} valid; mean bytes={report['mean_bytes']:.0f}")
    if not report['ready']:
        raise SystemExit('Cache audit failed; retry list written')


if __name__ == '__main__':
    main()
