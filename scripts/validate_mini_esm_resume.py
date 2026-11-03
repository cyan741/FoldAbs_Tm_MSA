#!/usr/bin/env python3
"""Verify a complete mini-ESM subset resumes without changing feature files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='/NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings/protenix_mini_esm_v0.5.0')
    p.add_argument('--gpu-ids', default='7,1')
    args = p.parse_args()
    profile = Path(args.profile).resolve()
    if json.loads((profile / 'completion_report.json').read_text())['status'] != 'complete':
        raise ValueError('Run extraction to completion before testing resume')
    features = profile / 'features'
    def snapshot():
        return {p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in features.glob('*.pt')}
    before = snapshot()
    original_stats = json.loads((features / 'extraction_stats.json').read_text())
    original_path = profile / 'initial_extraction_stats.json'
    if not original_path.exists():
        original_path.write_text(json.dumps(original_stats,indent=2))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-u',str(root / 'scripts/run_hitph_mini_esm.py'),
                    '--output-root',str(profile),'--gpu-ids',args.gpu_ids],cwd=root,check=True)
    after = snapshot()
    stats = json.loads((features / 'extraction_stats.json').read_text())
    if before != after or stats['samples_generated'] != 0 or stats['samples_reused'] != len(before):
        raise RuntimeError('Resume changed caches or recomputed samples')
    report = {'status':'passed','unchanged_feature_files':len(before),
              'samples_generated':stats['samples_generated'],'samples_reused':stats['samples_reused'],
              'resume_wall_seconds':stats['total_wall_seconds']}
    (profile / 'resume_validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
