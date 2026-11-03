#!/usr/bin/env python3
"""Compare matched head-kernel benchmarks and independent Torch repeats."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name('report_dual_encoder_pool.py')), str(args.root)], check=True, stdout=subprocess.DEVNULL)
    summaries = json.loads((args.root / 'summary_detailed.json').read_text())
    by_name = {row['name']: row for row in summaries}
    comparisons = []
    for world_size in (1, 2):
        baseline = by_name.get(f'gpu{world_size}_cuequivariance_r0')
        candidate = by_name.get(f'gpu{world_size}_torch_r0')
        if not baseline or not candidate or baseline['returncode'] or candidate['returncode']:
            continue
        assert baseline['global_groups_hash'] == candidate['global_groups_hash']
        assert baseline['optimizer_updates'] == candidate['optimizer_updates']
        comparisons.append({'world_size': world_size,
                            'torch_time_change_percent': 100 * (candidate['seconds'] / baseline['seconds'] - 1),
                            'torch_throughput_change_percent': 100 * (candidate['samples_per_second'] / baseline['samples_per_second'] - 1)})
    repeat = []
    for world_size in (1, 2):
        for rank in range(world_size):
            paths = [args.root / f'gpu{world_size}_torch_r{i}' / f'rank{rank}' / 'result.json' for i in range(2)]
            if not all(path.exists() for path in paths):
                continue
            a, b = [json.loads(path.read_text()) for path in paths]
            assert a['group_ids'] == b['group_ids']
            assert len(a['records']) == len(b['records'])
            pairs = list(zip(a['records'], b['records']))
            repeat.append({'world_size': world_size, 'rank': rank, 'updates': len(pairs),
                           'final_parameters_equal': a['parameter_checksums'][rank] == b['parameter_checksums'][rank],
                           'max_logit_difference': max(abs(x-y) for left, right in pairs for x,y in zip(left['logits'],right['logits'])),
                           'max_loss_difference': max(abs(left['loss']-right['loss']) for left,right in pairs),
                           'max_gradient_norm_difference': max(abs(left['gradient_norm']-right['gradient_norm']) for left,right in pairs)})
    report = {'performance': comparisons, 'torch_independent_repeats': repeat}
    (args.root / 'kernel_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
