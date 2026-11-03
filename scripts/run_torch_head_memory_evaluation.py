#!/usr/bin/env python3
"""Isolated A-D Torch-head memory sweep, then best-case repeat and long run."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def run_case(root, checkpoint, env, name, world_size, encoders, recompute, groups=48):
    case = root / name
    case.mkdir()
    row = dict(name=name, world_size=world_size, encoders_per_gpu=encoders,
               activation_checkpoint=recompute, head_kernel='torch',
               global_groups=groups, nominal_global_batch=3 * world_size)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    children, logs = [], []
    started = time.time()
    telemetry = (root / f'{name}_gpu.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu',
                                '--format=csv,noheader', '-l', '1'], stdout=telemetry)
    print('START', name, flush=True)
    try:
        for rank in range(world_size):
            log = (case / f'rank{rank}.log').open('w')
            logs.append(log)
            command = [sys.executable, '-u', 'scripts/benchmark_encoder_pool_ddp.py',
                       '--checkpoint', str(checkpoint), '--output', str(case / f'rank{rank}'),
                       '--encoders', str(encoders), '--groups', str(groups), '--stable-aggregate',
                       '--rank', str(rank), '--world-size', str(world_size), '--head-kernel', 'torch',
                       '--init-method', f'tcp://127.0.0.1:{port}']
            if recompute:
                command.append('--activation-checkpoint')
            children.append(subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)),
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 1200
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('Head rank failed; see rank logs')
            if time.monotonic() > deadline:
                raise TimeoutError('Benchmark exceeded 1200 seconds')
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError('Nonzero rank exit; see rank logs')
        reports = [json.loads((case / f'rank{r}/result.json').read_text()) for r in range(world_size)]
        ids = [group for report in reports for group in report['group_ids']]
        assert len(ids) == len(set(ids)) == groups
        assert len({report['global_groups_hash'] for report in reports}) == 1
        assert all(report['parameter_checksums_equal'] for report in reports)
        assert not any(encoder['encoder_has_grad'] for report in reports for encoder in report['encoder_reports'])
        seconds = max(r['end_epoch'] for r in reports) - min(r['start_epoch'] for r in reports)
        samples = sum(r['samples'] for r in reports)
        row.update(returncode=0, samples=samples, optimizer_updates=groups // world_size,
                   seconds=seconds, samples_per_second=samples / seconds,
                   parameter_checksums_equal=True, global_groups_hash=reports[0]['global_groups_hash'])
    except Exception as error:
        row.update(returncode=1, error=repr(error))
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
        monitor.terminate()
        monitor.wait()
        telemetry.close()
    row['process_seconds'] = time.time() - started
    print('DONE', json.dumps(row), flush=True)
    return row


def compare_repeats(root, original, repeated):
    rows = []
    for rank in range(original['world_size']):
        a, b = [json.loads((root / row['name'] / f'rank{rank}/result.json').read_text())
                for row in (original, repeated)]
        assert a['group_ids'] == b['group_ids']
        assert len(a['records']) == len(b['records'])
        pairs = list(zip(a['records'], b['records']))
        rows.append(dict(rank=rank, updates=len(pairs),
                         final_parameters_equal=a['parameter_checksums'][rank] == b['parameter_checksums'][rank],
                         max_logit_difference=max(abs(x-y) for left, right in pairs
                                                  for x, y in zip(left['logits'], right['logits'])),
                         max_loss_difference=max(abs(left['loss']-right['loss']) for left, right in pairs),
                         max_gradient_norm_difference=max(abs(left['gradient_norm']-right['gradient_norm'])
                                                          for left, right in pairs)))
    passed = all(r['final_parameters_equal'] and r['max_logit_difference'] == 0
                 and r['max_loss_difference'] == 0 and r['max_gradient_norm_difference'] == 0 for r in rows)
    return dict(original=original['name'], repeated=repeated['name'], passed=passed, ranks=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONHASHSEED='0',
               PROTENIX_ROOT_DIR='/NAS/luyq/foldabs_immrep25_20261029/assets', LAYERNORM_TYPE='torch',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               CUDA_MPS_PIPE_DIRECTORY=str(args.root / 'pipe'), CUDA_MPS_LOG_DIRECTORY=str(args.root / 'mps_log'))
    (args.root / 'pipe').mkdir()
    (args.root / 'mps_log').mkdir()
    subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True)
    summary = []
    def run(name, world_size, encoders, recompute, groups=48):
        row = run_case(args.root, args.checkpoint, env, name, world_size, encoders, recompute, groups)
        summary.append(row)
        write_json(args.root / 'summary.json', summary)
        return row
    try:
        for case in [('A_dual_enc4_cp', 2, 4, True), ('B_dual_enc3', 2, 3, False),
                     ('C_dual_enc2', 2, 2, False), ('D_single_enc4_cp', 1, 4, True)]:
            run(*case)
        successful = [row for row in summary if row['world_size'] == 2 and row['returncode'] == 0]
        if not successful:
            write_json(args.root / 'selection.json', dict(error='No dual-GPU case completed'))
            return
        best = max(successful, key=lambda row: row['samples_per_second'])
        write_json(args.root / 'selection.json', dict(selected=best['name'], criterion='Highest measured throughput among completed A/B/C; memory headroom reviewed separately'))
        repeat = run('E_repeat_' + best['name'], 2, best['encoders_per_gpu'], best['activation_checkpoint'])
        if repeat['returncode']:
            return
        comparison = compare_repeats(args.root, best, repeat)
        write_json(args.root / 'determinism.json', comparison)
        print('DETERMINISM', json.dumps(comparison), flush=True)
        if comparison['passed']:
            run('F_long_' + best['name'], 2, best['encoders_per_gpu'], best['activation_checkpoint'], groups=96)
    finally:
        subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env)


if __name__ == '__main__':
    main()
