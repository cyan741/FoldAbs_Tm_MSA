#!/usr/bin/env python3
"""Run isolated streams/shared-GPU throughput probes with private MPS cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--phase',choices=['streams','multiprocess'],required=True)
    args=ap.parse_args();root=args.root
    checkpoint=json.loads((root/'pause.json').read_text())['resume']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONHASHSEED='0',
        PROTENIX_ROOT_DIR='/NAS/luyq/foldabs_immrep25_20261029/assets',LAYERNORM_TYPE='torch',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    results=[]

    def run(name,processes=1,extra=(),mps=False):
        case=root/name;case.mkdir(exist_ok=False)
        gate=case/'go';children=[];logs=[];started=time.monotonic()
        print('START',name,time.strftime('%H:%M:%S'),flush=True)
        try:
            for rank in range(processes):
                output=case/f'rank{rank}'
                command=[sys.executable,'-u','scripts/benchmark_group_parallel.py',
                    '--checkpoint',checkpoint,'--output',str(output),'--workers',str(args.workers),
                    '--rank',str(rank),'--processes',str(processes),'--gate',str(gate),*extra]
                log=(case/f'rank{rank}.log').open('w');logs.append(log)
                children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+300
            while not all((case/f'rank{r}/ready.json').exists() for r in range(processes)):
                if any(p.poll() is not None for p in children):raise RuntimeError('Worker failed before start gate')
                if time.monotonic()>deadline:raise TimeoutError('warmup')
                time.sleep(1)
            gate.write_text(str(time.time()))
            for p in children:
                if p.wait(timeout=300):raise RuntimeError('Worker failed')
            reports=[json.loads((case/f'rank{r}/result.json').read_text()) for r in range(processes)]
            assert len({d['global_groups_hash'] for d in reports})==1
            ids=[i for d in reports for i in d['group_ids']]
            assert len(ids)==len(set(ids))==sum(d['groups'] for d in reports)
            seconds=max(d['end_epoch'] for d in reports)-min(d['start_epoch'] for d in reports)
            row={'name':name,'processes':processes,'mps':mps,'groups':len(ids),
                 'samples':sum(d['samples'] for d in reports),'seconds':seconds,
                 'samples_per_second':sum(d['samples'] for d in reports)/seconds,
                 'peak_allocated_sum_MiB':sum(d['peak_allocated_MiB'] for d in reports),
                 'peak_reserved_sum_MiB':sum(d['peak_reserved_MiB'] for d in reports),
                 'global_groups_hash':reports[0]['global_groups_hash'],'returncode':0}
        except Exception as e:
            row={'name':name,'returncode':1,'error':repr(e)}
        finally:
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            for log in logs:log.close()
        row['process_wall_seconds']=time.monotonic()-started
        results.append(row);(root/f'phase_{args.phase}.json').write_text(json.dumps(results,indent=2))
        print('FINISH',json.dumps(row),flush=True)
        return row

    if args.phase=='streams':
        run('streams',extra=['--mode','streams'])
        run('serial_correctness',extra=['--groups','8','--dropout-zero'])
        run('streams_correctness',extra=['--groups','8','--dropout-zero','--mode','streams'])
    else:
        run('plain_two',2,['--activation-checkpoint'])
        pipe=root/'mps_pipe';log=root/'mps_log';pipe.mkdir();log.mkdir()
        env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log))
        try:
            subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
            one=run('mps_one',1,['--activation-checkpoint'],True)
            two=run('mps_two',2,['--activation-checkpoint'],True)
            # Only spend more GPU work if two processes improve aggregate throughput.
            if one['returncode']==two['returncode']==0 and two['samples_per_second']>one['samples_per_second']*1.05:
                three=run('mps_three',3,['--activation-checkpoint'],True)
                if three['returncode']==0 and three['samples_per_second']>two['samples_per_second']*1.05:
                    run('mps_four',4,['--activation-checkpoint'],True)
            else:
                (root/'more_processes_skipped.json').write_text(json.dumps({
                    'reason':'Two MPS processes did not improve throughput by >5%, or a probe failed.'},indent=2))
        finally:
            subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,check=False)


if __name__=='__main__':main()
