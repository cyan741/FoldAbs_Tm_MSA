#!/usr/bin/env python3
"""Launch one generation-to-Foldtmp benchmark worker on each of eight GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--features-dir',type=Path,required=True)
a=p.parse_args()
start=time.perf_counter()
workers=[]
for gpu in range(8):
    work=a.root/f'write8_gpu{gpu}'
    log=(a.root/f'write8_gpu{gpu}.log').open('w')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    command=[sys.executable,'-u',str(Path(__file__).with_name('benchmark_mini_esm_routes.py')),
        '--input-json',str(a.root/'input.json'),'--esm-cache-dir',str(a.root/'esm_cache'),
        '--features-dir',str(a.features_dir),'--work-dir',str(work),
        '--mode','write','--offset',str(gpu),'--stride','8','--samples','6']
    proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    workers.append((gpu,proc,log,work))
report={'host':os.uname().nodename,'workers':[],'includes_startup':True,
        'exclusive_gpus':False,'samples_per_gpu':6}
for gpu,proc,log,work in workers:
    code=proc.wait();log.close()
    r={'gpu':gpu,'return_code':code}
    if (work/'report.json').exists(): r['report']=json.loads((work/'report.json').read_text())
    report['workers'].append(r)
report['wall_seconds']=time.perf_counter()-start
report['saved_count']=sum(len(w.get('report',{}).get('samples',[])) for w in report['workers'])
report['saved_bytes']=sum(r['file_bytes'] for w in report['workers'] for r in w.get('report',{}).get('samples',[]))
report['samples_per_second']=report['saved_count']/report['wall_seconds']
(a.root/'eight_gpu_write_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='workers'}),flush=True)
if any(w['return_code'] for w in report['workers']): raise SystemExit(1)
