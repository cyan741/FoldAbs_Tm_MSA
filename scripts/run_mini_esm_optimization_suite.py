#!/usr/bin/env python3
"""Sequential, same-GPU training throughput ablations; no feature writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/data/luyq/mini_esm_optimization_20261017')
script=Path(__file__).with_name('benchmark_mini_esm_routes.py').resolve()
results=[]

def run(name, kernel='torch', checkpoint=True, workers=0, cycles=4, blocks=4):
    work=root/name
    command=[sys.executable,'-u',str(script),'--input-json',str(root/'input.json'),
        '--labels-json',str(root/'labels.json'),'--esm-cache-dir',str(root/'esm_cache'),
        '--work-dir',str(work),'--features-dir',str(root/'unused'), '--mode','online',
        '--samples','18','--warmup-samples','3','--head-kernel',kernel,
        '--cycles',str(cycles),'--head-blocks',str(blocks),'--cpu-workers',str(workers),
        '--activation-checkpoint' if checkpoint else '--no-activation-checkpoint']
    print('START',name,flush=True)
    start=time.perf_counter()
    with (root/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    row={'name':name,'return_code':result.returncode,'command':command,'process_wall_seconds':time.perf_counter()-start}
    if result.returncode==0:
        row['report']=json.loads((work/'report.json').read_text())
    else:
        row['error_tail']='\n'.join((root/(name+'.log')).read_text().splitlines()[-18:])
    results.append(row)
    (root/'suite_results.json').write_text(json.dumps(results,indent=2))
    print('FINISH',name,'code',result.returncode,flush=True)
    return row

run('cueq_no_checkpoint',kernel='cuequivariance',checkpoint=False)
run('baseline_torch_checkpoint')
run('torch_no_checkpoint',checkpoint=False)
run('cueq_checkpoint',kernel='cuequivariance')
valid=[r for r in results if r['return_code']==0]
if not valid: raise RuntimeError('All implementations failed')
best=min(valid,key=lambda r:r['report']['summary']['wall_seconds']['mean'])['report']
run('best_prefetch2',kernel=best['head_kernel'],checkpoint=best['activation_checkpoint'],workers=2)
valid=[r for r in results if r['return_code']==0]
best=min(valid,key=lambda r:r['report']['summary']['wall_seconds']['mean'])['report']
(root/'best_fixed_architecture.json').write_text(json.dumps(best,indent=2))
for cycles,blocks in [(2,4),(1,4),(4,2),(4,1),(2,2),(1,1)]:
    run(f'cycles{cycles}_head{blocks}',kernel=best['head_kernel'],checkpoint=best['activation_checkpoint'],workers=best['cpu_workers'],cycles=cycles,blocks=blocks)
print('SUITE COMPLETE',flush=True)
