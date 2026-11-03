#!/usr/bin/env python3
"""Resume accelerated fixed-sample extraction in locally staged chunks.

Training negatives are intentionally absent: pass only the baseline input.
Each chunk preserves a validated index for recovery and final aggregation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def publish(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--local-root',type=Path,required=True)
    p.add_argument('--gpu-ids',default='1,4')
    p.add_argument('--chunk-size',type=int,default=256)
    a=p.parse_args()
    import fcntl
    a.local_root.mkdir(parents=True,exist_ok=True)
    lock=(a.local_root/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=a.profile/'esm_cache'
    cache=a.local_root/'esm_cache'
    features=a.profile/'features_cueq'
    features.mkdir(exist_ok=True)
    for part in ['metadata','embeddings/chains']: (cache/part).mkdir(parents=True,exist_ok=True)
    spec=json.loads((source/'run_spec.json').read_text())
    if (cache/'run_spec.json').exists() and json.loads((cache/'run_spec.json').read_text())!=spec:
        raise ValueError('Local ESM cache profile mismatch')
    publish(cache/'run_spec.json',spec)
    records=json.loads((a.profile/'inputs/baseline.json').read_text())
    expected={r['name'] for r in records}
    if len(expected)!=len(records): raise ValueError('Duplicate input identities')
    state={'status':'running','total':len(records),'completed':0,'gpu_ids':a.gpu_ids,
           'kernel_backend':'cuequivariance/cuequivariance','training_negatives':'online_not_cached',
           'features_dir':str(features),'started_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    rows=[]
    try:
        for offset in range(0,len(records),a.chunk_size):
            chunk=records[offset:offset+a.chunk_size]
            work=a.local_root/f'chunk_{offset:06d}'
            work.mkdir(exist_ok=True)
            index_path=work/'validated_index.json'
            if index_path.exists():
                saved=json.loads(index_path.read_text())
                if {r['sample_id'] for r in saved}=={r['name'] for r in chunk} and all((features/r['file']).is_file() for r in saved):
                    rows.extend(saved);state['completed']=len(rows);continue
            state.update(chunk_offset=offset,phase='staging_esm')
            publish(a.local_root/'state.json',state)
            for record in chunk:
                for entry in record['sequences']:
                    sequence=entry['proteinChain']['sequence']
                    key=hashlib.sha256(sequence.encode('ascii')).hexdigest()
                    for part,suffix in [('metadata','.json'),('embeddings/chains','.pt')]:
                        target=cache/part/(key+suffix)
                        if not target.exists():
                            for attempt in range(3):
                                try:
                                    temp=target.with_suffix(target.suffix+'.tmp')
                                    shutil.copyfile(source/part/(key+suffix),temp)
                                    os.replace(temp,target)
                                    break
                                except OSError:
                                    if attempt==2: raise
                                    time.sleep(2)
            path=work/'input.json';publish(path,chunk)
            command=[sys.executable,'-u',str(Path(__file__).with_name('extract_protenix_embeddings.py')),
                '--input-json',str(path),'--output-dir',str(features),'--work-dir',str(work),
                '--esm-cache-dir',str(cache),'--model-name','protenix_mini_esm_v0.5.0',
                '--cycles','4','--no-msa','--seed','42','--dtype','bf16','--cache-dtype','fp16',
                '--limit','0','--resume','--gpu-ids',a.gpu_ids,
                '--trimul-kernel','cuequivariance','--triatt-kernel','cuequivariance']
            state['phase']='extracting';publish(a.local_root/'state.json',state)
            for attempt in range(3):
                with (work/f'extract_attempt_{attempt}.log').open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0: break
                if attempt==2: raise RuntimeError(f'Chunk {offset} failed; see {work}')
                time.sleep(5)
            saved=json.loads((features/'index.json').read_text())
            if {r['sample_id'] for r in saved}!={r['name'] for r in chunk}: raise ValueError('Incomplete chunk')
            for row in saved: row['input_index']+=offset
            publish(index_path,saved)
            rows.extend(saved);state['completed']=len(rows)
            publish(a.local_root/'state.json',state)
            print(json.dumps(state),flush=True)
        if {r['sample_id'] for r in rows}!=expected: raise ValueError('Incomplete aggregate')
        publish(features/'index.json',rows)
        state.update(status='complete',phase='complete',cache_bytes=sum(r['file_bytes'] for r in rows),
                     completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        publish(features/'extraction_stats.json', {
            'samples_total':len(rows), 'total_cache_bytes':state['cache_bytes'],
            'extraction_config':json.loads((features/'run_spec.json').read_text()),
            'gpu_ids':a.gpu_ids.split(','), 'source':'validated_chunk_indexes',
            'training_negatives':'online_not_cached'})
        publish(a.local_root/'state.json',state)
        publish(a.profile/'completion_report_cueq.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        publish(a.local_root/'state.json',state)
        raise

if __name__=='__main__': main()
