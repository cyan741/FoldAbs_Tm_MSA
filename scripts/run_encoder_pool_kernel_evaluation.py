#!/usr/bin/env python3
"""Compare only head kernels with four frozen cuEq encoders per GPU; isolated."""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--cases',nargs='+');a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1',PYTHONHASHSEED='0',PROTENIX_ROOT_DIR='/NAS/luyq/foldabs_immrep25_20261029/assets',LAYERNORM_TYPE='torch',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MPS_PIPE_DIRECTORY=str(a.root/'pipe'),CUDA_MPS_LOG_DIRECTORY=str(a.root/'mps_log'))
    (a.root/'pipe').mkdir();(a.root/'mps_log').mkdir()
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True)
    summary=[]
    failed_torch_world_sizes=set()
    try:
        for world_size,kernel,repeat in [(1,'cuequivariance',0),(1,'torch',0),(1,'torch',1),(2,'cuequivariance',0),(2,'torch',0),(2,'torch',1)]:
            if repeat and world_size in failed_torch_world_sizes:continue
            encoders,groups=4,48
            name=f'gpu{world_size}_{kernel}_r{repeat}'
            if a.cases and name not in a.cases:continue
            case=a.root/name;case.mkdir();print('START',name,flush=True)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            init=f'tcp://127.0.0.1:{port}';children=[];logs=[];started=time.time()
            monitor_log=(a.root/f'{name}_gpu.csv').open('w')
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu','--format=csv,noheader','-l','1'],stdout=monitor_log)
            try:
                for rank in range(world_size):
                    child_env=dict(env,CUDA_VISIBLE_DEVICES=str(rank))
                    log=(case/f'rank{rank}.log').open('w');logs.append(log)
                    cmd=[sys.executable,'-u','scripts/benchmark_encoder_pool_ddp.py','--checkpoint',str(a.checkpoint),'--output',str(case/f'rank{rank}'),'--encoders',str(encoders),'--groups',str(groups),'--stable-aggregate','--rank',str(rank),'--world-size',str(world_size),'--head-kernel',kernel,'--init-method',init]
                    children.append(subprocess.Popen(cmd,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
                deadline=time.monotonic()+1200
                while any(p.poll() is None for p in children):
                    if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Head rank failed')
                    if time.monotonic()>deadline:raise TimeoutError('dual benchmark')
                    time.sleep(1)
                if any(p.returncode for p in children):raise RuntimeError('Nonzero rank exit')
                reports=[json.loads((case/f'rank{r}/result.json').read_text()) for r in range(world_size)]
                ids=[i for d in reports for i in d['group_ids']]
                assert len(ids)==len(set(ids))==groups
                assert len({d['global_groups_hash'] for d in reports})==1
                assert all(d['parameter_checksums_equal'] for d in reports)
                seconds=max(d['end_epoch'] for d in reports)-min(d['start_epoch'] for d in reports)
                samples=sum(d['samples'] for d in reports)
                row={'name':name,'encoders_per_gpu':encoders,'world_size':world_size,'head_kernel':kernel,'activation_checkpoint':False,'global_groups':groups,'samples':samples,'optimizer_updates':groups//world_size,'nominal_global_batch':3*world_size,'seconds':seconds,'samples_per_second':samples/seconds,'parameter_checksums_equal':True,'global_groups_hash':reports[0]['global_groups_hash'],'process_seconds':time.time()-started,'returncode':0}
            except Exception as e:
                row={'name':name,'returncode':1,'error':repr(e)}
            finally:
                for p in children:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                for p in children:
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                for log in logs:log.close()
                monitor.terminate();monitor.wait();monitor_log.close()
            summary.append(row);(a.root/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(row),flush=True)
            if row['returncode'] and kernel=='torch':failed_torch_world_sizes.add(world_size)
    finally:subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env)
if __name__=='__main__':main()
