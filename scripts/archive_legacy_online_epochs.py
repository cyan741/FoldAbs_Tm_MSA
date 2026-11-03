#!/usr/bin/env python3
"""Archive completed epochs from a running legacy online trainer without restarting it."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from foldabs_repro.tmp_prediction.checkpoints import archive_epoch,nominal_batch_size


def archive_completed(source, directory):
    value=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    history=value.get('history',[])
    if value.get('cursor') != 0 or not history or value['epoch'] != history[-1]['epoch']+1:
        return None
    epoch=history[-1]['epoch'];cfg=value['run_config']
    value={**value,'checkpoint_stage':'validation_complete','epoch_archive_source':str(source)}
    return archive_epoch(value,directory,batch_size=nominal_batch_size(SimpleNamespace(**cfg),cfg['world_size']),
                         learning_rate=cfg['learning_rate'],seed=cfg['seed'],epoch=epoch)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--once',action='store_true')
    args=p.parse_args();seen={};retained=set()
    status=args.run_dir/'epoch_archive_monitor.json'
    while True:
        for name in ('best.pt','last.pt'):
            source=args.run_dir/name
            if not source.exists():continue
            st=source.stat();identity=(st.st_ino,st.st_mtime_ns,st.st_size)
            if seen.get(name)==identity:continue
            path=archive_completed(source,args.run_dir)
            seen[name]=identity
            if path:
                retained.add(path.name)
                print(json.dumps({'event':'epoch_archived','source':name,'path':str(path)}),flush=True)
        complete=(args.run_dir/'completion.json').exists()
        value={'status':'complete' if complete else 'watching','archives':sorted(retained),
               'last_poll_unix':time.time(),'policy':'copy completed legacy epoch checkpoints; never stop training'}
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(status)
        if complete or args.once:break
        time.sleep(2)


if __name__=='__main__':main()
