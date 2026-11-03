#!/usr/bin/env python3
import argparse,datetime,json
from pathlib import Path
import numpy as np

def summarize(rows):
    if not rows:return {}
    u=np.array([r[2] for r in rows]);m=np.array([r[1] for r in rows])
    return {'samples':len(rows),'util_mean':float(u.mean()),'util_min':int(u.min()),'util_p05':float(np.percentile(u,5)),'util_p50':float(np.median(u)),'util_p95':float(np.percentile(u,95)),'fraction_ge95':float((u>=95).mean()),'fraction_below50':float((u<50).mean()),'fraction_zero':float((u==0).mean()),'physical_peak_MiB':int(m.max())}

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();out=[]
    for f in sorted(a.root.glob('enc*/result.json')):
        d=json.loads(f.read_text());rows=[]
        for line in (a.root/(f.parent.name+'_gpu.csv')).read_text().splitlines():
            t,i,m,u=[v.strip() for v in line.split(',')]
            if i=='0':rows.append((datetime.datetime.strptime(t,'%Y/%m/%d %H:%M:%S.%f').timestamp(),int(m.split()[0]),int(u.split()[0])))
        train=[r for r in rows if d['start_epoch']<=r[0]<=d['end_epoch']]
        e=d['encoders'];rs=d['records'];left=d['start_epoch']+sum(r['wait_seconds']+r['head_seconds'] for r in rs[:e]);right=d['end_epoch']-sum(r['wait_seconds']+r['head_seconds'] for r in rs[-e:])
        result={k:v for k,v in d.items() if k not in ('records','encoder_reports')}
        result.update(name=f.parent.name,whole_process=summarize(rows),training=summarize(train),middle=summarize([r for r in rows if left<=r[0]<=right] if len(rs)>2*e else []),middle_wait_seconds=sum(r['wait_seconds'] for r in rs[e:-e]),middle_head_seconds=sum(r['head_seconds'] for r in rs[e:-e]))
        out.append(result)
    (a.root/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
