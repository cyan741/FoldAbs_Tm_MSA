#!/usr/bin/env python3
import argparse,datetime,json
from pathlib import Path
from report_encoder_pool import summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();out=[]
    for row in json.loads((a.root/'summary.json').read_text()):
        if row['returncode']:out.append(row);continue
        name=row['name'];reports=[json.loads((a.root/name/f'rank{r}/result.json').read_text()) for r in range(row.get('world_size',2))]
        start=min(d['start_epoch'] for d in reports);end=max(d['end_epoch'] for d in reports)
        telemetry={0:[],1:[]}
        for line in (a.root/f'{name}_gpu.csv').read_text().splitlines():
            t,i,m,u=[s.strip() for s in line.split(',')];epoch=datetime.datetime.strptime(t,'%Y/%m/%d %H:%M:%S.%f').timestamp()
            telemetry[int(i)].append((epoch,int(m.split()[0]),int(u.split()[0])))
        ranks=[]
        for r,d in enumerate(reports):
            sync=[x['gradient_sync_ms'] for x in d['records'] if 'gradient_sync_ms' in x]
            ranks.append({'rank':r,'wait_seconds':d['wait_seconds'],'head_seconds':d['head_seconds'],'mean_gradient_sync_ms':sum(sync)/len(sync) if sync else None,'training':summarize([x for x in telemetry[r] if start<=x[0]<=end]),'whole_process':summarize(telemetry[r]),'encoder_has_grad':any(x['encoder_has_grad'] for x in d['encoder_reports']),'head_parameter_sha256':d['parameter_checksums'][r]})
        out.append(dict(row,ranks=ranks))
    (a.root/'summary_detailed.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
