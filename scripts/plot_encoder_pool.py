#!/usr/bin/env python3
import argparse,datetime,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    reports=[]
    for p in sorted(args.root.glob('enc*/result.json')):
        d=json.loads(p.read_text())
        if not d.get('verification_run',False):reports.append((p.parent.name,d))
    reports.sort(key=lambda x:x[1]['encoders'])
    fig,axes=plt.subplots(len(reports),1,figsize=(11,2.4*len(reports)),squeeze=False)
    for ax,(name,d) in zip(axes[:,0],reports):
        rows=[]
        for line in (args.root/(name+'_gpu.csv')).read_text().splitlines():
            t,i,m,u=[v.strip() for v in line.split(',')]
            if i=='0':rows.append((datetime.datetime.strptime(t,'%Y/%m/%d %H:%M:%S.%f').timestamp(),float(u.split()[0])))
        start=rows[0][0]
        ax.plot([r[0]-start for r in rows],[r[1] for r in rows],lw=1,label='GPU utilization (1 s samples)')
        ax.axvspan(d['start_epoch']-start,d['end_epoch']-start,color='green',alpha=.12,label='Measured training interval')
        ax.axhline(95,color='orange',ls='--',lw=.8)
        ax.set_ylim(-3,105);ax.set_ylabel('GPU util (%)');ax.set_title(f"{d['encoders']} encoders + 1 head | {d['samples']} samples | {d['seconds']:.2f} s | input wait {d['wait_seconds']:.2f} s",fontsize=10)
        ax.grid(alpha=.2)
    axes[0,0].legend(loc='lower right',fontsize=8);axes[-1,0].set_xlabel('Seconds since process launch')
    fig.tight_layout();fig.savefig(args.root/'gpu_timeline.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
