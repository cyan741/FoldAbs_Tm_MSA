#!/usr/bin/env python3
"""Summarize matched checkpoint probes and render static diagnostic plots."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from diagnose_bce_predictions import summary,loss
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    root=Path(sys.argv[1]);out=root/'epoch5_analysis'
    paired=pd.read_csv(root/'paired_probe/paired_predictions.csv')
    assert len(paired)==720
    manifests={s:pd.read_csv(root/f'paired_probe/{s}_manifest.csv') for s in ['train','validation']}
    epoch5=pd.read_csv(root/'epoch5_validation_predictions.csv')
    selected=manifests['validation'][['sample_id']].merge(epoch5,on='sample_id',validate='one_to_one')
    selected['split']='validation';selected['checkpoint']='epoch5'
    allrows=pd.concat([paired,selected[paired.columns]],ignore_index=True)
    result={}
    top=['VLDFAPPGA','YLQPRTFLL']
    for split in ['train','validation']:
        frame=allrows[allrows.split.eq(split)].merge(manifests[split][['sample_id','peptide']],on='sample_id',validate='many_to_one')
        result[split]={c:summary(g) for c,g in frame.groupby('checkpoint')}
        if split=='validation':
            result['dominant_peptides_on_matched_validation']={}
            for peptide in top:
                result['dominant_peptides_on_matched_validation'][peptide]={c:summary(g) for c,g in frame[frame.peptide.eq(peptide)].groupby('checkpoint') if g.label.nunique()==2}
            # Include per-class means even if a small peptide subset lacks negatives.
            result['matched_top_peptide_positive_probabilities']={p:{c:{'n':len(g),'mean_probability':expit(g.logit).mean(),'mean_bce':loss(g.label,g.logit).mean()}
                for c,g in frame[frame.peptide.eq(p)&frame.label.eq(1)].groupby('checkpoint')} for p in top}
    (out/'matched_checkpoint_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    history=json.loads((root/'history_snapshot.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    ax=axes[0,0];epochs=[r['epoch'] for r in history]
    ax.plot(epochs,[r['train_bce'] for r in history],marker='o',label='Train BCE')
    ax.plot(epochs,[r['validation']['bce'] for r in history],marker='o',label='Validation BCE')
    ax.axhline(np.log(2),ls=':',color='gray',label='Constant p=0.5')
    ax.axhline(.6365141683,ls='--',color='black',label='Constant p=1/3')
    ax.set(xlabel='Epoch',ylabel='BCE',title='Full-dataset training / validation history');ax.legend(fontsize=8)
    ax=axes[0,1]
    f=pd.read_csv(out/'validation_with_sequences.csv')
    for label,color in [(0,'tab:blue'),(1,'tab:orange')]:
        ax.hist(f.loc[f.label.eq(label),'logit'],bins=np.linspace(-8,5,40),density=True,alpha=.5,label=f'Label {label}',color=color)
    ax.axvline(-np.log(2),color='black',ls='--',label='Prior logit (p=1/3)')
    ax.set(xlabel='Logit',ylabel='Density',title='Epoch 5: unseen-peptide validation');ax.legend(fontsize=8)
    ax=axes[1,0]
    per=pd.read_csv(out/'phla_with_sequences.csv').head(6).iloc[::-1]
    ax.barh(per.peptide,per.loss_contribution,color='tab:orange')
    ax.set(xlabel='Contribution to total mean BCE',title='Epoch 5 loss concentrated in two peptides')
    ax=axes[1,1]
    for label,name,color in [(0,'Negative','tab:blue'),(1,'Positive','tab:orange')]:
        for split,style in [('train','-'),('validation','--')]:
            vals=[result[split][c][f'label_{label}']['mean_probability'] for c in ['early','late']]
            ax.plot([1,6],vals,style,marker='o',color=color,label=f'{split} {name}')
    ax.set(xticks=[1,6],xticklabels=['Epoch 1','Epoch 6 snapshot'],ylabel='Mean predicted positive probability',
           title='Same-input checkpoint probe (120 train / 240 val)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'bce_diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({s:{c:{k:r[k] for k in ['n','bce','auroc','logit_mean','logit_sd','mean_predicted_probability']} for c,r in result[s].items()} for s in ['train','validation']},indent=2))
    print(json.dumps(result['matched_top_peptide_positive_probabilities'],indent=2))


if __name__=='__main__':main()
