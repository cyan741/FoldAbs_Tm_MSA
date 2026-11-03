#!/usr/bin/env python3
"""Audit peptide-label priors induced by mismatch sampling; never modify training."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from train_tmp_specificity import PeptideMismatchSampling
from diagnose_bce_predictions import loss


def shortcut(frame):
    y=frame.label.to_numpy();ids=frame.pmhc_id.to_numpy()
    tcr=pd.factorize(list(zip(frame.tcra,frame.tcrb)))[0]
    probabilities=np.empty(len(frame))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,y,groups=tcr):
        stats=frame.iloc[tr].groupby('pmhc_id').label.agg(['sum','count'])
        # Beta prior centered at training fold prevalence, effective count 3.
        lookup=(stats['sum']+3*y[tr].mean())/(stats['count']+3)
        probabilities[te]=pd.Series(ids[te]).map(lookup).fillna(y[tr].mean())
    return {'bce':loss(y,logit(probabilities)).mean(),'auroc':roc_auc_score(y,probabilities),
            'constant_prevalence_bce':loss(y,np.full(len(y),logit(y.mean()))).mean(),
            'warning':'Within-split diagnostic with disjoint TCR folds; does not evaluate unseen-peptide transfer.'}


def main():
    root=Path('/NAS/luyq/foldabs_bce_diagnosis_20261029')
    sampling=PeptideMismatchSampling(SimpleNamespace(splits_dir=Path('/NAS/luyq/foldabs_immrep25_20261029/splits'),
        negatives_per_positive=2,negative_sampling_seed=42))
    train,_=sampling.epoch(6)
    val=sampling.manifest(sampling.frames['val'],'val')
    predicted=pd.read_csv(root/'epoch5_validation_predictions.csv')
    frame=val.merge(predicted[['sample_id','logit','bce']],on='sample_id',validate='one_to_one')
    assert len(frame)==7737 and np.max(abs(loss(frame.label,frame.logit)-frame.bce))<1e-6
    frame['hla_context']=frame.hla_sequence.map({v:k for k,v in sampling.hla_clips.items()})
    frame['probability']=expit(frame.logit)
    per=frame.groupby(['pmhc_id','peptide','hla_context']).agg(n=('label','size'),positives=('label','sum'),
        bce=('bce','mean'),mean_probability=('probability','mean'),mean_logit=('logit','mean')).reset_index()
    per['negatives']=per.n-per.positives;per['positive_fraction']=per.positives/per.n
    per['loss_contribution']=per.n*per.bce/len(frame)
    per=per.sort_values('loss_contribution',ascending=False)
    per.to_csv(root/'epoch5_analysis/phla_with_sequences.csv',index=False)
    top2=per.head(2).pmhc_id.tolist()
    included=frame.pmhc_id.isin(top2)
    result={'top2':per.head(2).to_dict('records'),'top2_total_loss_fraction':frame.loc[included,'bce'].sum()/frame.bce.sum(),
        'without_top2_bce':frame.loc[~included,'bce'].mean(),
        'top2_positive_fraction_of_all_positives':frame.loc[included,'label'].sum()/frame.label.sum(),
        'hla_stats':frame.groupby('hla_context').agg(n=('label','size'),positives=('label','sum'),bce=('bce','mean'),mean_probability=('probability','mean')).reset_index().to_dict('records'),
        'train_unique_peptides':train.peptide.nunique(),'val_unique_peptides':val.peptide.nunique(),
        'train_val_peptide_intersection':len(set(train.peptide)&set(val.peptide)),
        'peptide_only_tcr_disjoint_crossfit':{'train_epoch6':shortcut(train),'validation_internal_diagnostic':shortcut(val)}}
    train.groupby(['pmhc_id','peptide']).label.agg(['count','sum','mean']).to_csv(root/'epoch5_analysis/train_peptide_label_priors.csv')
    frame.to_csv(root/'epoch5_analysis/validation_with_sequences.csv',index=False)
    (root/'epoch5_analysis/sampling_shortcut_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
