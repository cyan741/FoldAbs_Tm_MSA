#!/usr/bin/env python3
"""Read-only BCE diagnosis: exact loss decomposition and held-out calibration."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold


def loss(y,z):
    return np.logaddexp(0,z)-y*z


def summary(frame):
    y=frame.label.to_numpy(float); z=frame.logit.to_numpy(float); p=expit(z)
    pi=y.mean(); mu=z.mean(); h=-pi*np.log(pi)-(1-pi)*np.log1p(-pi)
    l=loss(y,z); cov=np.mean((y-pi)*(z-mu))
    result={'n':len(y),'prevalence':pi,'bce':l.mean(),'constant_0.5_bce':np.log(2),
        'constant_prevalence_bce':h,'auroc':roc_auc_score(y,z) if len(set(y))==2 else None,
        'auprc':average_precision_score(y,z),'mean_predicted_probability':p.mean(),
        'logit_mean':mu,'logit_sd':z.std(), 'brier':np.mean((p-y)**2),
        'decomposition':{'prior_entropy':h,'mean_logit_offset_cost':np.logaddexp(0,mu)-pi*mu-h,
                         'logit_dispersion_cost':np.logaddexp(0,z).mean()-np.logaddexp(0,mu),
                         'label_score_covariance_gain':cov}}
    for label in (0,1):
        mask=y==label
        result[f'label_{label}']={'n':int(mask.sum()),'bce':l[mask].mean(),
          'bce_contribution':l[mask].sum()/len(y), 'mean_probability':p[mask].mean(),
          'logit_quantiles':np.quantile(z[mask],[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
          'probability_quantiles':np.quantile(p[mask],[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
          'wrong_at_0.5':int(((p[mask]>=.5)!=label).sum()),
          'confident_wrong_0.9':int(((p[mask]>=.9) if label==0 else (p[mask]<=.1)).sum())}
    result['top_loss_contribution']={str(frac):np.sort(l)[-max(1,int(np.ceil(len(l)*frac))):].sum()/l.sum()
                                     for frac in (.01,.05,.1)}
    return result


def fit(y,z,kind):
    if kind=='temperature':
        opt=minimize(lambda x:loss(y,z*x[0]).mean(),[1.],bounds=[(0.,20.)],method='L-BFGS-B')
        return float(opt.x[0]),0.
    if kind=='offset':
        opt=minimize(lambda x:loss(y,z+x[0]).mean(),[0.],bounds=[(-20.,20.)],method='L-BFGS-B')
        return 1.,float(opt.x[0])
    opt=minimize(lambda x:loss(y,z*x[0]+x[1]).mean(),[1.,0.],bounds=[(0.,20.),(-20.,20.)],method='L-BFGS-B')
    return float(opt.x[0]),float(opt.x[1])


def analyze(frame):
    result=summary(frame)
    y=frame.label.to_numpy(float);z=frame.logit.to_numpy(float)
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(z,y,groups=frame.pmhc_id))
    result['held_out_phla_calibration']={}
    for kind in ('offset','temperature','affine'):
        adjusted=np.empty(len(frame)); parameters=[]
        for train,test in folds:
            a,b=fit(y[train],z[train],kind);adjusted[test]=a*z[test]+b
            parameters.append({'a':a,'b':b})
        result['held_out_phla_calibration'][kind]={'bce':loss(y,adjusted).mean(),'fold_parameters':parameters}
    g=frame.groupby('pmhc_id')
    meanz=g.logit.transform('mean').to_numpy(); meany=g.label.transform('mean').to_numpy()
    result['phla_effects']={'num_phla':g.ngroups,'between_phla_logit_variance_fraction':np.var(meanz)/np.var(z),
        'between_phla_label_score_covariance':np.mean((meany-y.mean())*(meanz-z.mean())),
        'within_phla_label_score_covariance':np.mean((y-meany)*(z-meanz)),
        'diagnostic_phla_centered_bce':loss(y,z-meanz+np.log(y.mean()/(1-y.mean()))).mean(),
        'diagnostic_phla_centered_auroc':roc_auc_score(y,z-meanz)}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    frame=pd.read_csv(args.predictions);args.output_dir.mkdir(parents=True,exist_ok=True)
    assert frame.sample_id.is_unique and frame.label.isin([0,1]).all()
    result=analyze(frame)
    if 'bce' in frame:
        result['max_saved_bce_difference']=float(np.max(np.abs(frame.bce-loss(frame.label,frame.logit))))
    (args.output_dir/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    frame=frame.assign(recomputed_bce=loss(frame.label,frame.logit),probability=expit(frame.logit))
    per=frame.groupby('pmhc_id').agg(n=('label','size'), positives=('label','sum'),bce=('recomputed_bce','mean'),
        mean_logit=('logit','mean'),sd_logit=('logit','std'),mean_probability=('probability','mean'))
    per['loss_contribution']=per.n*per.bce/len(frame)
    per.sort_values('loss_contribution',ascending=False).to_csv(args.output_dir/'phla_diagnostics.csv')
    frame.nlargest(100,'recomputed_bce').to_csv(args.output_dir/'top_loss_records.csv',index=False)
    bins=pd.cut(frame.probability,np.linspace(0,1,11),include_lowest=True)
    frame.groupby(bins,observed=True).agg(n=('label','size'),positive_rate=('label','mean'),
        mean_probability=('probability','mean'),bce=('recomputed_bce','mean')).to_csv(args.output_dir/'reliability_bins.csv')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
