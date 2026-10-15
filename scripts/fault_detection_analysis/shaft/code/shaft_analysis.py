#!/usr/bin/env python3
"""Reproduce features, full-run LOOCV and fixed alarm thresholds.

python3 shaft_analysis.py --manifest runs.json --output-dir results
Each whole held-out run is excluded from scaling and LDA fitting.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,platform
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import shaft_features as f


def models():
    registry={}
    def add(key,columns,label):registry[key]=dict(features=columns,label=label)
    d=f.definitions().set_index('feature')
    for m in range(4):
        for param in f.CURRENT:
            key=f'M{m}_{param}';add(key,[key],d.loc[key,'label'])
    for param in f.CURRENT:add('M1M2_'+param,['M1_'+param,'M2_'+param],'M1 + M2 '+d.loc['M1_'+param,'label'][3:])
    for key in f.VIDEO:add(key,[key],d.loc[key,'label'])
    add('gain_pair',['gain_positive','gain_negative'],'Positive + negative gain')
    add('video_compact',['gain_difference','video_mae'],'Gain difference + centred video MAE')
    for m in [1,2]:
        for param in f.CURRENT:
            current=f'M{m}_{param}'
            for vid,suffix,label in [('gain_difference','gain','gain difference'),('video_mae','video','video MAE')]:
                add(current+'_'+suffix,[current,vid],d.loc[current,'label']+' + '+label)
    add('current_plane',['M1_absolute_maximum','M2_rms'],'M1 absolute maximum + M2 RMS')
    add('current_compact',['M1_absolute_maximum','M2_mean','M2_a2'],'M1 peak + M2 mean + M2 A2')
    add('mixed_compact',['M1_absolute_maximum','M2_mean','M2_a2','gain_difference'],'Three current features + gain difference')
    active=[f'M{m}_{p}' for m in [1,2] for p in f.CURRENT]
    add('all_active_current',active,'All 22 M1/M2 current parameters')
    add('all_active_mixed',active+f.VIDEO,'All 22 active current + 7 video parameters')
    return registry


MAIN=['M1_absolute_maximum','M1_rms','M2_mean','M2_a2','M2_rc','gain_difference','gain_pair','video_mae',
      'M1_absolute_maximum_gain','M2_mean_gain','M2_a2_gain','M2_a2_video','current_compact','mixed_compact']
AMP=['M1_absolute_maximum','M2_mean','M2_a2','gain_difference','gain_pair','M2_mean_gain','M2_a2_gain']


def estimator():
    return make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))


def counts(y,z,offset=0.):
    tn,fp,fn,tp=confusion_matrix(y,np.asarray(z)+offset>0,labels=[0,1]).ravel()
    return dict(tp=int(tp),fn=int(fn),fp=int(fp),tn=int(tn),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),
                balanced_accuracy=float(.5*(tp/(tp+fn)+tn/(tn+fp))))


def evaluate(cycles,conditions,out):
    predictions=[];folds=[];coefficients=[]
    reg=models()
    for window,frame in [(1,cycles),(3,conditions)]:
        for strategy in ['pooled','amplitude_specific']:
            scopes=[('all',frame)] if strategy=='pooled' else [(rg,frame[frame['range']==rg]) for rg in f.RANGES]
            for fit_range,part in scopes:
                print('LOOCV',window,strategy,fit_range,flush=True)
                for key,info in reg.items():
                    columns=info['features'];X=part[columns].to_numpy();y=part.true_class.to_numpy()
                    for run in part.run.unique():
                        test=(part.run==run).to_numpy();train=~test
                        if set(np.unique(y[train]))!={0,1}:raise ValueError('Training fold lacks a class')
                        fit=estimator().fit(X[train],y[train]);score=fit.decision_function(X[test])
                        training_runs=list(part.loc[train,'run'].unique())
                        if run in training_runs:raise ValueError('Run leakage')
                        meta=dict(window_cycles=window,strategy=strategy,fit_range=fit_range,model=key,held_out_run=run)
                        folds.append(dict(**meta,training_runs=';'.join(training_runs),n_train=int(train.sum()),n_test=int(test.sum()),n_features=len(columns)))
                        sc=fit.named_steps['standardscaler'];lda=fit.named_steps['lineardiscriminantanalysis']
                        raw=lda.coef_[0]/sc.scale_;intercept=float(lda.intercept_[0]-raw@sc.mean_)
                        if not np.allclose(X[test]@raw+intercept,score,rtol=1e-7,atol=1e-7):raise ValueError('Coefficient reconstruction failed')
                        for k,beta in zip(columns,raw):coefficients.append(dict(**meta,feature=k,raw_coefficient=float(beta),intercept=intercept))
                        for row,z in zip(part.loc[test].to_dict('records'),score):
                            predictions.append(dict(**meta,run=run,condition=row['condition'],cycle=int(row.get('cycle',0)),range=row['range'],
                                                    frequency_hz=row['frequency_hz'],waveform=row['waveform'],true_class=int(row['true_class']),score=float(z)))
    p=pd.DataFrame(predictions);metrics=[]
    keys=['window_cycles','strategy','fit_range','model']
    if p.duplicated(keys+['run','condition','cycle']).any():raise ValueError('Duplicate held-out prediction')
    for values,group in p.groupby(keys,sort=False):
        meta=dict(zip(keys,values));subgroups=[('all',group)]+([(rg,group[group['range']==rg]) for rg in f.RANGES] if meta['strategy']=='pooled' else [])
        for rg,g in subgroups:
            for threshold,offset in [('standard',0.),('sensitivity_oriented',float(np.log(3)))]:
                metrics.append(dict(**meta,evaluated_range=rg,threshold=threshold,score_offset=offset,**counts(g.true_class,g.score,offset)))
    for name,frame in [('loocv_predictions',p),('loocv_metrics',pd.DataFrame(metrics)),('fold_inventory',pd.DataFrame(folds)),
                       ('lda_coefficients',pd.DataFrame(coefficients))]:
        frame.to_csv(out/(name+'.csv'),index=False)
    return p,pd.DataFrame(metrics)


def describe(conditions,out):
    features=list(f.definitions().feature);corr=out/'correlations';corr.mkdir(parents=True,exist_ok=True);rows=[]
    for rg,part in [('all',conditions)]+[(rg,conditions[conditions['range']==rg]) for rg in f.RANGES]:
        for cls,name in [(None,'pooled'),(0,'nominal'),(1,'degraded')]:
            q=part if cls is None else part[part.true_class==cls]
            q[features].corr().to_csv(corr/(rg+'_'+name+'.csv'))
            if cls is not None:
                for feature in features:
                    v=q[feature];rows.append(dict(range=rg,true_class=cls,feature=feature,n=len(v),minimum=v.min(),q1=v.quantile(.25),median=v.median(),q3=v.quantile(.75),maximum=v.max()))
    pd.DataFrame(rows).to_csv(out/'parameter_distributions.csv',index=False)
    adjusted=conditions[features]-conditions.groupby('condition')[features].transform('mean')
    adjusted.corr().to_csv(corr/'condition_centred.csv')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'runs.json')
    ap.add_argument('--data-dir',type=Path);ap.add_argument('--features-csv',type=Path);ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs')
    ap.add_argument('--skip-figures',action='store_true');a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    cycles=f.extract(a.manifest,out,a.data_dir) if a.features_csv is None else pd.read_csv(a.features_csv)
    if not np.isfinite(cycles[list(f.definitions().feature)]).all().all():raise ValueError('Nonfinite features')
    cycles.to_csv(out/'cycle_features.csv',index=False);conditions=f.aggregate(cycles);conditions.to_csv(out/'condition_features.csv',index=False)
    f.definitions().to_csv(out/'feature_definitions.csv',index=False);describe(conditions,out);pred,metrics=evaluate(cycles,conditions,out)
    meta=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
              n_runs=cycles.run.nunique(),n_cycles=len(cycles),n_conditions=len(conditions),n_features=len(f.definitions()),models=models(),
              native_samples=True,aggregation='mean of three per-cycle features; average absolute gain differences after fitting',
              rc_phase_origin='logged cycle start; no phase-origin optimisation',thresholds={'standard':0.,'sensitivity_oriented':-float(np.log(3))})
    (out/'analysis_metadata.json').write_text(json.dumps(meta,indent=2))
    print(metrics[(metrics.window_cycles==3)&(metrics.strategy=='pooled')&(metrics.evaluated_range=='all')&(metrics.threshold=='standard')&metrics.model.isin(MAIN)][['model','tp','fn','fp','tn']].to_string(index=False),flush=True)
    if not a.skip_figures:
        import plots
        plots.run(out)

if __name__=='__main__':main()
