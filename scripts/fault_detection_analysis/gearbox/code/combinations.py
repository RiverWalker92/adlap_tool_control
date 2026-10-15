"""All 24 singles and 276 pairs using the existing gearbox validation folds.

Does not alter extracted features, original results, thresholds or DT models.
The chosen winners are exploratory; candidate selection is not nested.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
from pathlib import Path
from itertools import combinations
import argparse,json
import numpy as np
import pandas as pd
from analyse import COLUMNS,NAMES,model_definitions,fit_classifier
from pairwise_lda import PairwiseLDA

def rank(frame):
    return frame.sort_values(['FN','FP','balanced_accuracy','n_features','model'],ascending=[True,True,False,True,True],kind='stable')

def name(column):
    if column=='shape_corr':return 'M0-M3 waveform correlation'
    if column=='ellipse_area':return 'M0-M3 ellipse area'
    motor,parameter=column.split('_',1)
    return motor+' '+NAMES[parameter].lower()

def run(output):
    output=Path(output);out=output/'combinations';out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(output/'features.csv')
    pairs=[(i,-1) for i in range(len(COLUMNS))]+list(combinations(range(len(COLUMNS)),2))
    registry=[]
    for index,(i,j) in enumerate(pairs):
        a=COLUMNS[i];b=COLUMNS[j] if j>=0 else ''
        registry.append(dict(model=f'G{index:03d}',feature1=a,feature2=b,input1=name(a),input2=name(b) if b else '',
            n_features=1+(j>=0),mixed_parameter=bool(j>=0 and a.split('_',1)[-1]!=b.split('_',1)[-1])))
    registry=pd.DataFrame(registry);registry.to_csv(out/'model_registry.csv',index=False)
    first=np.array([i for i,j in pairs]);second=np.array([j for i,j in pairs])
    arrays={};predictions=[];summaries=[];folds=[]
    for length in [5,10]:
        d=data[data.n==length].reset_index(drop=True)
        X=d[COLUMNS].to_numpy();y=d.label.to_numpy(dtype=int)
        totals=np.zeros((len(pairs),2,2),dtype=int)
        for healthy in d[d.label==0].run.unique():
            for _,held in d[d.label==1].iterrows():
                overlap=(d.label==1)&(d.start<held.end)&(d.end>held.start)
                train=((d.run!=healthy)&~overlap).to_numpy()
                test=((d.run==healthy)|((d.label==1)&(d.window==held.window))).to_numpy()
                assert healthy not in d.loc[train,'run'].values
                assert not ((d.loc[train,'label']==1)&(d.loc[train,'start']<held.end)&(d.loc[train,'end']>held.start)).any()
                fitted=PairwiseLDA(first,second).fit(X[train],y[train],2)
                score=fitted.scores(X[test]);decision=score[:,:,1]-score[:,:,0];yp=(decision>0).astype(int)
                for a in range(2):
                    for b in range(2):totals[:,a,b]+=((y[test,None]==a)&(yp==b)).sum(0)
                meta=dict(n=length,held_nominal=healthy,held_degraded=held.window)
                folds.append(dict(**meta,training_observations=';'.join(d.loc[train,'run']+':'+d.loc[train,'window']),
                                  test_observations=';'.join(d.loc[test,'run']+':'+d.loc[test,'window'])))
                for pos,(_,obs) in enumerate(d[test].iterrows()):
                    for index,r in registry.iterrows():
                        predictions.append(dict(**meta,model=r.model,run=obs.run,window=obs.window,
                                                label=int(obs.label),score=float(decision[pos,index]),predicted=int(yp[pos,index])))
        for index,r in registry.iterrows():
            tn,fp,fn,tp=map(int,totals[index].ravel())
            summaries.append(dict(**r.to_dict(),n=length,TP=tp,FN=fn,TN=tn,FP=fp,
                balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp)))
        full=PairwiseLDA(first,second).fit(X,y,2)
        arrays[f'n{length}_coef']=full.coef;arrays[f'n{length}_intercept']=full.intercept
        print('Completed combinations:',length,'rotations',flush=True)
    metrics=pd.DataFrame(summaries);pred=pd.DataFrame(predictions)
    metrics.to_csv(out/'cv_metrics.csv',index=False);pred.to_csv(out/'cv_predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_manifest.csv',index=False)
    pd.concat([rank(g).head(20) for _,g in metrics.groupby('n')]).to_csv(out/'top_models.csv',index=False)
    np.savez_compressed(out/'full_models.npz',**arrays)
    # Descriptive full-run outcomes: fit all five-rotation blocks, apply to all30.
    d=data[data.n==5];fitted=PairwiseLDA(first,second).fit(d[COLUMNS].to_numpy(),d.label.to_numpy(dtype=int),2)
    full=data[data.n==30];scores=fitted.scores(full[COLUMNS].to_numpy());yp=scores[:,:,1]>scores[:,:,0]
    desc=[]
    for i,r in registry.iterrows():
        y=full.label.to_numpy();desc.append(dict(model=r.model,TP=int(((y==1)&yp[:,i]).sum()),FN=int(((y==1)&~yp[:,i]).sum()),
        TN=int(((y==0)&~yp[:,i]).sum()),FP=int(((y==0)&yp[:,i]).sum())))
    pd.DataFrame(desc).to_csv(out/'descriptive_30_counts.csv',index=False)
    # Reproduce all original candidate predictions with matching validation keys.
    old=pd.read_csv(output/'cv_predictions.csv');matched=0
    for old_name,definition in model_definitions().items():
        cols=definition['features'];q=registry[registry.apply(lambda r:set([r.feature1]+([r.feature2] if r.feature2 else []))==set(cols),axis=1)]
        assert len(q)==1
        a=old[old.model==old_name];b=pred[pred.model==q.iloc[0].model]
        joined=a.merge(b,on=['n','held_nominal','held_degraded','run','window','label'],suffixes=('_old','_new'),validate='one_to_one')
        assert len(joined)==len(a)==len(b)
        assert np.array_equal(joined.predicted_old,joined.predicted_new),old_name
        matched+=len(joined)
    # Reference sklearn checks on all full-data candidates and leaders in folds.
    checked=0
    for length in [5,10]:
        d=data[data.n==length].reset_index(drop=True);X=d[COLUMNS].to_numpy();y=d.label.to_numpy(dtype=int)
        fitted=PairwiseLDA(first,second).fit(X,y,2)
        for idx,r in registry.iterrows():
            cols=[r.feature1]+([r.feature2] if r.feature2 else [])
            reference=fit_classifier(d,{'features':cols});sc=reference['scale'];lda=reference['lda']
            co=lda.coef_[0]/sc.scale_;bias=lda.intercept_[0]-co@sc.mean_
            actual_co,actual_b=fitted.coefficients(idx)
            assert np.allclose(co,actual_co[0],rtol=1e-6,atol=1e-7),(length,r.model,'coef')
            assert np.allclose(bias,actual_b[0],rtol=1e-6,atol=1e-7),(length,r.model,'bias')
        leaders=list(rank(metrics[metrics.n==length]).head(8).model)
        leaders+=list(rank(metrics[(metrics.n==length)&metrics.mixed_parameter]).head(8).model)
        leaders+=['G005','G016','G139','G075','G083','G102','G140']
        for model in dict.fromkeys(leaders):
            r=registry[registry.model==model].iloc[0];cols=[r.feature1]+([r.feature2] if r.feature2 else [])
            for healthy in d[d.label==0].run.unique():
                for _,held in d[d.label==1].iterrows():
                    overlap=(d.label==1)&(d.start<held.end)&(d.end>held.start)
                    train=d[(d.run!=healthy)&~overlap]
                    test=d[(d.run==healthy)|((d.label==1)&(d.window==held.window))]
                    ref=fit_classifier(train,{'features':cols});yp=(ref.decision_function(test[cols])>0).astype(int)
                    actual=pred[(pred.n==length)&(pred.model==model)&(pred.held_nominal==healthy)&(pred.held_degraded==held.window)]
                    joined=test[['run','window']].merge(actual[['run','window','predicted']],on=['run','window'],validate='one_to_one')
                    assert np.array_equal(yp,joined.predicted),(length,model,healthy,held.window)
                    checked+=len(yp)
    checks=dict(n_features=24,n_single_candidates=24,n_pair_candidates=276,n_candidates=300,
        original_predictions_matched=matched,full_data_reference_fits=600,new_fold_predictions_checked=checked,
        training_only_standardisation=True,pair_specific_shrinkage=True,same_original_folds=True,
        standard_threshold=0,selection_is_exploratory=True,
        limitation='One degraded run; repeated dependent test decisions. No nested feature selection. Thirty-rotation counts are descriptive.')
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    return metrics


def export_tables(output):
    output=Path(output);out=output/'combinations';dest=output/'tables';dest.mkdir(exist_ok=True)
    reg=pd.read_csv(out/'model_registry.csv').fillna('')
    cv=pd.read_csv(out/'cv_metrics.csv');desc=pd.read_csv(out/'descriptive_30_counts.csv')
    selected=['G005','G016','G139','G075','G083','G140','G102']
    rows=[]
    for model in selected:
        row=reg[reg.model==model].iloc[0].to_dict()
        for key in ['input1','input2']:
            row[key]=row[key].replace(' rms',' RMS').replace('phase sd','phase SD')
        for n in [30,10,5]:
            q=desc[desc.model==model].iloc[0] if n==30 else cv[(cv.model==model)&(cv.n==n)].iloc[0]
            for c in ['TP','FN','TN','FP']:row[f'{n}_{c}']=int(q[c])
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest/'selected_lda_counts.csv',index=False)
    five=cv[cv.n==5];ten=cv[cv.n==10]
    both=five.merge(ten,on='model',suffixes=('_5','_10'))
    both=both[(both.FN_5==0)&(both.FP_5==0)&(both.FN_10==0)&(both.FP_10==0)]
    both.to_csv(dest/'perfect_pairs_both_lengths.csv',index=False)
