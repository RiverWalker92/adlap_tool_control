#!/usr/bin/env python3
"""Direct, interpretable one-/two-parameter LDA; leave complete run out.

Fixed feature models, auto covariance shrinkage, equal priors. No PCA or
p-value selection. Per cut, compare all / first / second+third selections.
"""
from pathlib import Path
import argparse,json,warnings
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from extract_features import METRICS,NAMES

ROOT=Path(__file__).resolve().parents[1]
SELECTIONS={'all':[1,2,3],'first':[1],'later':[2,3]}
SCENARIOS={
 'healthy_all':([0,1,2,3,4,6,8],2,['Healthy (0)','After filing (1,2,3,4,6,8)']),
 'healthy_blunt':([0,8],2,['Healthy (0)','Blunt (8)']),
 'healthy_medium146':([0,1,4,6],2,['Healthy (0)','Medium (1,4,6)']),
 'medium146_blunt':([1,4,6,8],2,['Medium (1,4,6)','Blunt (8)']),
 'three146':([0,1,4,6,8],3,['Healthy (0)','Medium (1,4,6)','Blunt (8)']),
 'three_all':([0,1,2,3,4,6,8],3,['Healthy (0)','Medium (1,2,3,4,6)','Blunt (8)']),
 'three4':([0,4,8],3,['Healthy (0)','Medium (4)','Blunt (8)'])}

def registry():
 rows=[]
 def add(name,cols,label,category):rows.append(dict(model=name,parameters=cols,label=label,category=category))
 for phase in ['closing','opening','cycle']:
  for m in [3,0]:
   for stat in METRICS:add(f'{phase}_M{m}_{stat}',[f'{phase}_M{m}_{stat}'],f'{phase.capitalize()} M{m} {NAMES[stat]}','residual_single')
  for stat in ['wave_corr','ellipse_area']:add(f'{phase}_joint_{stat}',[f'{phase}_joint_{stat}'],f'{phase.capitalize()} {NAMES[stat]}','joint_single')
 for motor in [0,3]:add(f'closing_M{motor}_rc30_80',[f'closing_M{motor}_rc30_80'],f'M{motor} closing RC 30-80%','residual_single')
 for phase in ['closing','opening','cycle']:
  for stat in METRICS:add(f'{phase}_paired_{stat}',[f'{phase}_M0_{stat}',f'{phase}_M3_{stat}'],f'{phase.capitalize()} M0 + M3 {NAMES[stat]}','motor_pair')
 for stat in METRICS:
  add(f'directions_M3_{stat}',[f'closing_M3_{stat}',f'opening_M3_{stat}'],f'M3 closing + opening {NAMES[stat]}','direction_pair')
 add('closing_paired_rc30_80',['closing_M0_rc30_80','closing_M3_rc30_80'],'M0 + M3 closing RC 30-80%','motor_pair')
 add('original_closing_mean',['closing_M3_original_mean'],'Original-sample closing mean','legacy')
 add('original_opening_mean',['opening_M3_original_mean'],'Original-sample opening mean','legacy')
 add('original_cycle_mean',['cycle_M3_original_mean'],'Original-sample cycle mean','legacy')
 add('original_directions_mean',['closing_M3_original_mean','opening_M3_original_mean'],'Original-sample closing + opening mean','legacy_pair')
 add('closing_peak_rc',['closing_M3_abs_max','closing_M3_rc30_80'],'M3 closing absolute maximum + RC 30-80%','interpretable_pair')
 add('closing_peak_mean',['closing_M3_abs_max','closing_M3_phase_mean'],'M3 closing absolute maximum + mean','interpretable_pair')
 add('closing_peak_rms',['closing_M3_abs_max','closing_M3_rms'],'M3 closing absolute maximum + RMS','interpretable_pair')
 add('closing_rc_rms',['closing_M3_rc30_80','closing_M3_rms'],'M3 closing RC 30-80% + RMS','interpretable_pair')
 add('closing_mean_sd',['closing_M3_phase_mean','closing_M3_phase_sd'],'Closing mean + SD','interpretable_pair')
 add('closing_mean_rms',['closing_M3_phase_mean','closing_M3_rms'],'Closing mean + RMS','interpretable_pair')
 add('cycle_mean_sd',['cycle_M3_phase_mean','cycle_M3_phase_sd'],'Cycle mean + SD','interpretable_pair')
 add('cycle_mean_rms',['cycle_M3_phase_mean','cycle_M3_rms'],'Cycle mean + RMS','interpretable_pair')
 assert all('current' not in c and 'filtered' not in c for r in rows for c in r['parameters'])
 assert len({r['model'] for r in rows})==len(rows) and all(1<=len(r['parameters'])<=2 for r in rows)
 return rows
def dataset(df,s,sel):
 levels,k,labels=SCENARIOS[s];d=df[df.iteration.isin(levels)&df.cut.isin(SELECTIONS[sel])].copy()
 if k==3:d['truth']=np.where(d.iteration==0,0,np.where(d.iteration==8,2,1))
 elif s=='medium146_blunt':d['truth']=(d.iteration==8).astype(int)
 else:d['truth']=(d.iteration!=0).astype(int)
 return d.reset_index(drop=True),k
def fit(tr,cols,k):
 sc=StandardScaler().fit(tr[cols].to_numpy());x=sc.transform(tr[cols].to_numpy())
 ld=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.full(k,1/k)).fit(x,tr.truth)
 coef=ld.coef_/sc.scale_;b=ld.intercept_-coef@sc.mean_
 return sc,ld,coef,b
def decision(fit,test,cols,k):
 sc,ld,co,b=fit;score=test[cols].to_numpy()@co.T+b;yp=(score[:,0]>0).astype(int) if k==2 else score.argmax(1)
 assert np.array_equal(yp,ld.predict(sc.transform(test[cols].to_numpy())))
 return score,yp
def decision_counts(cm,k,scenario):
 counts=dict(TP=int(cm[1,1]) if k==2 else np.nan,
             FN=int(cm[1,0]) if k==2 else np.nan,
             TN=int(cm[0,0]) if k==2 else np.nan,
             FP=int(cm[0,1]) if k==2 else np.nan)
 # Three-class fault alarms collapse intermediate and blunt into positive.
 # For medium-vs-blunt, positive is blunt, reference is medium (not healthy).
 counts.update(alarm_TP=int(cm[1:,1:].sum()),alarm_FN=int(cm[1:,0].sum()),
               alarm_TN=int(cm[0,0]),alarm_FP=int(cm[0,1:].sum()),
               reference_label=SCENARIOS[scenario][2][0],
               positive_labels=';'.join(SCENARIOS[scenario][2][1:]))
 return counts

def evaluate(df,out,models,scenarios=None):
 warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
 warnings.filterwarnings('ignore',message='Only one sample available.*',category=RuntimeWarning)
 scenarios=scenarios or list(SCENARIOS);predictions=[];folds=[];exports=[]
 for s in scenarios:
  for sel in SELECTIONS:
   d,k=dataset(df,s,sel)
   for model in models:
    name,cols=model['model'],model['parameters']
    if not np.isfinite(d[cols].to_numpy()).all():
     raise ValueError('Undefined requested parameter values: '+name)
    full=fit(d,cols,k);score,yp=decision(full,d,cols,k)
    exports.append(dict(scenario=s,selection=sel,model=name,parameters=cols,coefficients=full[2].tolist(),intercept=full[3].tolist(),training_correct=int((yp==d.truth).sum()),n_train=len(d),class_labels=SCENARIOS[s][2],positive_rule='score>0 (standard); score+ln(3)>0 (fixed sensitive reference)' if k==2 else 'argmax of three discriminant scores',scalar_boundary=float(-full[3][0]/full[2][0,0]) if k==2 and len(cols)==1 and abs(full[2][0,0])>1e-14 else None))
    for held in d.run_id.unique():
     tr=d[d.run_id!=held];te=d[d.run_id==held];assert not set(tr.run_id)&set(te.run_id)
     f=fit(tr,cols,k);scores,_=decision(f,te,cols,k)
     folds.append(dict(scenario=s,selection=sel,model=name,test_run=held,train_runs=';'.join(tr.run_id.unique()),n_train_cycles=len(tr),n_test_cycles=len(te),n_train_class0=int((tr.truth==0).sum()),n_train_last_class=int((tr.truth==k-1).sum()),parameter1=cols[0],parameter2=cols[1] if len(cols)>1 else '',coefficients=json.dumps(f[2].tolist()),intercept=json.dumps(f[3].tolist())))
     for weight in ([1,3] if k==2 else [1]):
      yp=(scores[:,0]+np.log(weight)>0).astype(int) if k==2 else scores.argmax(1)
      for j,(_,r) in enumerate(te.iterrows()):predictions.append(dict(scenario=s,selection=sel,model=name,weight=weight,cycle_id=r.cycle_id,run_id=r.run_id,iteration=int(r.iteration),run_number=int(r.run_number),cut=int(r.cut),truth=int(r.truth),prediction=int(yp[j]),score=float(scores[j,0]) if k==2 else np.nan))
  print('Completed',s,'models',len(models),'predictions so far',len(predictions),flush=True)
 p=pd.DataFrame(predictions);results=[]
 for keys,g in p.groupby(['scenario','selection','model','weight'],sort=False):
  s,sel,name,w=keys;k=SCENARIOS[s][1];cm=confusion_matrix(g.truth,g.prediction,labels=range(k));ns=cm.sum(1);rec=np.diag(cm)/ns
  row=dict(scenario=s,selection=sel,model=name,weight=w,n=len(g),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/len(g)),balanced_accuracy=float(rec.mean()),confusion_matrix=json.dumps(cm.tolist()),positive_correct=int(cm[1,1]) if k==2 else np.nan,positive_n=int(ns[1]) if k==2 else np.nan,sensitivity=float(rec[1]) if k==2 else np.nan,specificity=float(rec[0]),alarm_faulty_correct=int(cm[1:,1:].sum()),alarm_faulty_n=int(ns[1:].sum()),missed_as_reference=int(cm[1:,0].sum()))
  row.update(decision_counts(cm,k,s))
  for cl in range(k):row[f'class{cl}_correct']=int(cm[cl,cl]);row[f'class{cl}_n']=int(ns[cl])
  results.append(row)
 m=pd.DataFrame(results);p.to_csv(out/'loocv_predictions.csv',index=False);m.to_csv(out/'loocv_metrics.csv',index=False);pd.DataFrame(folds).to_csv(out/'fold_coefficients.csv',index=False)
 (out/'fitted_models.json').write_text(json.dumps(exports,indent=2));(out/'model_registry.json').write_text(json.dumps(models,indent=2))
 pd.DataFrame([{**r,'parameters':';'.join(r['parameters'])} for r in models]).to_csv(out/'model_registry.csv',index=False)
 strat=[]
 for keys,g in p[p.selection=='all'].groupby(['scenario','model','weight'],sort=False):
  s,name,w=keys;k=SCENARIOS[s][1]
  for label,cuts in [('first',[1]),('later',[2,3])]:
   gg=g[g.cut.isin(cuts)];cm=confusion_matrix(gg.truth,gg.prediction,labels=range(k));ns=cm.sum(1)
   strat.append(dict(scenario=s,model=name,weight=w,train_selection='all',test_selection=label,n=len(gg),correct=int(np.trace(cm)),balanced_accuracy=float(np.mean(np.diag(cm)/ns)),reference_correct=int(cm[0,0]),reference_n=int(ns[0]),alarm_faulty_correct=int(cm[1:,1:].sum()),alarm_faulty_n=int(ns[1:].sum())))
 pd.DataFrame(strat).to_csv(out/'cut_order_validation.csv',index=False)
 print('Total models',len(models),'folds',len(folds),'predictions',len(p),'result rows',len(m),flush=True)
 return p,m
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'outputs_residual/cut_features.csv');ap.add_argument('--output',type=Path,default=ROOT/'outputs_residual');ap.add_argument('--scenarios',nargs='*');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);evaluate(pd.read_csv(a.input),a.output,registry(),a.scenarios)
