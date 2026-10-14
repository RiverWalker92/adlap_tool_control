"""Exhaustive residual-only one/two-feature LDA, with run-wise validation.

Outputs are exploratory per-candidate CV results, not independent evidence
for a winner selected on these same results. A nested selection check is optional.
"""
from pathlib import Path
from itertools import combinations
import argparse,json
import numpy as np
import pandas as pd
from classify_cuts import SCENARIOS,SELECTIONS,dataset
from pairwise_lda import PairwiseLDA

ROOT=Path(__file__).resolve().parents[1]

def candidates(definitions):
    columns=definitions.column.tolist()
    pairs=[(i,-1) for i in range(len(columns))]+list(combinations(range(len(columns)),2))
    rows=[]
    for n,(i,j) in enumerate(pairs):
        a=definitions.iloc[i];b=definitions.iloc[j] if j>=0 else None
        rows.append(dict(model=f'F{n:04d}',feature1=columns[i],feature2=columns[j] if j>=0 else '',
                         input1=f'{"M"+str(a.motor) if str(a.motor) in ["0","3"] else "M0–M3"} {a.phase} {a["name"]}',
                         input2=f'{"M"+str(b.motor) if str(b.motor) in ["0","3"] else "M0–M3"} {b.phase} {b["name"]}' if j>=0 else '',
                         n_features=1 if j<0 else 2,
                         motor1=str(a.motor),motor2=str(b.motor) if j>=0 else '',
                         phase1=a.phase,phase2=b.phase if j>=0 else '',
                         mixed_parameter=(columns[i].split('_',2)[2]!=columns[j].split('_',2)[2]) if j>=0 else False))
    return columns,np.array([a for a,b in pairs]),np.array([b for a,b in pairs]),pd.DataFrame(rows)

def counts(y,pred,k):
    m=pred.shape[1];cm=np.zeros((m,k,k),dtype=np.int16)
    for a in range(k):
        for b in range(k):cm[:,a,b]=((y[:,None]==a)&(pred==b)).sum(0)
    return cm

def metrics(cm):
    k=cm.shape[1]
    n=cm.sum((1,2));correct=np.trace(cm,axis1=1,axis2=2)
    totals=cm.sum(2)
    rec=np.divide(np.diagonal(cm,axis1=1,axis2=2),totals,out=np.full_like(totals,np.nan,dtype=float),where=totals>0)
    result=dict(n=n,correct=correct,accuracy=correct/n,balanced_accuracy=rec.mean(1),
                alarm_TP=cm[:,1:,1:].sum((1,2)),alarm_FN=cm[:,1:,0].sum(1),
                alarm_TN=cm[:,0,0],alarm_FP=cm[:,0,1:].sum(1),
                specificity=rec[:,0])
    if k==2:result.update(TP=cm[:,1,1],FN=cm[:,1,0],TN=cm[:,0,0],FP=cm[:,0,1],sensitivity=rec[:,1])
    for a in range(k):result[f'class{a}_correct']=cm[:,a,a];result[f'class{a}_n']=totals[:,a]
    return result

def ranking(frame,severity=False):
    if severity:
        keys=['balanced_accuracy','alarm_FN','alarm_FP','n_features','model'];asc=[False,True,True,True,True]
    else:
        keys=['alarm_FN','alarm_FP','balanced_accuracy','n_features','model'];asc=[True,True,False,True,True]
    return frame.sort_values(keys,ascending=asc,kind='stable')

def run(root=ROOT,scenarios=None,nested=False):
    root=Path(root);base=root/'outputs';out=base/'combinations';out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(base/'cut_features.csv');definitions=pd.read_csv(base/'feature_definitions.csv')
    assert not definitions.column.str.contains('current|filtered').any()
    columns,i,j,registry=candidates(definitions)
    assert len(columns)==68 and len(registry)==2346
    registry.to_csv(out/'model_registry.csv',index=False)
    lookup={tuple(sorted([r.feature1]+([r.feature2] if r.n_features==2 else []))):n for n,r in enumerate(registry.itertuples())}
    old_registry=json.loads((base/'model_registry.json').read_text()) if (base/'model_registry.json').exists() else []
    old_predictions=pd.read_csv(base/'loocv_predictions.csv') if (base/'loocv_predictions.csv').exists() else None
    reference_models={r['model']:lookup[tuple(sorted(r['parameters']))] for r in old_registry if tuple(sorted(r['parameters'])) in lookup}
    result_frames=[];best=[];validation_rows=0;arrays={};full_arrays={};fold_arrays={};fold_rows=[]
    for scenario in scenarios or list(SCENARIOS):
        for selection in SELECTIONS:
            d,k=dataset(data,scenario,selection);X=d[columns].to_numpy();y=d.truth.to_numpy()
            key=f'{scenario}__{selection}'
            prediction=np.empty((len(d),len(registry)),dtype=np.int8)
            scores=np.empty((len(d),len(registry),k))
            fold_co=[];fold_b=[];runs=[]
            for held in d.run_id.unique():
                te=(d.run_id==held).to_numpy();tr=~te
                f=PairwiseLDA(i,j).fit(X[tr],y[tr],k)
                scores[te]=f.scores(X[te]);prediction[te]=scores[te].argmax(-1)
                fold_co.append(f.coef);fold_b.append(f.intercept);runs.append(held)
                fold_rows.append(dict(scenario=scenario,selection=selection,test_run=held,
                                      train_runs=';'.join(d.loc[tr,'run_id'].unique()),
                                      train_n=int(tr.sum()),test_n=int(te.sum())))
            if old_predictions is not None:
                for name,index in reference_models.items():
                    old=old_predictions[(old_predictions.scenario==scenario)&(old_predictions.selection==selection)&(old_predictions.model==name)&(old_predictions.weight==1)]
                    if len(old)==0:continue
                    aligned=d[['cycle_id']].merge(old[['cycle_id','prediction']],on='cycle_id',validate='one_to_one')
                    if not np.array_equal(aligned.prediction.to_numpy(),prediction[:,index]):
                        mismatch=d.loc[aligned.prediction.to_numpy()!=prediction[:,index],'cycle_id'].tolist()
                        raise AssertionError(f'Reference prediction mismatch: {key} {name} {mismatch}')
                    validation_rows+=len(old)
            cm=counts(y,prediction,k)
            table=registry.copy()
            for name,values in metrics(cm).items():table[name]=values
            table['scenario']=scenario;table['selection']=selection
            table['reference_label']=SCENARIOS[scenario][2][0]
            table['positive_labels']=';'.join(SCENARIOS[scenario][2][1:])
            # Confusion labels are exact classes; alarms collapse all non-reference classes.
            table['confusion_matrix']=[json.dumps(a.tolist()) for a in cm]
            result_frames.append(table)
            top=ranking(table)
            best.append(top.head(10).assign(ranking='fault_detection'))
            if k==3:best.append(ranking(table,severity=True).head(10).assign(ranking='severity'))
            arrays[key+'__prediction']=prediction;arrays[key+'__scores']=scores
            arrays[key+'__truth']=y;arrays[key+'__cycle_id']=d.cycle_id.to_numpy(dtype=str)
            arrays[key+'__run_id']=d.run_id.to_numpy(dtype=str)
            arrays[key+'__cut']=d.cut.to_numpy();arrays[key+'__confusion']=cm
            full=PairwiseLDA(i,j).fit(X,y,k)
            full_arrays[key+'__coef']=full.coef;full_arrays[key+'__intercept']=full.intercept
            fold_arrays[key+'__coef']=np.asarray(fold_co);fold_arrays[key+'__intercept']=np.asarray(fold_b)
            fold_arrays[key+'__test_run']=np.asarray(runs,dtype=str)
            print(f'{scenario} / {selection}: {len(registry)} candidates; best fault FN={int(top.iloc[0].alarm_FN)}, FP={int(top.iloc[0].alarm_FP)}',flush=True)
    results=pd.concat(result_frames,ignore_index=True)
    results.to_csv(out/'cv_metrics.csv',index=False)
    tops=pd.concat(best,ignore_index=True);tops.to_csv(out/'top_models.csv',index=False)
    np.savez_compressed(out/'cv_decisions.npz',**arrays)
    np.savez_compressed(out/'full_models.npz',**full_arrays)
    np.savez_compressed(out/'fold_models.npz',**fold_arrays)
    pd.DataFrame(fold_rows).to_csv(out/'fold_manifest.csv',index=False)
    # A small, readable prediction export for the reported candidates.
    selected=[]
    for (scenario,selection),g in tops.groupby(['scenario','selection'],sort=False):
        d,k=dataset(data,scenario,selection);key=f'{scenario}__{selection}'
        for name in g.model.unique():
            idx=int(registry.index[registry.model==name][0]);row=registry.iloc[idx]
            v=d[['cycle_id','run_id','iteration','run_number','cut','truth']].copy()
            v['prediction']=arrays[key+'__prediction'][:,idx];v['model']=name
            v['feature1']=row.feature1;v['feature2']=row.feature2;v['scenario']=scenario;v['selection']=selection
            selected.append(v)
    pd.concat(selected).to_csv(out/'selected_cv_predictions.csv',index=False)
    checks=dict(n_features=len(columns),single_models=len(columns),pair_models=len(registry)-len(columns),
                baseline_predictions_matched=validation_rows,reference_models_matched=len(reference_models),
                training_standardisation_only=True,pair_specific_Ledoit_Wolf=True,equal_priors=True,
                whole_runs_held_out=True,one_observation_per_cut=True,no_measured_current_features=True,
                best_model_selection_is_exploratory=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    if nested:
        nested_selection(data,columns,i,j,registry,out,scenarios)
    return results

def nested_selection(data,columns,i,j,registry,out,scenarios=None):
    """Outer runs are untouched during selection; skip inner folds emptying a class.

    With only two healthy runs, withholding one leaves a single healthy run.
    Its inner holdout is infeasible, so inner selection then lacks healthy test
    observations. Export this limitation for every fold rather than hiding it.
    """
    rows=[];folds=[]
    for scenario in scenarios or ['healthy_all']:
        for selection in SELECTIONS:
            d,k=dataset(data,scenario,selection);X=d[columns].to_numpy();y=d.truth.to_numpy()
            for held in d.run_id.unique():
                outer_test=(d.run_id==held).to_numpy();outer_train=~outer_test
                inner_true=[];inner_pred=[];used=[];skipped=[]
                for validation in d.loc[outer_train,'run_id'].unique():
                    test=outer_train&(d.run_id==validation).to_numpy();train=outer_train&~test
                    if len(np.unique(y[train]))!=k:
                        skipped.append(validation);continue
                    f=PairwiseLDA(i,j).fit(X[train],y[train],k)
                    inner_true.append(y[test]);inner_pred.append(f.predict(X[test]));used.append(validation)
                yi=np.concatenate(inner_true);pi=np.concatenate(inner_pred)
                cm=counts(yi,pi,k);score=metrics(cm)
                table=registry.copy()
                for name,values in score.items():table[name]=values
                # BA over classes represented in inner tests; exported as such.
                totals=cm.sum(2);present=totals[0]>0
                recalls=np.divide(np.diagonal(cm,axis1=1,axis2=2),totals,out=np.full_like(totals,np.nan,dtype=float),where=totals>0)
                table['balanced_accuracy']=np.nanmean(recalls,axis=1)
                choice=ranking(table).iloc[0];index=int(table.index[table.model==choice.model][0])
                f=PairwiseLDA(i,j).fit(X[outer_train],y[outer_train],k)
                yp=f.predict(X[outer_test])[:,index]
                for (_,observation),predicted in zip(d.loc[outer_test].iterrows(),yp):
                    rows.append(dict(scenario=scenario,selection=selection,cycle_id=observation.cycle_id,
                                     run_id=held,truth=int(observation.truth),prediction=int(predicted),
                                     selected_model=choice.model,feature1=choice.feature1,feature2=choice.feature2))
                folds.append(dict(scenario=scenario,selection=selection,outer_test_run=held,selected_model=choice.model,
                                  feature1=choice.feature1,feature2=choice.feature2,inner_test_runs=';'.join(used),
                                  skipped_inner_runs=';'.join(skipped),inner_test_class_counts=json.dumps(totals[0].tolist()),
                                  inner_healthy_test_available=bool(present[0]) if scenario!='medium146_blunt' else None))
            print('Nested selection:',scenario,selection,flush=True)
    predictions=pd.DataFrame(rows);predictions.to_csv(out/'nested_predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'nested_selection_folds.csv',index=False)
    summaries=[]
    for (scenario,selection),g in predictions.groupby(['scenario','selection'],sort=False):
        k=SCENARIOS[scenario][1];cm=counts(g.truth.to_numpy(),g.prediction.to_numpy()[:,None],k)
        summary={name:values[0] for name,values in metrics(cm).items()}
        summary.update(scenario=scenario,selection=selection,confusion_matrix=json.dumps(cm[0].tolist()))
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(out/'nested_metrics.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--scenarios',nargs='*');ap.add_argument('--nested',action='store_true')
    args=ap.parse_args();run(args.root,args.scenarios,args.nested)
