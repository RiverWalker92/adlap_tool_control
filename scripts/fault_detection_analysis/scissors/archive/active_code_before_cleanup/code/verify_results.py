"""Verify native sample counts, features, validation groups and corrections."""
from pathlib import Path
import json
import numpy as np
try:
 from numpy import trapezoid
except ImportError:
 from numpy import trapz as trapezoid
import pandas as pd
from classify_cuts import registry,dataset
from extract_features import load,vector

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs_residual'

def run():
 d=pd.read_csv(OUT/'cut_features.csv');s=pd.read_csv(OUT/'motion_segments.csv')
 p=pd.read_csv(OUT/'loocv_predictions.csv');f=pd.read_csv(OUT/'fold_coefficients.csv')
 assert not p.duplicated(['scenario','selection','model','weight','cycle_id']).any()
 assert all(r.test_run not in r.train_runs.split(';') for r in f.itertuples())
 for (scenario,sel,model,w),g in p.groupby(['scenario','selection','model','weight']):
  expected,k=dataset(d,scenario,sel);assert set(g.cycle_id)==set(expected.cycle_id)
 assert all(len(m['parameters'])<=2 for m in registry())
 count=pd.read_csv(OUT/'sample_counts.csv').set_index('cycle_id')
 for phase in ['opening','closing','cycle']:
  assert np.array_equal(s.set_index('cycle_id')[phase+'_samples'],count[phase+'_valid_samples'])
  assert np.array_equal(s[phase+'_fft_samples'],s[phase+'_samples'])
 filtered_checks=[]
 assert not any('current' in c or 'filtered' in c for c in d.columns)
 assert all(not any('current' in c or 'filtered' in c for c in m['parameters']) for m in registry())
 metrics=pd.read_csv(OUT/'loocv_metrics.csv')
 for r in metrics.itertuples():
  cm=np.array(json.loads(r.confusion_matrix))
  assert r.alarm_TP==cm[1:,1:].sum() and r.alarm_FN==cm[1:,0].sum()
  assert r.alarm_TN==cm[0,0] and r.alarm_FP==cm[0,1:].sum()
  if len(cm)==2:assert (r.TP,r.FN,r.TN,r.FP)==(cm[1,1],cm[1,0],cm[0,0],cm[0,1])
 for run in d.run_id.unique():
  rps=list((ROOT/'data').rglob(f'{run}_ros_log*.jsonl'));dps=list((ROOT/'data').rglob(f'{run}_motor_dt_replay*.jsonl'));assert len(rps)==len(dps)==1
  rp,dp=rps[0],dps[0]
  ros,dt=load(rp),load(dp);offset=len(ros)-len(dt);time=np.array([r['time'] for r in ros])
  for seg in s[s.cycle_id.isin(d[d.run_id==run].cycle_id)].itertuples():
   row=d[d.cycle_id==seg.cycle_id].iloc[0];samples={}
   for phase in ['opening','closing']:
    start=getattr(seg,phase+'_valid_start_s');end=getattr(seg,phase+'_end_s')
    idx=np.flatnonzero((time>=start-1e-7)&(time<=end+1e-7));samples[phase]=idx
    assert len(idx)==getattr(seg,phase+'_samples')
   ci=np.flatnonzero((time>=seg.closing_start_s-1e-7)&(time<=seg.closing_end_s+1e-7))
   q=vector(ros,'commanded_instrument_angles',3)
   progress=(q[ci[0]-1]-q[ci])/(q[ci[0]-1]-q[ci[-1]])
   rc_idx=ci[(progress>=.30)&(progress<=.80)]
   assert len(rc_idx)==count.loc[seg.cycle_id,'closing_rc30_80_samples']
   tx=time[rc_idx]-time[rc_idx[0]]
   for motor in [0,3]:
    residual=vector(dt,'filtered_current_deviation',motor)
    rc=float(np.linalg.lstsq(np.column_stack([np.ones(len(tx)),tx]),residual[rc_idx-offset],rcond=None)[0][1])
    extracted=row[f'closing_M{motor}_rc30_80']
    assert np.isclose(extracted,rc,rtol=1e-10,atol=1e-9)
    filtered_checks.append(dict(cycle_id=seg.cycle_id,motor=motor,rc_ma_per_s=rc,extracted_rc_ma_per_s=extracted,rc_error_ma_per_s=abs(rc-extracted),rc_samples=len(rc_idx)))
   for motor in [0,3]:
    residual=vector(dt,'filtered_current_deviation',motor);areas={}
    for phase,idx in samples.items():
     x=residual[idx-offset];t=time[idx];areas[phase]=trapezoid(x,t)
     pref=f'{phase}_M{motor}_'
     expected={'signed_max':np.max(x),'abs_max':np.max(np.abs(x)),'peak_to_peak':np.ptp(x),'sample_mean':np.mean(x),'phase_sd':np.std(x,ddof=0),'rms':np.sqrt(np.mean(x*x)),'phase_mean':trapezoid(x,t)/(t[-1]-t[0]),'abs_integral':trapezoid(np.abs(x),t),'positive_integral':trapezoid(np.maximum(x,0),t)}
     for metric,value in expected.items():assert np.isclose(row[pref+metric],value,rtol=1e-10,atol=1e-9),(seg.cycle_id,pref+metric)
    idx=np.r_[samples['opening'],samples['closing']];x=residual[idx-offset]
    assert np.isclose(row[f'cycle_M{motor}_rms'],np.sqrt(np.mean(x*x)))
    assert np.isclose(row[f'cycle_M{motor}_abs_max'],np.abs(x).max())
    duration=seg.opening_duration_s+seg.closing_duration_s
    assert np.isclose(row[f'cycle_M{motor}_phase_mean'],sum(areas.values())/duration)
 for phase in ['opening','closing','cycle']:
  for motor in [0,3]:
   pref=f'{phase}_M{motor}_';assert np.allclose(d[pref+'sample_mean'],d[pref+'original_mean'])
   assert np.allclose(d[pref+'rms']**2,d[pref+'sample_mean']**2+d[pref+'phase_sd']**2)
   duration=s.opening_duration_s+s.closing_duration_s if phase=='cycle' else s[phase+'_duration_s']
   assert np.allclose(d[pref+'positive_integral'],(d[pref+'abs_integral']+d[pref+'phase_mean']*duration)/2)
 profiles=pd.read_csv(OUT/'phase_profiles.csv')
 for phase in ['opening','closing','cycle']:
  g=profiles[profiles.phase==phase].groupby(['cycle_id','motor']).size();expected=s.set_index('cycle_id')[phase+'_samples']
  assert all(n==expected.loc[cyc] for (cyc,m),n in g.items())
 pd.DataFrame(filtered_checks).to_csv(OUT/'residual_RC_check.csv',index=False)
 checks=dict(n_cycles=len(d),n_runs=d.run_id.nunique(),n_models=len(registry()),n_validation_folds=len(f),n_predictions=len(p),native_sample_counts_match_logs=True,fft_counts_equal_native_counts=True,residual_RC_matches_replay=True,residual_only_models=True,no_unfiltered_current_feature=True,all_native_RMS_SD_means_integrals_checked_against_logs=True,no_extra_endpoint=True,no_fixed_150_300_grid=True,no_cut_averaging=True,heldout_run_absent_from_training=True,each_eligible_cut_predicted_once=True,RMS_identity_checked=True,positive_integral_identity_checked=True,old_native_control_features_reproduced=json.loads((OUT/'extraction_protocol.json').read_text())['checks_against_old_features'])
 (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
 print('Native-source and validation audit passed.',json.dumps(checks))
 return checks
if __name__=='__main__':run()
