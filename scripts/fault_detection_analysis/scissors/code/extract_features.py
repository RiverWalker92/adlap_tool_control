#!/usr/bin/env python3
"""Audited single-cut profiles: valid opening, closing and active cycle.

No waveform or feature averaging across cuts. Fixed Motor DT predictions.
All observed samples per direction and active cycle; no fixed phase count.
Time integrals use endpoints and exclude the gap between valid directions.
"""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
try:
 from numpy import trapezoid
except ImportError:
 from numpy import trapz as trapezoid
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MAPPING={0:['121222','122221'],1:['130008','130634'],2:['132706','133157'],3:['134408','134815'],4:['140753','141519'],6:['150213','150506'],8:['151435','152040']}
METRICS=['signed_max','abs_max','phase_mean','abs_integral','positive_integral','rms','phase_sd','peak_to_peak','harmonic1','harmonic2']
NAMES={'signed_max':'Signed maximum','abs_max':'Absolute maximum','phase_mean':'Mean','abs_integral':'Absolute integral','positive_integral':'Positive integral','rms':'RMS','phase_sd':'SD','peak_to_peak':'Peak-to-peak range','harmonic1':'First Fourier amplitude','harmonic2':'Second Fourier amplitude','wave_corr':'M0-M3 waveform correlation','ellipse_area':'M0-M3 ellipse area'}
DEFS={'signed_max':'max(r) over all original valid samples','abs_max':'max(abs(r)) over all original valid samples','phase_mean':'Integral r dt / valid segment duration','abs_integral':'Integral abs(r) dt; trapezoids using original timestamps and existing endpoints','positive_integral':'Integral max(r,0) dt; trapezoids using original timestamps and existing endpoints','rms':'sqrt(mean(r^2)), all original valid samples','phase_sd':'Population SD (ddof=0), all original valid samples','peak_to_peak':'max(r)-min(r), all original valid samples','harmonic1':'2*abs(FFT(uniform native-N grid)[1])/N; one profile period','harmonic2':'2*abs(FFT(uniform native-N grid)[2])/N; two profile periods','wave_corr':'cov(r0,r3)/(SD0*SD3), paired original valid samples','ellipse_area':'pi*sqrt(det(population covariance(r0,r3))); no confidence multiplier'}
UNITS={m:('mA s' if 'integral' in m else 'mA') for m in METRICS};UNITS.update(wave_corr='dimensionless',ellipse_area='mA^2')

def closing_residual_slope(time,residual):
 t=np.asarray(time,float);x=np.asarray(residual,float)
 if len(t)<3 or not np.isfinite(t).all() or not np.isfinite(x).all():
  raise ValueError('Closing RC needs at least three finite paired samples.')
 t=t-t.mean()
 if t@t<=0:raise ValueError('Closing RC needs distinct timestamps.')
 return float(t@(x-x.mean())/(t@t))

def load(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def vector(rows,key,motor):return np.array([r[key][motor] if r.get(key) is not None and r[key][motor] is not None else np.nan for r in rows],float)
def groups(t,q,active,sign):
 mask=np.r_[False,sign*np.diff(q)>1e-6]&active;ids=np.flatnonzero(mask)
 return [g for g in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(g)>20 and t[g[-1]]-t[g[0]]>1]
def direction_profile(tt,xx):
 # Magnitude, moments and integrals use every original observation.
 t=np.asarray(tt,float)-tt[0];profile=np.asarray(xx,float).copy();duration=float(t[-1])
 assert len(profile)==len(t) and len(profile)>2 and np.all(np.diff(t)>0)
 areas=(float(trapezoid(profile,t)),float(trapezoid(np.abs(profile),t)),float(trapezoid(np.maximum(profile,0),t)))
 # DFT requires a uniform time grid. Its length equals the number of
 # actual valid observations, including their existing first/last samples.
 fft_profile=np.interp(np.linspace(0,duration,len(t)),t,profile)
 return profile,duration,areas,fft_profile,t

def cycle_profile(op,cl):
 po,to,ao,fo,tpo=op;pc,tc,ac,fc,tpc=cl;duration=to+tc
 profile=np.r_[po,pc];times=np.r_[tpo,to+tpc]
 n=len(po)+len(pc);grid=np.linspace(0,duration,n);fft_profile=np.empty(n)
 opening=grid<to
 fft_profile[opening]=np.interp(grid[opening],tpo,po)
 fft_profile[~opening]=np.interp(grid[~opening]-to,tpc,pc)
 # Opening/closing boundary samples share the compressed active-time
 # coordinate. Integrate each original direction separately, not across
 # the omitted reversal interval; native sample counts remain unchanged.
 return profile,duration,tuple(float(a+b) for a,b in zip(ao,ac)),fft_profile,times

def describe(p,duration,areas,fft_profile,times):
 x=p;fft=np.fft.rfft(fft_profile);mean=float(x.mean());sd=float(x.std(ddof=0));rms=float(np.sqrt(np.mean(x*x)))
 assert len(fft_profile)==len(x)
 assert np.isclose(rms*rms,mean*mean+sd*sd,rtol=1e-10,atol=1e-10)
 return dict(signed_max=float(p.max()),abs_max=float(np.abs(p).max()),phase_mean=float(areas[0]/duration),abs_integral=areas[1],positive_integral=areas[2],rms=rms,phase_sd=sd,peak_to_peak=float(np.ptp(p)),harmonic1=float(2*np.abs(fft[1])/len(x)),harmonic2=float(2*np.abs(fft[2])/len(x)))

def joint(a,b):
 assert len(a)==len(b)
 x=np.vstack([a,b]);cov=np.cov(x,bias=True);den=np.sqrt(cov[0,0]*cov[1,1]);corr=float(cov[0,1]/den) if den>1e-12 else np.nan
 return dict(wave_corr=corr,ellipse_area=float(np.pi*np.sqrt(max(0,np.linalg.det(cov)))))

def extract(folder,out,mapping=None):
 mapping=mapping or MAPPING
 rows=[];manifest=[];profile_rows=[];segments=[];qa=[];counts=[]
 for iteration,stamps in mapping.items():
  for rn,stamp in enumerate(stamps,1):
   rfiles=list(folder.rglob('*'+stamp+'_ros_log*.jsonl'));dfiles=list(folder.rglob('*'+stamp+'_motor_dt_replay*.jsonl'));assert len(rfiles)==len(dfiles)==1,(stamp,rfiles,dfiles)
   rp,dp=rfiles[0],dfiles[0];ros,dt=load(rp),load(dp);offset=len(ros)-len(dt);assert offset>=0
   assert all(r['measured_currents']==d['raw_measured_currents'] and r['motor_target_positions']==d['logged_motor_target_positions'] for r,d in zip(ros[offset:],dt)),('alignment',stamp)
   delta=np.array([r['time']-d['time'] for r,d in zip(ros[offset:],dt)]);assert np.ptp(delta)<1e-8
   assert all(d.get('gearbox_variant')=='gearbox_2' and d.get('instrument_config')=='scissors' for d in dt)
   t=np.array([r['time'] for r in ros]);q=vector(ros,'commanded_instrument_angles',3);active=np.array([str(r.get('sequence_condition','')).startswith('triangle') for r in ros]);op,cl=groups(t,q,active,1),groups(t,q,active,-1)
   assert len(op)==len(cl)==3 and all(o[-1]+1==c[0] for o,c in zip(op,cl)),stamp
   currents={m:vector(dt,'filtered_measured_currents',m) for m in [0,3]};preds={m:vector(dt,'offline_predicted_currents',m) for m in [0,3]};res={m:vector(dt,'filtered_current_deviation',m) for m in [0,3]}
   assert all(np.allclose(currents[m]-preds[m],res[m],equal_nan=True) for m in [0,3])
   run_id=re.sub(r'_ros_log.*\.jsonl$','',rp.name)
   manifest.append(dict(iteration=iteration,run_number=rn,run_id=run_id,ros_file=rp.name,replay_file=dp.name,ros_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),replay_sha256=hashlib.sha256(dp.read_bytes()).hexdigest(),offset_rows=offset,time_offset_s=float(delta[0]),exact_alignment=True))
   for cut,(o,c) in enumerate(zip(op,cl),1):
    cp=(q[c[0]-1]-q[c])/(q[c[0]-1]-q[c[-1]]);opp=(q[o]-q[o[0]-1])/(q[o[-1]]-q[o[0]-1]);vc=c[cp>=.03];vo=o[opp>=.03];ci,oi=vc-offset,vo-offset
    assert ci.min()>=0 and oi.min()>=0
    r=dict(iteration=iteration,run_number=rn,run_id=run_id,cut=cut,cycle_id=f'{iteration}.{rn}.{cut}')
    counts.append(dict(iteration=iteration,run_number=rn,cut=cut,cycle_id=r['cycle_id'],run_id=run_id,opening_original_samples=len(o),closing_original_samples=len(c),cycle_original_samples=len(o)+len(c),opening_valid_samples=len(vo),closing_valid_samples=len(vc),cycle_valid_samples=len(vo)+len(vc),closing_rc30_80_samples=int(((cp>=.30)&(cp<=.80)).sum()),logging_step_median_ms=float(1000*np.median(np.diff(t[np.r_[o,c]])))))
    profiles={}
    for motor in [0,3]:
     assert np.isfinite(res[motor][np.r_[oi,ci]]).all()
     opening=direction_profile(t[vo],res[motor][oi]);closing=direction_profile(t[vc],res[motor][ci]);cycle=cycle_profile(opening,closing)
     for phase,prof in [('closing',closing),('opening',opening),('cycle',cycle)]:
      profiles[(motor,phase)]=prof;r.update({f'{phase}_M{motor}_{metric}':val for metric,val in describe(*prof).items()})
      r[f'{phase}_M{motor}_sample_mean']=float(prof[0].mean())
      assert len(prof[0])==len(ci) if phase=='closing' else len(prof[0])==len(oi) if phase=='opening' else len(prof[0])==len(oi)+len(ci)
      for j,val in enumerate(prof[0]):profile_rows.append(dict(cycle_id=r['cycle_id'],iteration=iteration,run_number=rn,cut=cut,phase=phase,motor=motor,phase_fraction=float(prof[4][j]/prof[1]),native_sample_index=j,active_time_s=float(prof[4][j]),residual_ma=float(val)))
     for phase,idx in [('closing',ci),('opening',oi),('cycle',np.r_[oi,ci])]:r[f'{phase}_M{motor}_original_mean']=float(res[motor][idx].mean())
    for phase in ['closing','opening','cycle']:
     r.update({f'{phase}_joint_{metric}':val for metric,val in joint(profiles[(0,phase)][0],profiles[(3,phase)][0]).items()})
    inds=c[(cp>=.30)&(cp<=.80)]
    for motor in [0,3]:
     r[f'closing_M{motor}_rc30_80']=closing_residual_slope(t[inds],res[motor][inds-offset])
    targets0=vector(ros,'motor_target_positions',0);targets3=vector(ros,'motor_target_positions',3)
    segments.append(dict(iteration=iteration,run_number=rn,cut=cut,cycle_id=r['cycle_id'],closing_start_s=float(t[c[0]]),closing_valid_start_s=float(t[vc[0]]),closing_end_s=float(t[c[-1]]),opening_start_s=float(t[o[0]]),opening_valid_start_s=float(t[vo[0]]),opening_end_s=float(t[o[-1]]),closing_duration_s=profiles[(3,'closing')][1],opening_duration_s=profiles[(3,'opening')][1],closing_original_samples=len(c),opening_original_samples=len(o),cycle_original_samples=len(c)+len(o),closing_samples=len(ci),opening_samples=len(oi),cycle_samples=len(ci)+len(oi),closing_fft_samples=len(ci),opening_fft_samples=len(oi),cycle_fft_samples=len(ci)+len(oi),M0_target_range_pulses=float(np.ptp(targets0[vc])),M3_target_range_pulses=float(np.ptp(targets3[vc]))))
    qa.append(dict(cycle_id=r['cycle_id'],original_abs_residual_peak=float(np.abs(res[3][ci]).max()),native_abs_residual_peak=r['closing_M3_abs_max'],peak_error=abs(float(np.abs(res[3][ci]).max())-r['closing_M3_abs_max']),phase_mean=r['closing_M3_phase_mean'],original_sample_mean=r['closing_M3_original_mean']))
    rows.append(r)
 d=pd.DataFrame(rows);assert len(d)==3*sum(len(v) for v in mapping.values()) and d.groupby('run_id').size().eq(3).all()
 ref=ROOT/'data/reference_features.csv'
 use_ref=ref.exists() and mapping==MAPPING
 if use_ref:
  old=pd.read_csv(ref);keys=['iteration','run_number','cut'];check=d.merge(old,on=keys,validate='one_to_one',suffixes=('','_old'))
  for a,b in [('closing_M3_original_mean','closing_residual_ma'),('opening_M3_original_mean','opening_residual_ma'),('cycle_M3_original_mean','cycle_residual_ma')]:assert np.allclose(check[a],check[b],atol=1e-8),a
 d.to_csv(out/'cut_features.csv',index=False);pd.DataFrame(manifest).to_csv(out/'source_alignment.csv',index=False);pd.DataFrame(segments).to_csv(out/'motion_segments.csv',index=False);pd.DataFrame(qa).to_csv(out/'native_sample_check.csv',index=False);pd.DataFrame(profile_rows).to_csv(out/'phase_profiles.csv',index=False);pd.DataFrame(counts).to_csv(out/'sample_counts.csv',index=False)
 definitions=[]
 for phase in ['closing','opening','cycle']:
  for motor in [0,3]:
   for metric in METRICS:definitions.append(dict(column=f'{phase}_M{motor}_{metric}',name=NAMES[metric],signal='Motor DT current residual',phase=phase,motor=motor,definition=DEFS[metric],unit=UNITS[metric]))
  for metric in ['wave_corr','ellipse_area']:definitions.append(dict(column=f'{phase}_joint_{metric}',name=NAMES[metric],signal='M0 and M3 residuals',phase=phase,motor='0+3',definition=DEFS[metric],unit=UNITS[metric]))
 for motor in [0,3]:
  definitions.append(dict(column=f'closing_M{motor}_rc30_80',name='Closing RC 30-80%',signal='Motor DT current residual',phase='closing',motor=motor,definition='OLS slope of residual against native time over 30-80% of commanded closing travel',unit='mA/s'))
 pd.DataFrame(definitions).to_csv(out/'feature_definitions.csv',index=False)
 config=dict(mapping=mapping,residual='Supplied filtered measured minus predicted current, fixed DT; no DT retraining',filter='Supplied causal MA5',measured_current_features='None. Measured current is read only to verify the residual definition and source alignment.',windows='Opening and closing: initial3% of commanded travel excluded. No pauses. Closing residual RC for M0 and M3:30-80% commanded closing travel.',sample_policy='Use all actual valid original samples. No fixed count and no extra endpoint.', sample_count_summary={phase:{'min':int(pd.DataFrame(segments)[phase+'_samples'].min()),'max':int(pd.DataFrame(segments)[phase+'_samples'].max()),'median':float(pd.DataFrame(segments)[phase+'_samples'].median())} for phase in ['opening','closing','cycle']},interpolation='Only for DFT: uniform time grid with exactly the native valid sample count of that specific direction or active cycle. No interpolation across the omitted reversal gap. All other parameters use native samples/timestamps directly.',integrals='Native-sample trapezoids in seconds, including the existing first/last sample. Full cycle=sum separate opening+closing integrals; no excluded-gap integration.',phase_mean='Native-time integral divided by valid duration. Active-cycle integral divided by summed directional durations. Distinct from arithmetic sample mean.',sd='Population ddof0 on all native valid observations',fft='DFT on a uniform grid of N equal to actual observed sample count per window; h1/h2 describe profile shape, not physical rotation orders',joint='Pearson waveform correlation and pi sqrt(det(population covariance)) from paired original M0/M3 samples',same_data_verified=True,checks_against_old_features=use_ref,rc='Residual OLS against native time, M0 and M3, 30-80% commanded closing travel',dt_training_independence='Not established from run/replay metadata')
 (out/'extraction_protocol.json').write_text(json.dumps(config,indent=2))
 print('Extracted',len(d),'cuts;',len(d.columns),'columns. Undefined joint cells:',int(d[[c for c in d if 'joint_wave' in c]].isna().sum().sum()))
 print(pd.DataFrame(segments)[['M0_target_range_pulses','M3_target_range_pulses']].describe().round(3).to_string())
 return d

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--mapping',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);mapping={int(k):v for k,v in json.loads(a.mapping.read_text()).items()} if a.mapping else None;extract(a.input,a.output,mapping)
