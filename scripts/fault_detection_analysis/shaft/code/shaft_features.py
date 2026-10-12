"""Native-sample DOF2 extraction; fixed logged DT predictions are read only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT = ['maximum','absolute_maximum','mean','absolute_integral','positive_integral','rms','sd','range','a1','a2','rc']
VIDEO = ['gain_positive','gain_negative','gain_difference','video_mae','video_rmse','video_peak','video_relative_mae']
RANGES = ['small','medium','full']
AMPLITUDES = {'r0p2':('small',.10),'r0p5':('medium',.20),'r1p0':('full',.40)}


def definitions():
    labels=['Signed maximum','Absolute maximum','Phase mean','Absolute phase integral','Positive phase integral','RMS','Population SD','Peak-to-peak range','Fundamental amplitude','Second harmonic amplitude','RC coefficient']
    formulas=['max r','max |r|','integral r dphi / (2*pi)','integral |r| dphi','integral max(r,0) dphi','sqrt(mean r^2)','sqrt(mean (r - sample_mean)^2)','max r - min r','sqrt(a1^2+b1^2), native-sample least squares','sqrt(a2^2+b2^2), native-sample least squares','OLS slope of residual against fixed-origin command-cycle phase, with intercept']
    rows=[]
    for m in range(4):
        for key,label,formula in zip(CURRENT,labels,formulas):
            rows.append(dict(feature=f'M{m}_{key}',label=f'M{m} {label}',family='current',unit='mA rad' if 'integral' in key else ('mA/rad' if key=='rc' else 'mA'),definition=formula))
    labels=['Positive gain','Negative gain','Absolute gain difference','Centred video MAE','Centred video RMSE','Absolute centred video peak','Relative video MAE']
    formulas=['positive-command slope of video vs hybrid, shared intercept','negative-command slope of video vs hybrid, shared intercept','abs(gain_positive-gain_negative)','mean abs(video-hybrid-cycle_offset)','sqrt(mean centred_video_residual^2)','max abs(centred_video_residual)','video_mae / nominal command amplitude in degrees']
    for key,label,formula in zip(VIDEO,labels,formulas):
        rows.append(dict(feature=key,label=label,family='video',unit='deg' if key in ['video_mae','video_rmse','video_peak'] else '-',definition=formula))
    return pd.DataFrame(rows)


def load_rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def column(rows,key,index=None):
    values=[]
    for r in rows:
        value=r.get(key)
        if index is not None:value=value[index] if isinstance(value,list) and len(value)>index else None
        values.append(np.nan if value is None else value)
    return np.asarray(values,dtype=float)


def interpolate(t,y,q,max_gap=.10):
    valid=np.isfinite(t)&np.isfinite(y);t=t[valid];y=y[valid]
    if len(t)<2 or not np.all(np.diff(t)>0):raise ValueError('Invalid interpolation timestamps')
    z=np.interp(q,t,y,left=np.nan,right=np.nan)
    i=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
    z[(t[i]-t[i-1]>max_gap)&(q>t[i-1])&(q<t[i])]=np.nan
    return z


def integral(y,x):
    return float(np.sum((y[:-1]+y[1:])*.5*np.diff(x)))


def current_parameters(t,r,start,end):
    """All original unique samples for statistics, harmonics and RC.

    Phase origin is the logged start of this command cycle; one cycle spans
    2*pi. Only phase integration closes the periodic curve with the first value
    at 0 and 2*pi. No added points enter maxima, RMS, SD, harmonics or RC.
    The phase-integral mean is distinct from the arithmetic sample mean.
    """
    t=np.asarray(t,float);r=np.asarray(r,float)
    if len(r)<8 or not np.isfinite(r).all() or not np.all(np.diff(t)>0):raise ValueError('Invalid current samples')
    phi=2*np.pi*(t-start)/(end-start)
    if np.min(phi)<-1e-8 or np.max(phi)>=2*np.pi+1e-8:raise ValueError('Samples outside cycle')
    phase=np.r_[0.,phi,2*np.pi];closed=np.r_[r[0],r,r[0]]
    design=np.column_stack([np.ones(len(r)),np.sin(phi),np.cos(phi),np.sin(2*phi),np.cos(2*phi)])
    coef=np.linalg.lstsq(design,r,rcond=None)[0]
    slope=np.linalg.lstsq(np.column_stack([np.ones(len(r)),phi]),r,rcond=None)[0][1]
    return dict(maximum=float(r.max()),absolute_maximum=float(abs(r).max()),mean=integral(closed,phase)/(2*np.pi),
                absolute_integral=integral(abs(closed),phase),positive_integral=integral(np.maximum(closed,0),phase),
                rms=float(np.sqrt(np.mean(r*r))),sd=float(np.std(r,ddof=0)),range=float(np.ptp(r)),
                a1=float(np.hypot(*coef[1:3])),a2=float(np.hypot(*coef[3:5])),rc=float(slope)),float(np.mean(r))


def extract(manifest_path,out,data_dir=None):
    manifest_path=Path(manifest_path).resolve();manifest=json.loads(manifest_path.read_text());root=manifest_path.parent
    if data_dir:
        root=Path(data_dir).resolve()
        for source in manifest['runs']:
            source['files']={};source.pop('sha256',None)
            for kind in ['ros_log','motor_dt_replay','webcam_angles']:
                matches=list(root.rglob(source['run']+'_'+kind+'.jsonl'))
                if len(matches)!=1:raise ValueError(f"Expected one {kind} for {source['run']}; found {len(matches)}")
                source['files'][kind]=str(matches[0].relative_to(root))
    records=[];audits=[]
    for source in manifest['runs']:
        paths={k:root/v for k,v in source['files'].items()}
        hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
        if source.get('sha256') and hashes!=source['sha256']:raise ValueError('Input hash mismatch '+source['run'])
        ros=load_rows(paths['ros_log']);motor=load_rows(paths['motor_dt_replay']);video=load_rows(paths['webcam_angles'])
        tr=column(ros,'time');tm=column(motor,'time')
        cr=np.asarray([r.get('sequence_condition') for r in ros],object);cm=np.asarray([r.get('sequence_condition') for r in motor],object)
        names=list(dict.fromkeys(x for x in cr if x not in [None,'pause']))
        if len(names)!=18 or not np.all(np.diff(tr)>0) or not np.all(np.diff(tm)>0):raise ValueError('Unexpected sequence/timestamps')
        shifts=np.array([tr[np.flatnonzero(cr==n)[0]]-tm[np.flatnonzero(cm==n)[0]] for n in names]);shift=float(np.median(shifts));tm+=shift
        if np.ptp(shifts)>.012:raise ValueError('Unstable replay alignment')
        origin=float(np.median(column(ros,'ros_timestamp')-tr))
        tv=column(video,'ros_time_s')-origin;y=column(video,'measured_angle_red_shaft_zeroed_filtered')
        valid=np.isfinite(tv)&np.isfinite(y);tv=tv[valid];y=y[valid];ix=np.argsort(tv);tv=tv[ix];y=y[ix]
        if not np.all(np.diff(tv)>0):raise ValueError('Nonmonotonic video')
        pred=interpolate(tr,np.rad2deg(column(ros,'hybrid_predicted_instrument_angles',1)),tv)
        cmd=column(ros,'commanded_instrument_angles',1);cv=interpolate(tr,cmd,tv)
        vc=cr[np.clip(np.searchsorted(tr,tv,side='right')-1,0,len(tr)-1)]
        residuals=[];filter_errors=[]
        for m in range(4):
            filtered=column(motor,'filtered_measured_currents',m);raw=column(motor,'raw_measured_currents',m)
            residual=filtered-column(motor,'offline_predicted_currents',m)
            if np.nanmax(abs(residual-column(motor,'filtered_current_deviation',m)))>1e-8:raise ValueError('Current residual mismatch')
            err=float(np.nanmax(abs(filtered-pd.Series(raw).rolling(5,min_periods=1).mean().to_numpy())))
            if err>1e-8:raise ValueError('Unexpected current filter')
            residuals.append(residual);filter_errors.append(err)
        currents=np.column_stack(residuals)
        nearest=np.clip(np.searchsorted(tr,tm),1,len(tr)-1);nearest=np.where(abs(tr[nearest-1]-tm)<abs(tr[nearest]-tm),nearest-1,nearest)
        target=column(motor,'logged_motor_target_positions',2);logged=column(ros,'motor_target_positions',2)[nearest];ok=np.isfinite(target)&np.isfinite(logged)
        exact=float(np.mean(target[ok]==logged[ok]));
        if exact<.99:raise ValueError('Replay target alignment failed')
        dt=float(np.median(np.diff(tm)));dv=float(np.median(np.diff(tv)))
        for name in names:
            idx=np.flatnonzero(cr==name)
            if np.any(np.diff(idx)!=1):raise ValueError('Condition occurs in separated intervals')
            start=float(tr[idx[0]]);end=float(tr[idx[-1]+1]) if idx[-1]+1<len(tr) else float(tr[idx[-1]]+np.median(np.diff(tr)))
            wave,freq,range_code=name.split('_');frequency=float(freq[1:].replace('p','.'));range_name,amplitude=AMPLITUDES[range_code]
            period=(end-start)/3
            if abs(period-1/frequency)>.03:raise ValueError('Unexpected cycle duration')
            hold=(tv>=start-.55)&(tv<=start-.05)&np.isfinite(pred)
            if hold.sum()<6:raise ValueError('Missing prediction hold reference')
            reference=float(np.median(pred[hold]))
            for cycle in range(3):
                a=start+cycle*period;b=start+(cycle+1)*period
                mask=(cm==name)&(tm>=a)&(tm<b)&np.isfinite(currents).all(axis=1)
                vm=(vc==name)&(tv>=a)&(tv<b)&np.isfinite(pred)&np.isfinite(cv)
                ci=mask.sum()/(period/dt);vi=vm.sum()/(period/dv)
                halfcoverage=[np.sum(s*cv[vm]>1e-5)/(period/(2*dv)) for s in [1,-1]]
                if ci<.8 or vi<.8 or min(halfcoverage)<.75:raise ValueError('Insufficient sample coverage')
                row=dict(run=source['run'],true_class=int(source['true_class']),label=source['label'],condition=name,
                         cycle=cycle+1,range=range_name,waveform=wave,frequency_hz=frequency,command_amplitude_rad=amplitude,
                         start_s=a,end_s=b,current_n=int(mask.sum()),video_n=int(vm.sum()),current_coverage=ci,video_coverage=vi,
                         observed_command_peak_rad=float(np.max(abs(cmd[(tr>=a)&(tr<b)]))))
                for m in range(4):
                    features,sample_mean=current_parameters(tm[mask],currents[mask,m],a,b)
                    for k,v in features.items():row[f'M{m}_{k}']=v
                    row[f'M{m}_arithmetic_mean']=sample_mean
                    if not np.isclose(features['rms']**2,sample_mean**2+features['sd']**2,atol=1e-8):raise ValueError('RMS identity failed')
                measured=y[vm];prediction=pred[vm];command=cv[vm]
                raw=measured-prediction;bias=float(raw.mean());centred=raw-bias
                x=prediction-reference
                positive=np.where(command>1e-5,x,0.);negative=np.where(command< -1e-5,x,0.)
                design=np.column_stack([np.ones(len(x)),positive,negative])
                spans=[np.ptp(x[s*command>1e-5]) for s in [1,-1]]
                if len(x)<12 or min(spans)<.5 or np.linalg.matrix_rank(design)!=3 or np.linalg.cond(design)>1000:raise ValueError('Invalid gain fit')
                gains=np.linalg.lstsq(design,measured,rcond=None)[0]
                row.update(gain_positive=float(gains[1]),gain_negative=float(gains[2]),gain_difference=float(abs(gains[1]-gains[2])),
                           gain_intercept_deg=float(gains[0]),video_offset_deg=bias,video_mae=float(abs(centred).mean()),
                           video_rmse=float(np.sqrt(np.mean(centred**2))),video_peak=float(abs(centred).max()),
                           video_relative_mae=float(abs(centred).mean()/np.rad2deg(amplitude)))
                if abs(centred.mean())>1e-9:raise ValueError('Video centring failed')
                records.append(row)
        audits.append(dict(run=source['run'],class_label=source['label'],ros_rows=len(ros),motor_rows=len(motor),video_rows=len(video),
                           motor_to_ros_shift_s=shift,shift_spread_s=float(np.ptp(shifts)),target_M2_match=exact,
                           current_dt_s=dt,video_dt_s=dv,filter_max_errors_ma=filter_errors,sha256=hashes))
        print('Extracted',source['run'],54,'cycles',flush=True)
    frame=pd.DataFrame(records);features=list(definitions().feature)
    if len(frame)!=54*len(manifest['runs']) or not np.isfinite(frame[features]).all().all():raise ValueError('Incomplete feature table')
    if frame.duplicated(['run','condition','cycle']).any():raise ValueError('Duplicate cycles')
    frame.to_csv(out/'cycle_features.csv',index=False)
    (out/'data_audit.json').write_text(json.dumps(audits,indent=2))
    (out/'resolved_manifest.json').write_text(json.dumps(manifest,indent=2))
    return frame


def aggregate(frame):
    ids=['run','true_class','label','condition','range','waveform','frequency_hz','command_amplitude_rad']
    groups=frame.groupby(ids,sort=False)
    if not (groups.size()==3).all():raise ValueError('Each condition must contain exactly three cycles')
    return groups[list(definitions().feature)].mean().reset_index()
