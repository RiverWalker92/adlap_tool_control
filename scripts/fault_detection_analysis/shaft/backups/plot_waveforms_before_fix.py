"""Plot full-amplitude shaft waveforms and motor tracking at reduced hybrid peaks.

Visual interpolation only: native source samples and classifier results are unchanged.
Each condition keeps its physical duration; pauses are excluded using logged labels.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
 from shaft_features import load_rows,column,interpolate
except ImportError:
 from dof2_features import load_rows,column,interpolate
from artifact_io import save_figure

ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='code' else Path(__file__).resolve().parent
COLORS={'healthy':'#287ca8','faulty':'#e58032','video':'#329653','hybrid':'#d04d50'}
CONDITIONS=[f'{wave}_f{frequency}_r1p0' for frequency in ['0p5','1p0','2p0'] for wave in ['sinusoid','triangle']]

def load_run(source,manifest_dir,data_dir=None):
 paths={k:manifest_dir/v for k,v in source['files'].items()}
 if data_dir:
  for kind in paths:
   found=list(data_dir.rglob(source['run']+'_'+kind+'.jsonl'))
   if len(found)!=1:raise ValueError(f'Expected one {kind} file for {source["run"]}, found {len(found)}')
   paths[kind]=found[0]
 hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
 if not data_dir and source.get('sha256') and source['sha256']!=hashes:raise ValueError('Input hash mismatch: '+source['run'])
 ros=load_rows(paths['ros_log']);motor=load_rows(paths['motor_dt_replay']);video=load_rows(paths['webcam_angles'])
 tr=column(ros,'time');tm=column(motor,'time');cr=np.array([r.get('sequence_condition') for r in ros],object);cm=np.array([r.get('sequence_condition') for r in motor],object)
 names=list(dict.fromkeys(n for n in cr if n not in [None,'pause']))
 shifts=[tr[np.flatnonzero(cr==n)[0]]-tm[np.flatnonzero(cm==n)[0]] for n in names]
 shift=float(np.median(shifts))
 if np.ptp(shifts)>.012:raise ValueError('Unstable replay alignment')
 tm=tm+shift
 origin=float(np.nanmedian(column(ros,'ros_timestamp')-tr));tv=column(video,'ros_time_s')-origin;yv=column(video,'measured_angle_red_shaft_zeroed_filtered')
 valid=np.isfinite(tv)&np.isfinite(yv);tv=tv[valid];yv=yv[valid];order=np.argsort(tv);tv=tv[order];yv=yv[order]
 pred=np.rad2deg(column(ros,'hybrid_predicted_instrument_angles',1));command=np.rad2deg(column(ros,'commanded_instrument_angles',1))
 predv=interpolate(tr,pred,tv)
 targets=np.column_stack([column(ros,'motor_target_positions',m) for m in range(4)])
 measured=np.column_stack([column(ros,'measured_motor_positions',m) for m in range(4)])
 commanded=np.column_stack([column(ros,'commanded_motor_positions',m) for m in range(4)])
 currents=np.column_stack([column(motor,'filtered_current_deviation',m) for m in range(4)])
 position_residual=np.column_stack([column(motor,'encoder_deviation',m) for m in range(4)])
 position_measured=np.column_stack([column(motor,'measured_positions',m) for m in range(4)])
 position_predicted=np.column_stack([column(motor,'offline_predicted_positions',m) for m in range(4)])
 valid=np.isfinite(position_residual)&np.isfinite(position_measured)&np.isfinite(position_predicted)
 if not valid.any() or np.max(abs((position_measured-position_predicted-position_residual)[valid]))>1e-8:
  raise ValueError('Motor-DT position residual does not equal measured minus predicted position')
 records={}
 for name in CONDITIONS:
  idx=np.flatnonzero(cr==name)
  if not len(idx) or np.any(np.diff(idx)!=1):raise ValueError('Missing or disjoint full-amplitude condition '+name)
  start=tr[idx[0]];end=tr[idx[-1]+1] if idx[-1]+1<len(tr) else tr[idx[-1]]+np.median(np.diff(tr));duration=end-start;period=duration/3
  frequency=float(name.split('_')[1][1:].replace('p','.'))
  if abs(period-1/frequency)>.03:raise ValueError('Unexpected three-cycle condition duration')
  vm=(tv>=start)&(tv<end)&np.isfinite(predv)
  vt=tv[vm]-start;vv=yv[vm].copy();pv=predv[vm]
  offsets=[]
  for cycle in range(3):
   keep=(vt>=cycle*period)&(vt<(cycle+1)*period)
   if keep.sum()<8:raise ValueError('Insufficient video in cycle')
   offset=float(np.mean(vv[keep]-pv[keep]));vv[keep]-=offset;offsets.append(offset)
  mm=(cm==name)&(tm>=start)&(tm<end)
  rt=tr[idx]-start
  records[name]=dict(start=float(start),end=float(end),duration=float(duration),period=float(period),rt=rt,command=command[idx],hybrid=pred[idx],target=targets[idx],measured=measured[idx],commanded=commanded[idx],mt=tm[mm]-start,current=currents[mm],position_residual=position_residual[mm],dt_position_measured=position_measured[mm],dt_position_predicted=position_predicted[mm],vt=vt,video=vv,offsets=offsets)
 return dict(run=source['run'],true_class=int(source['true_class']),conditions=records,sha256=hashes,shift=shift)

def curve(record,time_key,value_key,q,motor=None):
 values=record[value_key];values=values[:,motor] if motor is not None else values
 return interpolate(record[time_key],values,q)

def band_stats(values):
 values=np.asarray(values,float);valid=np.isfinite(values).all(axis=0)
 mean=np.full(values.shape[1],np.nan);low=mean.copy();high=mean.copy()
 mean[valid]=values[:,valid].mean(axis=0);low[valid]=values[:,valid].min(axis=0);high[valid]=values[:,valid].max(axis=0)
 return mean,low,high

def style(axes):
 for ax in axes:
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(loc='upper right',fontsize=7,ncol=2)

def select_event(run,healthy,name):
 record=run['conditions'][name];duration=record['duration'];q=np.arange(0.,duration,.01);cycle=np.minimum((q/record['period']).astype(int),2)
 hy=curve(record,'rt','hybrid',q);ref=np.nanmean([curve(r['conditions'][name],'rt','hybrid',q) for r in healthy],axis=0)
 reference=ref.copy()
 for j in range(3):
  mask=cycle==j;reference[mask]+=np.nanmean(hy[mask])-np.nanmean(ref[mask])
 positive=np.zeros(len(q),bool)
 for j in range(3):
  mask=cycle==j;centre=np.nanmean(reference[mask]);positive[mask]=(reference[mask]-centre)>.5*(np.nanmax(reference[mask])-centre)
 delta=reference-hy;eligible=positive&np.isfinite(delta)
 peak=int(np.nanargmax(np.where(eligible,delta,np.nan)));event=q[peak]
 return q,hy,reference,float(event),float(delta[peak])

def overview(runs,figures,zoom_condition,inset_window,forced_run=None):
 fig = plt.figure(figsize=(14, 15))
 grid = fig.add_gridspec(
     7, 1,
     height_ratios=[1, 1, 1, 1, 1, 0.9, 0.9],
     hspace=0.4
 )

 axes = []
 for row in range(5):
     axes.append(fig.add_subplot(
         grid[row, 0],
         sharex=axes[0] if axes else None
     ))

 inset_angle = fig.add_subplot(grid[5, 0])
 inset_motor = fig.add_subplot(
     grid[6, 0], sharex=inset_angle
 )

 for ax in axes[:-1]:
     ax.tick_params(labelbottom=False)
 cursor=0.;segments=[]
 for name in CONDITIONS:
  duration=float(np.median([r['conditions'][name]['duration'] for r in runs]));q=np.arange(0.,duration,.01);x=q+cursor
  first=runs[0]['conditions'][name];axes[0].plot(x,curve(first,'rt','command',q),color='#454545',label='Instrument command' if cursor==0 else None)
  for cls,color,label in [(0,COLORS['healthy'],'Healthy'),(1,COLORS['faulty'],'Faulty')]:
   selected=[r for r in runs if r['true_class']==cls]
   for panel,motor in [(1,1),(2,2)]:
    ys=np.array([curve(r['conditions'][name],'mt','current',q,motor) for r in selected]);mean,low,high=band_stats(ys)
    axes[panel].plot(x,mean,color=color,label=label+' mean' if cursor==0 else None)
    axes[panel].fill_between(x,low,high,color=color,alpha=.18)
   panel=3 if cls==0 else 4
   for key,time,color,label in [('video','vt',COLORS['video'],'Video, constant offset corrected'),('hybrid','rt',COLORS['hybrid'],'Hybrid prediction')]:
    ys=np.array([curve(r['conditions'][name],time,key,q) for r in selected]);mean,low,high=band_stats(ys);axes[panel].plot(x,mean,color=color,label=label if cursor==0 else None)
    axes[panel].fill_between(x,low,high,color=color,alpha=.16)
  label=('Sine' if name.startswith('sinusoid') else 'Triangle')+'\n'+name.split('_')[1][1:].replace('p','.')+' Hz'
  axes[0].text(cursor+duration/2,1.02,label,transform=axes[0].get_xaxis_transform(),ha='center',fontsize=8)
  segments.append(dict(condition=name,concatenated_start_s=cursor,concatenated_end_s=cursor+duration))
  cursor+=duration
  for ax in axes:ax.axvline(cursor,color='#777777',lw=.7,ls=':',alpha=.5)
 axes[0].set_ylabel('Command [deg]');axes[1].set_ylabel('M1 residual [mA]');axes[2].set_ylabel('M2 residual [mA]')
 axes[3].set_ylabel('Healthy angle [deg]');axes[4].set_ylabel('Faulty angle [deg]');axes[4].set_xlabel('Concatenated active time [s]; pauses removed')
 axes[0].set_title('Shaft bending — full amplitude (0.40 rad)',pad=36)
 style(axes)
 healthy=[r for r in runs if r['true_class']==0];faulty=[r for r in runs if r['true_class']==1 and (forced_run is None or r['run']==forced_run)]
 if not faulty:raise ValueError('No matching faulty run for inset')
 choices=[(r,*select_event(r,healthy,zoom_condition)) for r in faulty]
 run,q,hy,reference,event,deficit=max(choices,key=lambda item:item[-1])
 record=run['conditions'][zoom_condition];duration=record['duration']
 start=max(0.,event-inset_window/2);end=min(duration,start+inset_window)
 if end-start<inset_window:start=max(0.,end-inset_window)
 idx=(record['rt']>=start)&(record['rt']<=end);t=record['rt'][idx];vm=(record['vt']>=start)&(record['vt']<=end)
 inset_angle.plot(t,record['hybrid'][idx],color=COLORS['hybrid'],lw=1.4,label='Faulty hybrid')
 inset_angle.plot(record['vt'][vm],record['video'][vm],color=COLORS['video'],lw=1.2,label='Faulty video')
 inset_angle.plot(q,reference,color=COLORS['healthy'],ls='--',lw=1.2,label='Healthy hybrid*')
 mi=(record['mt']>=start)&(record['mt']<=end);residual_time=record['mt'][mi]
 faulty_residual=record['position_residual'][mi,2]
 inset_motor.plot(residual_time,faulty_residual,color=COLORS['faulty'],lw=1.4,label='Faulty')
 healthy_residual=[curve(h['conditions'][zoom_condition],'mt','position_residual',residual_time,2) for h in healthy]
 mean,low,high=band_stats(healthy_residual)
 inset_motor.plot(residual_time,mean,color=COLORS['healthy'],ls='--',lw=1.2,label='Healthy mean')
 inset_motor.fill_between(residual_time,low,high,color=COLORS['healthy'],alpha=.15)
 inset_motor.axhline(0,color='#555555',lw=.7)
 inset_angle.set_ylabel('Angle [deg]',fontsize=8);inset_motor.set_ylabel('M2 position residual [counts]',fontsize=8);inset_motor.set_xlabel('Condition time [s]',fontsize=8)
 inset_angle.set_title('Motor-DT residual detail\nFaulty run '+run['run'].split('_')[-2]+' '+run['run'].split('_')[-1]+'\nTriangle 0.5 Hz',fontsize=9,pad=10)
 for ax in [inset_angle,inset_motor]:
  ax.set_xlim(start,end);ax.grid(alpha=.2);ax.tick_params(labelsize=7);ax.axvline(event,color='#777777',ls=':',lw=.7)
  ax.legend(fontsize=6,loc='upper left',framealpha=.85);ax.spines[['top','right']].set_visible(False)
  ax.locator_params(axis='x',nbins=3)
 inset_angle.tick_params(labelbottom=False)
 # Limit the angle inset vertically to its displayed interval, not all three cycles.
 shown=(q>=start)&(q<=end);all_angles=np.r_[record['hybrid'][idx],record['video'][vm],reference[shown]]
 lo,hi=np.nanmin(all_angles),np.nanmax(all_angles);pad=max(1.,.12*(hi-lo));inset_angle.set_ylim(lo-pad,hi+pad)
 segment=next(item for item in segments if item['condition']==zoom_condition)
 left=segment['concatenated_start_s']+start;right=segment['concatenated_start_s']+end
 axes[4].axvspan(left,right,facecolor='#8b6aaf',alpha=.12,edgecolor='#6f5688',lw=.7)


 fig.text(
     .075, .035,
     'Main plots: class means; bands: run min–max. '
     'Video offset corrected per cycle; pauses removed.',
     fontsize=8
 )
 fig.text(
     .075, .02,
     'Healthy hybrid: constant offset aligned. '
     'Position residuals: logged motor DT, no offset correction.',
     fontsize=8
 )
 fig.subplots_adjust(
     left=.075, right=.985, bottom=.10, top=.94
 )
 save_figure(fig,figures/'full_amplitude_waveforms')
 return segments,dict(run=run['run'],condition=zoom_condition,event_log_s=record['start']+event,condition_time_s=event,hybrid_deficit_deg=deficit,inset_start_s=start,inset_end_s=end)


def zoom(runs,name,figures,out,window,forced_run=None,render=False):
 healthy=[r for r in runs if r['true_class']==0];faulty=[r for r in runs if r['true_class']==1 and (forced_run is None or r['run']==forced_run)]
 if not faulty:raise ValueError('No matching faulty run')
 summaries=[];samples=[]
 for run in faulty:
  record=run['conditions'][name];duration=record['duration'];q,hy,reference,event,deficit=select_event(run,healthy,name)
  start=max(0.,event-window/2);end=min(duration,event+window/2)
  if end-start<window:start=max(0.,end-window);end=min(duration,start+window)
  native=(record['rt']>=start)&(record['rt']<=end)
  if native.sum()<8:raise ValueError('Insufficient zoom samples')
  fig,axes=plt.subplots(6,1,figsize=(11,11),sharex=True)
  t=record['rt'][native];c=curve(record,'rt','command',t);hp=record['hybrid'][native]
  vm=(record['vt']>=start)&(record['vt']<=end)
  axes[0].plot(t,c,color='#555555',alpha=.6,lw=1,label='Instrument command')
  axes[0].plot(t,hp,color=COLORS['hybrid'],lw=1.8,label='Faulty hybrid prediction')
  axes[0].plot(record['vt'][vm],record['video'][vm],color=COLORS['video'],label='Faulty video, offset corrected')
  axes[0].plot(q,reference,color=COLORS['healthy'],ls='--',label='Healthy hybrid mean, cycle offset aligned')
  axes[0].set_ylabel('Angle [deg]')
  for m in range(4):
   target=record['target'][native,m];measured=record['measured'][native,m];baseline=float(target[0]);ax=axes[m+1]
   ax.plot(t,target-baseline,color='#333333',lw=1.5,label=f'M{m} controller target')
   ax.plot(t,measured-baseline,color=COLORS['faulty'],label=f'M{m} measured position')
   cmd=record['commanded'][native,m]
   if np.isfinite(cmd).sum()>1:ax.plot(t,cmd-baseline,color='#a26bb7',ls=':',label=f'M{m} upstream command')
   if m==2:
    healthy_motor=[]
    for h in healthy:
     hr=h['conditions'][name];hm=curve(hr,'rt','measured',t,m);ht=curve(hr,'rt','target',np.array([t[0]]),m)[0]
     healthy_motor.append(hm-ht)
    ax.plot(t,np.nanmean(healthy_motor,axis=0),color=COLORS['healthy'],ls='--',label='Healthy M2 measured, target-start aligned')
   error=measured-target;axes[5].plot(t,error,label=f'M{m}',lw=1)
   ax.set_ylabel(f'M{m} position\n[encoder counts]')
   summaries.append(dict(run=run['run'],condition=name,motor=f'M{m}',zoom_start_log_s=record['start']+start,zoom_end_log_s=record['start']+end,event_log_s=record['start']+event,event_condition_s=event,hybrid_deficit_deg=float(deficit),event_tracking_error_counts=float(np.interp(event,t,error)),tracking_mae_counts=float(np.nanmean(abs(error))),tracking_max_abs_counts=float(np.nanmax(abs(error))),target_reference_counts=baseline))
   for i,ti in enumerate(t):samples.append(dict(run=run['run'],condition=name,log_time_s=record['start']+ti,condition_time_s=ti,motor=f'M{m}',controller_target_counts=target[i],measured_position_counts=measured[i],tracking_error_counts=error[i],hybrid_deg=hp[i],instrument_command_deg=c[i]))
  axes[5].set_ylabel('Measured − target\n[encoder counts]');axes[5].axhline(0,color='#555555',lw=.6)
  axes[5].set_xlabel('Time within active condition [s]')
  for ax in axes:ax.axvline(event,color='#777777',ls=':',lw=1);ax.set_xlim(start,end)
  style(axes);axes[0].set_title(name.replace('_',' ')+' — '+run['run']+'\nZoom selected by largest positive-peak hybrid deficit versus healthy phase reference',fontsize=10)
  fig.text(.08,.012,'Motor positions subtract a fixed initial target for each run and motor; healthy M2 is aligned the same way. Vertical line: selected event. Logged snapshots; no lag correction.',fontsize=8)
  fig.tight_layout(rect=(0,.035,1,.98))
  if render:save_figure(fig,figures/('motor_tracking_zoom_'+run['run']))
  else:plt.close(fig)
 return summaries,samples

def export_position_residual_samples(runs,metadata,out):
 records=[];summaries=[];name=metadata['condition'];start=metadata['inset_start_s'];end=metadata['inset_end_s'];event=metadata['condition_time_s']
 for run in runs:
  if run['true_class']==1 and run['run']!=metadata['run']:continue
  record=run['conditions'][name];mask=(record['mt']>=start)&(record['mt']<=end)
  t=record['mt'][mask];residual=record['position_residual'][mask,2]
  measured=record['dt_position_measured'][mask,2];predicted=record['dt_position_predicted'][mask,2]
  if not np.isfinite(residual).all():raise ValueError('Nonfinite position residual in inset')
  assert np.allclose(measured-predicted,residual,atol=1e-8)
  summaries.append(dict(run=run['run'],true_class=run['true_class'],condition=name,event_condition_s=event,event_position_residual_counts=float(np.interp(event,t,residual)),position_residual_mae_counts=float(abs(residual).mean()),position_residual_max_abs_counts=float(abs(residual).max())))
  for i,ti in enumerate(t):records.append(dict(run=run['run'],true_class=run['true_class'],condition=name,condition_time_s=ti,aligned_log_time_s=record['start']+ti,dt_measured_position_counts=measured[i],dt_predicted_position_counts=predicted[i],position_residual_counts=residual[i]))
 pd.DataFrame(records).to_csv(out/'position_residual_zoom_samples.csv',index=False);pd.DataFrame(summaries).to_csv(out/'position_residual_zoom_summary.csv',index=False)
 print('Motor-DT position residual = measured minus DT prediction [encoder counts].')
 print(pd.DataFrame(summaries).to_string(index=False))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'runs.json');p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path,default=ROOT/('outputs' if (ROOT/'code').exists() else 'results'));p.add_argument('--zoom-condition',default='triangle_f0p5_r1p0');p.add_argument('--zoom-window',type=float,default=2.0);p.add_argument('--zoom-run');p.add_argument('--separate-zooms',action='store_true');p.add_argument('--inset-window',type=float,default=3.0);a=p.parse_args()
 if a.zoom_condition not in CONDITIONS or a.zoom_window<=0 or a.inset_window<=0:raise ValueError('Use a full-amplitude condition and positive zoom window')
 manifest=json.loads(a.manifest.read_text());runs=[load_run(s,a.manifest.resolve().parent,a.data_dir) for s in manifest['runs']]
 if sum(r['true_class']==0 for r in runs)!=2 or sum(r['true_class']==1 for r in runs)!=2:raise ValueError('This figure protocol requires two healthy and two faulty runs')
 out=a.output_dir.resolve();figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
 segments,inset=overview(runs,figures,a.zoom_condition,a.inset_window,a.zoom_run);export_position_residual_samples(runs,inset,out);summary,samples=zoom(runs,a.zoom_condition,figures,out,a.zoom_window,a.zoom_run,a.separate_zooms)
 pd.DataFrame(summary).to_csv(out/'motor_tracking_zoom_summary.csv',index=False);pd.DataFrame(samples).to_csv(out/'motor_tracking_zoom_samples.csv',index=False)
 (out/'waveform_plot_metadata.json').write_text(json.dumps(dict(conditions=CONDITIONS,segments=segments,inset=inset,zoom_condition=a.zoom_condition,zoom_window_s=a.zoom_window,video_offset='mean(video - hybrid) removed per cycle',band='minimum and maximum of the two runs per class',alignment='logged condition boundaries; no lag fitting',event_rule='largest positive-peak deficit against healthy mean after cycle-offset alignment',units_motor='logged encoder counts; no conversion to angle',motor_inset_signal='encoder_deviation = measured_positions - offline_predicted_positions',source_hashes={r['run']:r['sha256'] for r in runs}),indent=2))
 print('Wrote full_amplitude_waveforms.png with a motor-DT position-residual inset.')
 if a.separate_zooms:print('Also wrote individual motor_tracking_zoom PNGs.')
 print('Zoom condition:',a.zoom_condition)
 print(pd.DataFrame(summary)[['run','motor','event_log_s','hybrid_deficit_deg','tracking_mae_counts','tracking_max_abs_counts']].to_string(index=False))
if __name__=='__main__':main()
