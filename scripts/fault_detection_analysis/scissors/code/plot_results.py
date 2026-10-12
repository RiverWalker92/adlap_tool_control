"""Scientific figures (PNG/SVG) and count-based gearbox-style tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from classify_cuts import SCENARIOS,SELECTIONS,dataset
from extract_features import METRICS,NAMES,UNITS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';MAIN=OUT;APP=OUT;FIGURES=OUT/'figures'
D=pd.read_csv(OUT/'cut_features.csv');M=pd.read_csv(OUT/'loocv_metrics.csv')
P=pd.read_csv(OUT/'loocv_predictions.csv');F=json.loads((OUT/'fitted_models.json').read_text())
SELN={'all':'Cuts 1,2,3','first':'Cut 1 only','later':'Cuts 2,3 separately'}
COLORS=['#19836f','#dfa141','#ce4c62'];LEVELS=[0,1,2,3,4,6,8]
LABELS={'directions_M3_phase_mean':'Closing + opening residual means','directions_M3_peak_to_peak':'Closing + opening residual ranges','closing_M3_phase_mean':'Closing residual mean','opening_M3_phase_mean':'Opening residual mean','cycle_M3_phase_mean':'Active-cycle residual mean','directions_M3_rms':'Closing + opening residual RMS','opening_M3_rms':'Opening residual RMS','filtered_peak_rc':'Filtered absolute current peak + slope 30–80%','filtered_peak':'Filtered absolute current peak','rc30_80':'Current slope 30–80%','original_directions_mean':'Native-sample closing + opening means','cycle_M3_peak_to_peak':'Active-cycle residual range','filtered_peak_closing_phase_mean':'Filtered absolute peak + closing residual mean'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':210})

def metric(s,sel,model,w=1):return M[(M.scenario==s)&(M.selection==sel)&(M.model==model)&(M.weight==w)].iloc[0]
def save(fig,name,folder=MAIN):
 folder=FIGURES;folder.mkdir(parents=True,exist_ok=True)
 temporary=folder/('.'+name+'.pending.png')
 fig.savefig(temporary,bbox_inches='tight');temporary.replace(folder/(name+'.png'))
 plt.close(fig)
def ct(r,k):return f'{int(r[f"class{k}_correct"])}/{int(r[f"class{k}_n"])}'
def tex(*args,**kwargs):
 pass

def tables():
 mainmodels=['filtered_peak','rc30_80','filtered_peak_rc','closing_M3_phase_mean','opening_M3_phase_mean','cycle_M3_phase_mean','filtered_peak_closing_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean']
 requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area']
 def binary_table(models):
  rows=[]
  for sel in SELECTIONS:
   for model in models:
    r=metric('healthy_all',sel,model)
    rows.append(dict(cut_selection=SELN[sel],model=model,LDA_input=LABELS.get(model,NAMES.get(model.replace('closing_M3_','').replace('closing_joint_',''),model)),healthy_correct=ct(r,0),specificity_pct=100*r.specificity,after_filing_correct=ct(r,1),sensitivity_pct=100*r.sensitivity,balanced_accuracy_pct=100*r.balanced_accuracy,correct=f'{int(r.correct)}/{int(r.n)}'))
  return pd.DataFrame(rows)
 t=binary_table(mainmodels);t.to_csv(MAIN/'healthy_vs_faulty_summary.csv',index=False)
 compact=t[['cut_selection','LDA_input','healthy_correct','specificity_pct','after_filing_correct','sensitivity_pct','balanced_accuracy_pct']].copy();compact.columns=['Cuts','LDA input','Healthy correct','Specificity','After filing correct','Sensitivity','Balanced accuracy']
 binary_table(requested).to_csv(MAIN/'feature_validation_summary.csv',index=False)
 tex(compact,MAIN/'healthy_vs_faulty_summary.tex',caption='Leave-one-complete-run-out LDA for individual scissor cuts. Healthy: iteration 0; after filing: iterations 1,2,3,4,6,8. Equal priors, automatic covariance shrinkage; every held-out cut is predicted once. Percentages summarize these small, exploratory data.',label='tab:scissors_binary')
 rows=[]
 for s in ['healthy_blunt','healthy_medium146','medium146_blunt','three146','three_all','three4']:
  for sel in SELECTIONS:
   for model in ['directions_M3_phase_mean','directions_M3_peak_to_peak','opening_M3_rms','filtered_peak_rc']:
    r=metric(s,sel,model);k=SCENARIOS[s][1]
    rows.append(dict(scenario=s,cut_selection=SELN[sel],model=model,parameters=LABELS[model],reference_correct=ct(r,0),middle_or_positive_correct=ct(r,1),blunt_correct=ct(r,2) if k==3 else '',correct=f'{int(r.correct)}/{int(r.n)}',balanced_accuracy_pct=100*r.balanced_accuracy,faulty_alarm_correct=f'{int(r.alarm_faulty_correct)}/{int(r.alarm_faulty_n)}' if s!='medium146_blunt' else '',missed_as_healthy=int(r.missed_as_reference) if s!='medium146_blunt' else None))
 t=pd.DataFrame(rows);t.to_csv(MAIN/'scenario_comparison.csv',index=False)
 tex(t,MAIN/'scenario_comparison.tex',caption='Direct one- or two-feature LDA. Class recall counts and balanced accuracy are calculated from predictions with entire runs held out. Medium groups are iteration-defined.',label='tab:scissors_scenarios')
 for s in ['healthy_all','three146','three_all','three4']:
  models=['directions_M3_phase_mean','directions_M3_peak_to_peak','opening_M3_rms','filtered_peak_rc']
  for sel in SELECTIONS:
   for model in models:
    g=P[(P.scenario==s)&(P.selection==sel)&(P.model==model)&(P.weight==1)]
    summary=g.assign(correct=(g.truth==g.prediction).astype(int),alarm_missed=((g.truth>0)&(g.prediction==0)).astype(int)).groupby(['iteration','run_number','run_id']).agg(cuts=('cut','size'),correct=('correct','sum'),alarm_missed=('alarm_missed','sum')).reset_index()
    summary['scenario']=s;summary['selection']=sel;summary['model']=model
    rowsfile=APP/'LOOCV_per_run.csv'
    if s=='healthy_all' and sel=='all' and model==models[0]:summary.to_csv(rowsfile,index=False)
    else:summary.to_csv(rowsfile,index=False,header=False,mode='a')
 sens=M[(M.scenario=='healthy_all')&(M.model=='directions_M3_phase_mean')].copy()
 sens.to_csv(MAIN/'alarm_threshold_comparison.csv',index=False)

def boxplots(cols,name,folder=MAIN):
 fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),layout='constrained')
 rng=np.random.default_rng(912)
 for ax,col in zip(axes.flat,cols):
  data=[D.loc[D.iteration==it,col].to_numpy() for it in LEVELS]
  bp=ax.boxplot(data,positions=np.arange(7),widths=.48,patch_artist=True,showfliers=False)
  for patch,it in zip(bp['boxes'],LEVELS):patch.set(facecolor=COLORS[0] if it==0 else COLORS[2] if it==8 else COLORS[1],alpha=.18)
  for i,it in enumerate(LEVELS):
   g=D[D.iteration==it]
   for rn,mk in [(1,'o'),(2,'^')]:
    v=g[g.run_number==rn];ax.scatter(i+rng.uniform(-.14,.14,len(v)),v[col],s=26,marker=mk,color=COLORS[0] if it==0 else COLORS[2] if it==8 else COLORS[1],edgecolor='white',linewidth=.4,zorder=3)
  stat=col.replace('closing_M3_','').replace('opening_M3_','').replace('closing_joint_','')
  label=NAMES.get(stat,{'closing_filtered_abs_peak':'Filtered absolute current peak','closing_current_rc30_80':'Slope 30–80% closing'}.get(col,col))
  if col.startswith(('closing_M3_','opening_M3_')) and any(x.startswith('opening_') for x in cols):
   label=('Closing' if col.startswith('closing_') else 'Opening')+' residual '+('mean' if stat=='phase_mean' else 'range' if stat=='peak_to_peak' else label.lower())
  unit=UNITS.get(stat,'mA/s' if 'rc30' in col else 'mA')
  ax.set(title=label,xlabel='Iteration',ylabel=f'[{unit}]',xticks=np.arange(7),xticklabels=LEVELS);ax.grid(axis='y',alpha=.12)
 fig.suptitle('One point = one cut; no averaging across cuts',fontweight='bold')
 fig.supxlabel('6 cuts per iteration (2 runs × 3 cuts); circle = run 1, triangle = run 2. Boxplots are descriptive.',fontsize=10)
 save(fig,name,folder)

def profiles():
 pr=pd.read_csv(OUT/'phase_profiles.csv')
 fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey='row')
 cols=['#548dad','#cf9450','#895fa4']
 for row,phase in enumerate(['closing','opening']):
  for j,it in enumerate([0,4,8]):
   ax=axs[row,j]
   for rn in [1,2]:
    for cut in [1,2,3]:
     g=pr[(pr.iteration==it)&(pr.run_number==rn)&(pr.cut==cut)&(pr.motor==3)&(pr.phase==phase)]
     ax.plot(g.phase_fraction*100,g.residual_ma,c=cols[cut-1],ls='-' if rn==1 else '--',lw=1,label=f'Cut {cut}' if rn==1 else None,alpha=.82)
   ax.set(title=f'{phase.capitalize()} · iteration {it}',xlabel='Valid direction duration [%]',ylabel='Motor 3 current residual [mA]');ax.axhline(0,c='#8e989d',lw=.7);ax.grid(alpha=.1)
   if row==0 and j==0:ax.legend(fontsize=8)
 fig.suptitle('Individual waveforms: event timing remains visible',fontweight='bold')
 fig.supxlabel('Time normalized within each valid movement; solid = run 1, dashed = run 2. No waveform averaging.',fontsize=10)
 save(fig,'01_individual_profiles')

def lda_plot(s,model,name):
 mod=next(x for x in F if x['scenario']==s and x['selection']=='all' and x['model']==model);cols=mod['parameters']
 ds,k=dataset(D,s,'all');xy=ds[cols].to_numpy();mins=xy.min(0);maxs=xy.max(0);span=np.maximum(maxs-mins,1);mins-=.09*span;maxs+=.12*span
 fig,axs=plt.subplots(1,3,figsize=(13.4,4.6),layout='constrained')
 xx=np.linspace(mins[0],maxs[0],220);yy=np.linspace(mins[1],maxs[1],160);X,Y=np.meshgrid(xx,yy);grid=np.c_[X.ravel(),Y.ravel()]
 for ax,sel in zip(axs,SELECTIONS):
  mod=next(x for x in F if x['scenario']==s and x['selection']==sel and x['model']==model);co=np.array(mod['coefficients']);b=np.array(mod['intercept']);score=grid@co.T+b
  pred=(score[:,0]>0).astype(int) if k==2 else score.argmax(1)
  palette=[COLORS[0],COLORS[2]] if k==2 else COLORS
  ax.contourf(X,Y,pred.reshape(X.shape),levels=np.arange(k+1)-.5,colors=palette,alpha=.14)
  pairs=[(None,None)] if k==2 else [(0,1),(0,2),(1,2)]
  for i,j in pairs:
   a=co[0] if k==2 else co[i]-co[j];bb=b[0] if k==2 else b[i]-b[j]
   if abs(a[1])>1e-14:bx=np.linspace(mins[0],maxs[0],1000);by=-(a[0]*bx+bb)/a[1]
   elif abs(a[0])>1e-14:by=np.linspace(mins[1],maxs[1],1000);bx=np.full(1000,-bb/a[0])
   else:continue
   valid=(bx>=mins[0])&(bx<=maxs[0])&(by>=mins[1])&(by<=maxs[1])
   if k==3:
    z=np.c_[bx,by]@co.T+b;valid&=z[:,3-i-j]<=z[:,i]+1e-8
   ax.plot(np.where(valid,bx,np.nan),np.where(valid,by,np.nan),c='#67757e',lw=.8)
  d,_=dataset(D,s,sel)
  for cl in range(k):
   for rn,mk in [(1,'o'),(2,'^')]:
    good=(d.truth==cl)&(d.run_number==rn)
    ax.scatter(d.loc[good,cols[0]],d.loc[good,cols[1]],marker=mk,c=palette[cl],s=38,label=SCENARIOS[s][2][cl] if rn==1 else None,edgecolor='white',linewidth=.6,zorder=3)
  isrange='peak_to_peak' in cols[0];word='range' if isrange else 'mean'
  ax.set(title=f'{SELN[sel]} · n={len(d)}',xlim=(mins[0],maxs[0]),ylim=(mins[1],maxs[1]),xlabel=f'Closing M3 residual {word} [mA]',ylabel=f'Opening M3 residual {word} [mA]');ax.grid(alpha=.1)
  if sel=='all':ax.legend(fontsize=7,loc='upper left')
 fig.suptitle(LABELS[model]+' · descriptive LDA fit',fontweight='bold')
 fig.supxlabel('Full-data boundaries illustrate the model; reported performance comes from separate leave-one-run-out fits.',fontsize=10)
 save(fig,name)

def performance():
 models=['filtered_peak_rc','closing_M3_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean']
 names=['Filtered absolute peak\n+ slope','Closing\nresidual mean','Closing + opening\nRMS','Closing + opening\nrange','Closing + opening\nmean']
 fig,ax=plt.subplots(figsize=(11.5,4.3),layout='constrained')
 colors=['#397c98','#d7a24c','#8665a8']
 for j,sel in enumerate(SELECTIONS):
  values=[100*metric('healthy_all',sel,m).balanced_accuracy for m in models]
  bars=ax.bar(np.arange(len(models))+(j-1)*.23,values,.22,color=colors[j],label=SELN[sel])
  for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=8)
 ax.set(xticks=range(len(models)),xticklabels=names,ylim=(0,112),yticks=[0,25,50,75,100],ylabel='Balanced accuracy [%]',title='Healthy (0) vs all after-filing iterations · run-wise validation');ax.legend(fontsize=9,loc='lower left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 save(fig,'04_binary_model_comparison')

def confusion():
 fig,axs=plt.subplots(2,3,figsize=(11,7.2),layout='constrained')
 for row,model in enumerate(['directions_M3_phase_mean','directions_M3_peak_to_peak']):
  for ax,sel in zip(axs[row],SELECTIONS):
   r=metric('three146',sel,model);cm=np.array(json.loads(r.confusion_matrix));norm=cm/cm.sum(1)[:,None]
   ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
   for i in range(3):
    for j in range(3):ax.text(j,i,f'{cm[i,j]} / {cm[i].sum()}',ha='center',va='center',fontsize=12,color='white' if norm[i,j]>.55 else '#263c4a')
   ax.set(title=f'{SELN[sel]} · BA {100*r.balanced_accuracy:.1f}%',xticks=range(3),yticks=range(3),xticklabels=['Healthy','Medium','Blunt'],yticklabels=['Healthy','Medium','Blunt'],xlabel='Predicted class',ylabel=('Mean pair' if row==0 else 'Range pair')+' · true class')
 fig.suptitle('Correct severity class differs from a correct fault alarm',fontweight='bold');fig.supxlabel('Medium = iterations 1,4,6. Counts are individual held-out cuts; errors in the lower-left column miss a fault.',fontsize=10)
 save(fig,'07_confusion_matrices')

def order_and_correlations():
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 cmap=plt.get_cmap('tab10');columns=['closing_M3_phase_mean','opening_M3_phase_mean','closing_M3_peak_to_peak','opening_M3_peak_to_peak']
 for ax,col in zip(axs.flat,columns):
  for i,it in enumerate(LEVELS):
   for rn in [1,2]:
    g=D[(D.iteration==it)&(D.run_number==rn)].sort_values('cut');ax.plot(g.cut,g[col],c=cmap(i),marker='o' if rn==1 else '^',ls='-' if rn==1 else '--',lw=.8,ms=4,label=str(it) if rn==1 else None)
  ax.set(title=col.replace('_M3_',' residual ').replace('_',' ').capitalize(),xticks=[1,2,3],xlabel='Cut order within run',ylabel='[mA]');ax.grid(alpha=.1)
 axs[0,0].legend(title='Iteration',ncol=4,fontsize=8)
 fig.suptitle('Within-run cut order: lines connect observations, not averages',fontweight='bold')
 save(fig,'cut_order',APP)
 cols=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area'];labels=[NAMES[m] for m in METRICS]+['M0–M3 correlation','M0–M3 ellipse area']
 for group,levels in [('healthy',[0]),('medium146',[1,4,6]),('blunt',[8])]:
  ds=D[D.iteration.isin(levels)];corr=ds[cols].corr();corr.to_csv(APP/f'Pearson_closing_{group}.csv')
  fig,ax=plt.subplots(figsize=(11.8,9.6),layout='constrained');im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
  for i in range(12):
   for j in range(12):
    v=corr.iloc[i,j];ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else '—',ha='center',va='center',fontsize=7,color='white' if abs(v)>.7 else '#243642')
  ax.set(xticks=range(12),yticks=range(12),xticklabels=labels,yticklabels=labels,title=f'Within-group Pearson correlations · {group} · n={len(ds)} cuts');ax.tick_params(axis='x',labelrotation=65);fig.colorbar(im,ax=ax,shrink=.7,label='Pearson r')
  fig.supxlabel('Exploratory associations; cuts in a run are related. This is not validation accuracy.',fontsize=10)
  save(fig,f'feature_correlations_{group}',APP)

def main():
 MAIN.mkdir(parents=True,exist_ok=True);APP.mkdir(parents=True,exist_ok=True)
 tables();profiles();boxplots(['closing_M3_phase_mean','opening_M3_phase_mean','closing_M3_peak_to_peak','opening_M3_peak_to_peak','closing_filtered_abs_peak','closing_current_rc30_80'],'02_main_features')
 requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area']
 boxplots(requested[:6],'additional_parameters_1',APP);boxplots(requested[6:],'additional_parameters_2',APP)
 lda_plot('healthy_all','directions_M3_phase_mean','03_LDA_binary_mean');performance()
 lda_plot('three146','directions_M3_phase_mean','05_LDA_three_class_mean');lda_plot('three146','directions_M3_peak_to_peak','06_LDA_three_class_range');confusion();order_and_correlations()
 # Complete matrices remain available without presenting a pooled correlation as evidence of detection.
 pd.concat([D[D.iteration.isin(levels)].select_dtypes('number').corr().assign(group=g) for g,levels in [('healthy',[0]),('medium146',[1,4,6]),('blunt',[8])]],keys=['healthy','medium146','blunt']).to_csv(APP/'feature_correlations_by_group.csv')
 print('PNG figures, main tables, correlation tables written.')
if __name__=='__main__':main()
