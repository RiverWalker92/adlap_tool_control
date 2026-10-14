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
LABELS={
 'directions_M3_phase_mean':'M3 closing + opening mean',
 'directions_M3_peak_to_peak':'M3 closing + opening range',
 'closing_M3_phase_mean':'M3 closing mean',
 'opening_M3_phase_mean':'M3 opening mean',
 'cycle_M3_phase_mean':'M3 active-cycle mean',
 'directions_M3_rms':'M3 closing + opening RMS',
 'opening_M3_rms':'M3 opening RMS',
 'opening_paired_phase_mean':'M0 + M3 opening mean',
 'closing_M3_abs_max':'M3 closing absolute maximum',
 'closing_M3_rc30_80':'M3 closing RC 30–80%',
 'closing_M0_rc30_80':'M0 closing RC 30–80%',
 'closing_paired_rc30_80':'M0 + M3 closing RC 30–80%',
 'closing_peak_rc':'M3 closing absolute maximum + RC 30–80%',
 'closing_peak_mean':'M3 closing absolute maximum + mean',
 'original_directions_mean':'M3 native-sample closing + opening mean',
 'cycle_M3_peak_to_peak':'M3 active-cycle range'}

def feature_label(col):
 if '_joint_' in col:
  phase,stat=col.split('_joint_')
  return f'{phase.capitalize()} {NAMES[stat]} [{UNITS[stat]}]'
 phase,motor,stat=col.split('_',2)
 label={'rc30_80':'RC 30–80%', 'phase_mean':'Mean', 'phase_sd':'SD',
        'harmonic1':'First Fourier amplitude','harmonic2':'Second Fourier amplitude'}.get(stat,NAMES.get(stat,stat))
 unit='mA/s' if stat=='rc30_80' else UNITS[stat]
 return f'{motor} {phase.replace("cycle","active cycle")} {label.lower()} [{unit}]'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':210})

def metric(s,sel,model,w=1):return M[(M.scenario==s)&(M.selection==sel)&(M.model==model)&(M.weight==w)].iloc[0]
def save(fig,name,folder=MAIN):
 folder=FIGURES;folder.mkdir(parents=True,exist_ok=True)
 temporary=folder/('.'+name+'.pending.png')
 fig.savefig(temporary,bbox_inches='tight');temporary.replace(folder/(name+'.png'))
 fig.savefig(folder/(name+'.svg'),bbox_inches='tight')
 plt.close(fig)
def ct(r,k):return f'{int(r[f"class{k}_correct"])}/{int(r[f"class{k}_n"])}'
def tex(frame,path,caption,label):
 def escape(x):
  return str(x).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
 header=' & '.join(escape(c) for c in frame.columns)+r' \\'
 rows=[' & '.join(escape(v) for v in row)+r' \\' for row in frame.itertuples(index=False,name=None)]
 spec='Xlrrrr' if len(frame.columns)==6 else 'Xllrrrr'
 path.write_text('\n'.join([r'\begin{table}[t]',r'\centering\scriptsize',
  r'\setlength{\tabcolsep}{3pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',
  r'\begin{tabularx}{\columnwidth}{@{}'+spec+r'@{}}',r'\toprule',header,
  r'\midrule',*rows,r'\bottomrule',r'\end{tabularx}',r'\end{table}']))

def tables():
 mainmodels=['closing_M3_abs_max','closing_M3_rc30_80','closing_peak_rc','closing_peak_mean','closing_M3_phase_mean','opening_M3_phase_mean','cycle_M3_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean','opening_paired_phase_mean','closing_paired_rc30_80']
 requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area','closing_M3_rc30_80']
 def binary_table(models):
  rows=[]
  for sel in SELECTIONS:
   for model in models:
    r=metric('healthy_all',sel,model)
    rows.append(dict(TP=int(r.TP),FN=int(r.FN),TN=int(r.TN),FP=int(r.FP),cut_selection=SELN[sel],model=model,LDA_input=LABELS.get(model,NAMES.get(model.replace('closing_M3_','').replace('closing_joint_',''),model)),healthy_correct=ct(r,0),specificity_pct=100*r.specificity,after_filing_correct=ct(r,1),sensitivity_pct=100*r.sensitivity,balanced_accuracy_pct=100*r.balanced_accuracy,correct=f'{int(r.correct)}/{int(r.n)}'))
  return pd.DataFrame(rows)
 t=binary_table(mainmodels);t.to_csv(MAIN/'healthy_vs_faulty_summary.csv',index=False)
 binary_table(requested).to_csv(MAIN/'feature_validation_summary.csv',index=False)
 compact=t[((t.model=='closing_peak_rc')&(t.cut_selection==SELN['all']))|t.model.isin(['directions_M3_phase_mean','opening_paired_phase_mean'])].copy()
 compact=compact[['LDA_input','cut_selection','TP','FN','TN','FP']]
 compact.columns=['LDA inputs','Cuts','TP','FN','TN','FP']
 compact['Cuts']=compact['Cuts'].replace({SELN['all']:'All',SELN['first']:'1',SELN['later']:'2,3'})
 compact.to_csv(MAIN/'scissors_binary_compact.csv',index=False)
 tex(compact,MAIN/'scissors_binary_compact.tex',caption='Run-wise cross-validation for healthy (0) versus after filing (1,2,3,4,6,8), the positive class. All inputs are current-residual parameters. Cuts 2 and 3 remain separate observations; no averaging. Standard LDA decision rule.',label='tab:scissors_binary_compact')
 rows=[]
 for s in ['healthy_blunt','healthy_medium146','medium146_blunt','three146','three_all','three4']:
  for sel in SELECTIONS:
   for model in ['directions_M3_phase_mean','directions_M3_peak_to_peak','opening_M3_rms','opening_paired_phase_mean','closing_peak_rc']:
    r=metric(s,sel,model);k=SCENARIOS[s][1]
    rows.append(dict(TP=r.TP,FN=r.FN,TN=r.TN,FP=r.FP,alarm_TP=int(r.alarm_TP),alarm_FN=int(r.alarm_FN),alarm_TN=int(r.alarm_TN),alarm_FP=int(r.alarm_FP),scenario=s,cut_selection=SELN[sel],model=model,parameters=LABELS[model],reference_correct=ct(r,0),middle_or_positive_correct=ct(r,1),blunt_correct=ct(r,2) if k==3 else '',correct=f'{int(r.correct)}/{int(r.n)}',balanced_accuracy_pct=100*r.balanced_accuracy,faulty_alarm_correct=f'{int(r.alarm_faulty_correct)}/{int(r.alarm_faulty_n)}' if s!='medium146_blunt' else '',missed_as_healthy=int(r.missed_as_reference) if s!='medium146_blunt' else None))
 t=pd.DataFrame(rows);t.to_csv(MAIN/'scenario_comparison.csv',index=False)
 selected=[]
 for scenario,medium in [('three_all','1,2,3,4,6'),('three146','1,4,6'),('three4','4')]:
  for sel,model in [('all','directions_M3_phase_mean'),('later','opening_paired_phase_mean')]:
   r=metric(scenario,sel,model)
   selected.append(dict(intermediate=medium,model=model,cut_selection=sel,healthy_correct=ct(r,0),intermediate_correct=ct(r,1),blunt_correct=ct(r,2),alarm_TP=int(r.alarm_TP),alarm_FN=int(r.alarm_FN),alarm_TN=int(r.alarm_TN),alarm_FP=int(r.alarm_FP)))
 severity=pd.DataFrame(selected);severity.to_csv(MAIN/'selected_threeclass_summary.csv',index=False)
 small=severity[['intermediate','model','cut_selection','healthy_correct','intermediate_correct','blunt_correct','alarm_FN']].copy()
 small['model']=small['model'].replace({'directions_M3_phase_mean':'A','opening_paired_phase_mean':'B'})
 small['cut_selection']=small['cut_selection'].replace({'all':'All','later':'2,3'})
 small.columns=['Intermediate','Inputs','Cuts','Healthy','Intermediate','Blunt','Fault FN']
 tex(small,MAIN/'scenario_comparison.tex',caption='Run-wise three-class validation. A: M3 closing and opening means. B: M0 and M3 opening means. All inputs describe current residuals. Class counts require the exact label; Fault FN counts intermediate or blunt cuts predicted as healthy. Cuts 2 and 3 remain separate.',label='tab:scissors_severity')
 for s in ['healthy_all','three146','three_all','three4']:
  models=['directions_M3_phase_mean','directions_M3_peak_to_peak','opening_M3_rms','opening_paired_phase_mean','closing_peak_rc']
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
 nrows=int(np.ceil(len(cols)/3))
 fig,axes=plt.subplots(nrows,3,figsize=(13.5,3.6*nrows),layout='constrained',squeeze=False)
 rng=np.random.default_rng(912)
 for ax,col in zip(axes.flat,cols):
  data=[D.loc[D.iteration==it,col].to_numpy() for it in LEVELS]
  bp=ax.boxplot(data,positions=np.arange(7),widths=.48,patch_artist=True,showfliers=False)
  for patch,it in zip(bp['boxes'],LEVELS):patch.set(facecolor=COLORS[0] if it==0 else COLORS[2] if it==8 else COLORS[1],alpha=.18)
  for i,it in enumerate(LEVELS):
   g=D[D.iteration==it]
   for rn,mk in [(1,'o'),(2,'^')]:
    v=g[g.run_number==rn];ax.scatter(i+rng.uniform(-.14,.14,len(v)),v[col],s=26,marker=mk,color=COLORS[0] if it==0 else COLORS[2] if it==8 else COLORS[1],edgecolor='white',linewidth=.4,zorder=3)
  label=feature_label(col)
  unit=label.rsplit('[',1)[1].rstrip(']')
  label=label.rsplit(' [',1)[0]
  ax.set(title=label,xlabel='Iteration',ylabel=f'[{unit}]',xticks=np.arange(7),xticklabels=LEVELS);ax.grid(axis='y',alpha=.12)
 fig.suptitle('One point = one cut; no averaging across cuts',fontweight='bold')
 for ax in axes.flat[len(cols):]:ax.set_visible(False)
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
   ax.set(title=f'{phase.capitalize()} · iteration {it}',xlabel='Valid direction duration [%]',ylabel='M3 [mA]');ax.axhline(0,c='#8e989d',lw=.7);ax.grid(alpha=.1)
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
  ax.set(title=f'{SELN[sel]} · n={len(d)}',xlim=(mins[0],maxs[0]),ylim=(mins[1],maxs[1]),xlabel=feature_label(cols[0]),ylabel=feature_label(cols[1]));ax.grid(alpha=.1)
  if sel=='all':ax.legend(fontsize=7,loc='upper left')
 fig.suptitle(LABELS[model]+' · descriptive LDA fit',fontweight='bold')
 fig.supxlabel('Full-data boundaries illustrate the model; reported performance comes from separate leave-one-run-out fits.',fontsize=10)
 save(fig,name)

def performance():
 models=['closing_peak_rc','closing_M3_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean','opening_paired_phase_mean']
 names=['M3 closing\npeak + RC','M3 closing\nmean','M3 closing + opening\nRMS','M3 closing + opening\nrange','M3 closing + opening\nmean','M0 + M3 opening\nmean']
 fig,axs=plt.subplots(2,1,figsize=(11.5,7),layout='constrained',sharex=True)
 for ax,count,total,title in zip(axs,['FN','FP'],['positive_n','class0_n'],['Missed after-filing cuts','Healthy cuts classified as after filing']):
  for j,sel in enumerate(SELECTIONS):
   results=[metric('healthy_all',sel,m) for m in models]
   values=[int(r[count]) for r in results]
   bars=ax.bar(np.arange(len(models))+(j-1)*.23,values,.22,color=['#397c98','#d7a24c','#8665a8'][j],label=SELN[sel])
   for bar,r in zip(bars,results):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.15,f'{int(r[count])}/{int(r[total])}',ha='center',fontsize=8)
  ax.set(ylabel=count,title=title,ylim=(0,max(1,max(int(metric('healthy_all',sel,m)[count]) for sel in SELECTIONS for m in models))+2))
  ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axs[0].legend(fontsize=8,ncol=3)
 axs[1].set(xticks=range(len(models)),xticklabels=names)
 fig.suptitle('Healthy vs after filing · run-wise cross-validation')
 save(fig,'04_binary_model_comparison')

def confusion():
 fig,axs=plt.subplots(2,3,figsize=(11,7.2),layout='constrained')
 for row,model in enumerate(['directions_M3_phase_mean','directions_M3_peak_to_peak']):
  for ax,sel in zip(axs[row],SELECTIONS):
   r=metric('three146',sel,model);cm=np.array(json.loads(r.confusion_matrix));norm=cm/cm.sum(1)[:,None]
   ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
   for i in range(3):
    for j in range(3):ax.text(j,i,f'{cm[i,j]} / {cm[i].sum()}',ha='center',va='center',fontsize=12,color='white' if norm[i,j]>.55 else '#263c4a')
   ax.set(title=f'{SELN[sel]} · correct {int(r.correct)}/{int(r.n)}',xticks=range(3),yticks=range(3),xticklabels=['Healthy','Medium','Blunt'],yticklabels=['Healthy','Medium','Blunt'],xlabel='Predicted class',ylabel=('Mean pair' if row==0 else 'Range pair')+' · true class')
 fig.suptitle('Correct severity class differs from a correct fault alarm',fontweight='bold');fig.supxlabel('Medium = iterations 1,4,6. Counts are individual held-out cuts; errors in the lower-left column miss a fault.',fontsize=10)
 save(fig,'07_confusion_matrices')

def order_and_correlations():
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 cmap=plt.get_cmap('tab10');columns=['closing_M3_phase_mean','opening_M3_phase_mean','closing_M3_peak_to_peak','opening_M3_peak_to_peak']
 for ax,col in zip(axs.flat,columns):
  for i,it in enumerate(LEVELS):
   for rn in [1,2]:
    g=D[(D.iteration==it)&(D.run_number==rn)].sort_values('cut');ax.plot(g.cut,g[col],c=cmap(i),marker='o' if rn==1 else '^',ls='-' if rn==1 else '--',lw=.8,ms=4,label=str(it) if rn==1 else None)
  ax.set(title=col.replace('_M3_',' ').replace('_',' ').capitalize(),xticks=[1,2,3],xlabel='Cut order within run',ylabel='[mA]');ax.grid(alpha=.1)
 axs[0,0].legend(title='Iteration',ncol=4,fontsize=8)
 fig.suptitle('Within-run cut order: lines connect observations, not averages',fontweight='bold')
 save(fig,'cut_order',APP)
 cols=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area','closing_M3_rc30_80'];labels=[NAMES[m] for m in METRICS]+['M0–M3 correlation','M0–M3 ellipse area','RC 30–80%']
 for group,levels in [('pooled',LEVELS),('healthy',[0]),('medium_all',[1,2,3,4,6]),('medium146',[1,4,6]),('blunt',[8])]:
  ds=D[D.iteration.isin(levels)];corr=ds[cols].corr();corr.to_csv(APP/f'Pearson_closing_{group}.csv')
  fig,ax=plt.subplots(figsize=(11.8,9.6),layout='constrained');im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
  for i in range(len(cols)):
   for j in range(len(cols)):
    v=corr.iloc[i,j];ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else '—',ha='center',va='center',fontsize=7,color='white' if abs(v)>.7 else '#243642')
  ax.set(xticks=range(len(cols)),yticks=range(len(cols)),xticklabels=labels,yticklabels=labels,title=f'Within-group Pearson correlations · {group} · n={len(ds)} cuts');ax.tick_params(axis='x',labelrotation=65);fig.colorbar(im,ax=ax,shrink=.7,label='Pearson r')
  fig.supxlabel('Exploratory associations; cuts in a run are related. This is not validation accuracy.',fontsize=10)
  save(fig,f'feature_correlations_{group}',APP)

def main():
 MAIN.mkdir(parents=True,exist_ok=True);APP.mkdir(parents=True,exist_ok=True)
 tables();profiles();boxplots(['closing_M3_phase_mean','opening_M3_phase_mean','closing_M3_peak_to_peak','opening_M3_peak_to_peak','closing_M3_abs_max','closing_M3_rc30_80'],'02_main_features')
 requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area','closing_M3_rc30_80']
 boxplots(requested[:6],'additional_parameters_1',APP);boxplots(requested[6:],'additional_parameters_2',APP)
 lda_plot('healthy_all','directions_M3_phase_mean','03_LDA_binary_mean');lda_plot('healthy_all','opening_paired_phase_mean','03b_LDA_binary_opening_pair');performance()
 lda_plot('three146','directions_M3_phase_mean','05_LDA_three_class_mean');lda_plot('three146','directions_M3_peak_to_peak','06_LDA_three_class_range');confusion();order_and_correlations()
 # Feature-only matrices exclude numeric run, cut and iteration identifiers.
 cols=pd.read_csv(OUT/'feature_definitions.csv').column.tolist()
 groups=[('pooled',LEVELS),('healthy',[0]),('medium_all',[1,2,3,4,6]),('medium146',[1,4,6]),('blunt',[8])]
 pd.concat([D[D.iteration.isin(levels)][cols].corr() for g,levels in groups],keys=[g for g,levels in groups]).to_csv(APP/'feature_correlations_by_group.csv')
 print('Residual-only PNG/SVG figures, CSV/LaTeX tables and correlation tables written.')
if __name__=='__main__':main()
