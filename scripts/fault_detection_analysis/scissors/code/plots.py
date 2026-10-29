"""Current scissor summaries and figures (PNG and CSV only)."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import *
from classify_cuts import SCENARIOS,SELECTIONS,dataset
from extract_features import METRICS,NAMES,UNITS
from combinations import summarise
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';MAIN=OUT/'tables';APP=MAIN;FIGURES=OUT/'figures'
SELN={'all':'All cuts','first':'Cut 1','later':'Cuts 2 and 3'}
COLORS=THREE_COLOURS
LEVELS = [0, 1, 2, 3, 4, 6, 8]

STROKE_MAP = {
    0: 0,
    1: 2,
    2: 4,
    3: 6,
    4: 8,
    6: 12,
    8: 16,
}
LABELS={'directions_M3_phase_mean':'M3 closing and opening mean',
'directions_M3_peak_to_peak':'M3 closing and opening range',
'opening_paired_phase_mean':'M0 and M3 opening mean',
'closing_peak_rc':'M3 closing absolute maximum and RC',
'closing_M3_phase_mean':'M3 closing mean','opening_M3_phase_mean':'M3 opening mean',
'cycle_M3_phase_mean':'M3 active-cycle mean','directions_M3_rms':'M3 closing and opening RMS',
'opening_M3_rms':'M3 opening RMS'}
D=M=P=None
CACHE={}
def save(fig,name,folder=None):save_png(fig,FIGURES/name)
def feature_label(col):
 if '_joint_' in col:
  phase,stat=col.split('_joint_')
  return f'{phase.capitalize()} {NAMES[stat]} [{UNITS[stat]}]'
 phase,motor,stat=col.split('_',2)
 label={'rc30_80':'RC 30–80%', 'phase_mean':'Mean', 'phase_sd':'SD',
        'harmonic1':'Fundamental amplitude','harmonic2':'Second harmonic amplitude'}.get(stat,NAMES.get(stat,stat))
 unit='mA/s' if stat=='rc30_80' else UNITS[stat]
 return f'{motor} {phase.replace("cycle","active cycle")} {label.lower().replace("peak-to-peak range","range")} [{unit}]'


def metric(s,sel,model,w=1):return M[(M.scenario==s)&(M.selection==sel)&(M.model==model)&(M.weight==w)].iloc[0]


def ct(r,k):return f'{int(r[f"class{k}_correct"])}/{int(r[f"class{k}_n"])}'


def tables():
 mainmodels=['closing_M3_abs_max','closing_M3_rc30_80','closing_peak_rc','closing_peak_mean','closing_M3_phase_mean','opening_M3_phase_mean','cycle_M3_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean','opening_paired_phase_mean','closing_paired_rc30_80']
 requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area','closing_M3_rc30_80']
 def binary_table(models):
  rows=[]
  for sel in SELECTIONS:
   for model in models:
    r=metric('healthy_all',sel,model)
    rows.append(dict(TP=int(r.TP),FN=int(r.FN),TN=int(r.TN),FP=int(r.FP),cut_selection=SELN[sel],model=model,LDA_input=LABELS.get(model,NAMES.get(model.replace('closing_M3_','').replace('closing_joint_',''),model)),healthy_correct=ct(r,0),specificity_pct=100*r.specificity,after_abrasion_correct=ct(r,1),sensitivity_pct=100*r.sensitivity,balanced_accuracy_pct=100*r.balanced_accuracy,correct=f'{int(r.correct)}/{int(r.n)}'))
  return pd.DataFrame(rows)
 t=binary_table(mainmodels);t.to_csv(MAIN/'healthy_vs_faulty_summary.csv',index=False)
 binary_table(requested).to_csv(MAIN/'feature_validation_summary.csv',index=False)
 compact=t[((t.model=='closing_peak_rc')&(t.cut_selection==SELN['all']))|t.model.isin(['directions_M3_phase_mean','opening_paired_phase_mean'])].copy()
 compact=compact[['LDA_input','cut_selection','TP','FN','TN','FP']]
 compact.columns=['LDA inputs','Cuts','TP','FN','TN','FP']
 compact['Cuts']=compact['Cuts'].replace({SELN['all']:'All',SELN['first']:'1',SELN['later']:'2,3'})
 compact.to_csv(MAIN/'scissors_binary_compact.csv',index=False)
 rows=[]
 for s in ['healthy_blunt','healthy_medium146','medium146_blunt','three146','three_all','three4']:
  for sel in SELECTIONS:
   for model in ['directions_M3_phase_mean','directions_M3_peak_to_peak','opening_M3_rms','opening_paired_phase_mean','closing_peak_rc']:
    r=metric(s,sel,model);k=SCENARIOS[s][1]
    rows.append(dict(TP=r.TP,FN=r.FN,TN=r.TN,FP=r.FP,alarm_TP=int(r.alarm_TP),alarm_FN=int(r.alarm_FN),alarm_TN=int(r.alarm_TN),alarm_FP=int(r.alarm_FP),scenario=s,cut_selection=SELN[sel],model=model,parameters=LABELS[model],reference_correct=ct(r,0),middle_or_positive_correct=ct(r,1),blunt_correct=ct(r,2) if k==3 else '',correct=f'{int(r.correct)}/{int(r.n)}',balanced_accuracy_pct=100*r.balanced_accuracy,faulty_alarm_correct=f'{int(r.alarm_faulty_correct)}/{int(r.alarm_faulty_n)}' if s!='medium146_blunt' else '',missed_as_healthy=int(r.missed_as_reference) if s!='medium146_blunt' else None))
 t=pd.DataFrame(rows);t.to_csv(MAIN/'scenario_comparison.csv',index=False)
 selected=[]
 for scenario,medium in [('three_all','2,4,6,8,12'),('three146','2,8,12'),('three4','8')]:
  for sel,model in [('all','directions_M3_phase_mean'),('later','opening_paired_phase_mean')]:
   r=metric(scenario,sel,model)
   selected.append(dict(intermediate=medium,model=model,cut_selection=sel,healthy_correct=ct(r,0),intermediate_correct=ct(r,1),blunt_correct=ct(r,2),alarm_TP=int(r.alarm_TP),alarm_FN=int(r.alarm_FN),alarm_TN=int(r.alarm_TN),alarm_FP=int(r.alarm_FP)))
 severity=pd.DataFrame(selected);severity.to_csv(MAIN/'selected_threeclass_summary.csv',index=False)
 small=severity[['intermediate','model','cut_selection','healthy_correct','intermediate_correct','blunt_correct','alarm_FN']].copy()
 small['model']=small['model'].replace({'directions_M3_phase_mean':'A','opening_paired_phase_mean':'B'})
 small['cut_selection']=small['cut_selection'].replace({'all':'All','later':'2,3'})
 small.columns=['Intermediate','Inputs','Cuts','Healthy','Intermediate','Blunt','Fault FN']
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


def boxplots(cols, name, folder=MAIN):
    nrows = int(np.ceil(len(cols) / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 2.7 * nrows + .35),
                             layout='constrained', squeeze=False)
    rng = np.random.default_rng(912)
    for ax, col in zip(axes.flat, cols):
        data = [D.loc[D.iteration == it, col].to_numpy() for it in LEVELS]
        bp = ax.boxplot(data, positions=np.arange(len(LEVELS)), widths=.48,
                        patch_artist=True, showfliers=False)
        for patch, it in zip(bp['boxes'], LEVELS):
            patch.set(facecolor=COLORS[0] if it == 0 else COLORS[2] if it == 8 else COLORS[1], alpha=.18)
        for i, it in enumerate(LEVELS):
            g = D[D.iteration == it]
            for rn, mk in [(1, 'o'), (2, 's')]:
                v = g[g.run_number == rn]
                ax.scatter(i + rng.uniform(-.14, .14, len(v)), v[col], s=26,
                           marker=mk, color=COLORS[0] if it == 0 else COLORS[2] if it == 8 else COLORS[1],
                           edgecolor='white', linewidth=.4, zorder=3)
        label = feature_label(col)
        unit = label.rsplit('[', 1)[1].rstrip(']')
        label = label.rsplit(' [', 1)[0]
        ax.set(title=label, xlabel='Abrasive strokes per blade', ylabel=f'[{unit}]',
               xticks=np.arange(len(LEVELS)), xticklabels=[STROKE_MAP[it] for it in LEVELS])
        ax.grid(axis='y', alpha=.12)
    fig.suptitle('Current-residual parameters', fontweight='normal')
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.supxlabel('Circle: run 1; square: run 2', fontsize=8)
    save(fig, name, folder)


def profiles():
    pr = pd.read_csv(OUT / 'phase_profiles.csv')
    fig, axs = plt.subplots(2, 3, figsize=(9, 5.4), layout='constrained', sharey='row')
    cols = [BLUE, ORANGE, GREY]
    for row, phase in enumerate(['closing', 'opening']):
        for j, it in enumerate([0, 4, 8]):
            ax = axs[row, j]
            for rn in [1, 2]:
                for cut in [1, 2, 3]:
                    g = pr[(pr.iteration == it) & (pr.run_number == rn) &
                           (pr.cut == cut) & (pr.motor == 3) & (pr.phase == phase)]
                    ax.plot(g.phase_fraction * 100, g.residual_ma, c=cols[cut - 1],
                            ls='-' if rn == 1 else '--', lw=1,
                            label=f'Cut {cut}' if rn == 1 else None, alpha=.82)
            ax.set(title=f'{phase.capitalize()} · {STROKE_MAP[it]} abrasive strokes',
                   xlabel='Valid direction duration [%]', ylabel='M3 [mA]')
            ax.axhline(0, c='#8e989d', lw=.7)
            ax.grid(alpha=.1)
            if row == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('M3 current-residual profiles', fontweight='normal')
    fig.supxlabel('Time normalized within each valid movement; solid = run 1, dashed = run 2. No waveform averaging.', fontsize=10)
    save(fig, '01_individual_profiles')


def performance():
 models=['closing_peak_rc','closing_M3_phase_mean','directions_M3_rms','directions_M3_peak_to_peak','directions_M3_phase_mean','opening_paired_phase_mean']
 names=['M3 closing\npeak + RC','M3 closing\nmean','M3 closing + opening\nRMS','M3 closing + opening\nrange','M3 closing + opening\nmean','M0 + M3 opening\nmean']
 fig,axs=plt.subplots(2,1,figsize=(9,5.5),layout='constrained',sharex=True)
 for ax,count,total,title in zip(axs,['FN','FP'],['positive_n','class0_n'],['Missed after abrasion cuts','Healthy cuts classified as after abrasion']):
  for j,sel in enumerate(SELECTIONS):
   results=[metric('healthy_all',sel,m) for m in models]
   values=[int(r[count]) for r in results]
   bars=ax.bar(np.arange(len(models))+(j-1)*.23,values,.22,color=[BLUE,YELLOW,GREY][j],label=SELN[sel])
   for bar,r in zip(bars,results):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.15,f'{int(r[count])}/{int(r[total])}',ha='center',fontsize=8)
  ax.set(ylabel=count,title=title,ylim=(0,max(1,max(int(metric('healthy_all',sel,m)[count]) for sel in SELECTIONS for m in models))+2))
  ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axs[0].legend(fontsize=8,ncol=3)
 axs[1].set(xticks=range(len(models)),xticklabels=names)
 fig.suptitle('Healthy vs after abrasion · run-wise cross-validation')
 save(fig,'04_binary_model_comparison')


def order_and_correlations():
    fig, axs = plt.subplots(2, 2, figsize=(8, 5.6), layout='constrained')
    colours = [BLUE, YELLOW, '#5D9E8A', '#A9829A', '#80689E', '#8B8855', ORANGE]
    cmap = lambda i: colours[i]
    columns = ['closing_M3_phase_mean', 'opening_M3_phase_mean',
               'closing_M3_peak_to_peak', 'opening_M3_peak_to_peak']
    for ax, col in zip(axs.flat, columns):
        for i, it in enumerate(LEVELS):
            for rn in [1, 2]:
                g = D[(D.iteration == it) & (D.run_number == rn)].sort_values('cut')
                ax.plot(g.cut, g[col], c=cmap(i), marker='o' if rn == 1 else 's',
                        ls='-' if rn == 1 else '--', lw=.8, ms=4,
                        label=str(STROKE_MAP[it]) if rn == 1 else None)
        ax.set(title=col.replace('_M3_', ' ').replace('_', ' ').capitalize(),
               xticks=[1, 2, 3], xlabel='Cut order within run', ylabel='[mA]')
        ax.grid(alpha=.1)
    axs[0, 0].legend(title='Abrasive strokes per blade', ncol=4, fontsize=8)
    fig.suptitle('Within-run cut order: lines connect observations, not averages', fontweight='normal')
    save(fig, 'cut_order', APP)

    cols = [f'closing_M3_{m}' for m in METRICS] + [
        'closing_joint_wave_corr', 'closing_joint_ellipse_area', 'closing_M3_rc30_80']
    labels = [NAMES[m] for m in METRICS] + [
        'M0–M3 correlation', 'M0–M3 ellipse area', 'RC 30–80%']
    for group, levels in [
        ('pooled', LEVELS),
        ('healthy', [0]),
        ('medium_all', [1, 2, 3, 4, 6]),
        ('medium146', [1, 4, 6]),
        ('blunt', [8]),
    ]:
        ds = D[D.iteration.isin(levels)]
        corr = ds[cols].corr()
        corr.to_csv(APP / f'Pearson_closing_{group}.csv')
        fig, ax = plt.subplots(figsize=(9.2, 7.6), layout='constrained')
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.iloc[i, j]
                ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '—',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > .7 else '#243642')
        ax.set(xticks=range(len(cols)), yticks=range(len(cols)),
               xticklabels=labels, yticklabels=labels,
               title=f'Within-group Pearson correlations · {group} · n={len(ds)} cuts')
        ax.tick_params(axis='x', labelrotation=65)
        fig.colorbar(im, ax=ax, shrink=.7, label='Pearson r')
        fig.supxlabel('Exploratory associations; cuts in a run are related. This is not validation accuracy.', fontsize=10)
        save(fig, f'feature_correlations_{group}', APP)


def closing_profiles():
    frame=pd.read_csv(OUT/'phase_profiles.csv')
    fig,axes=plt.subplots(2,4,figsize=(10,5.3));fig.subplots_adjust(left=.07,right=.99,bottom=.12,top=.83,hspace=.55,wspace=.35)
    cut_colours=[BLUE,ORANGE,GREY]
    iteration_keys = [0, 1, 2, 3, 4, 6, 8]
    stroke_labels  = [0, 2, 4, 6, 8, 12, 16]

    for ax, iteration, stroke_label in zip(axes.flat, iteration_keys, stroke_labels):
        for run in [1,2]:
            for cut in [1,2,3]:
                q = frame[
                    (frame.motor == 3)
                    & (frame.phase == 'closing')
                    & (frame.iteration == iteration)
                    & (frame.run_number == run)
                    & (frame.cut == cut)
                ]
                ax.plot(
                    q.phase_fraction * 100,
                    q.residual_ma,
                    color=cut_colours[cut-1],
                    ls='-' if run == 1 else '--',
                    lw=.8,
                    alpha=.85
                )

        ax.set(
            title=f'Abrasive strokes per blade: {stroke_label}',
            xlabel='Closing duration [%]',
            ylabel='M3 residual [mA]'
        )
        plt.grid(ax)
    axes.flat[-1].axis('off')
    handles=[Line2D([],[],color=c,lw=1,label=f'Cut {i+1}') for i,c in enumerate(cut_colours)]
    handles += [Line2D([],[],color=GREY,ls=style,label=f'Run {i+1}') for i,style in enumerate(['-','--'])]
    figure_legend(fig,handles,3);save(fig,'closing_profiles_all_iterations')

def panel(ax,scenario,selection,model,title):
    source=OUT/'combinations'
    if not CACHE:
        CACHE.update(registry=pd.read_csv(source/'model_registry.csv').fillna(''),
            full=np.load(source/'full_models.npz'),cv=np.load(source/'cv_decisions.npz'),
            metrics=pd.read_csv(source/'cv_metrics.csv').set_index(['scenario','selection','model']))
    reg=CACHE['registry']
    index=reg.index[reg.model==model][0];row=reg.loc[index]
    cols=[row.feature1]+([row.feature2] if row.feature2 else [])
    full=CACHE['full'];cv=CACHE['cv']
    key=f'{scenario}__{selection}';ids=cv[key+'__cycle_id'].astype(str)
    d=D.set_index('cycle_id').loc[ids].reset_index();truth=cv[key+'__truth'];pred=cv[key+'__prediction'][:,index]
    k=int(truth.max())+1
    co=full[key+'__coef'][index].copy();bias=full[key+'__intercept'][index]
    x=d[cols[0]].to_numpy()
    single=len(cols)==1
    y=truth+(d.cut.to_numpy()-2)*.06+(d.run_number.to_numpy()-1.5)*.04 if single else d[cols[1]].to_numpy()
    lim=list(limits(x,y))
    if single:co[:,1]=0;lim[2:]=[-.35,k-.65]
    lda_regions(ax,co,bias,lim)
    palette=CLASS_COLOURS if k==2 else THREE_COLOURS
    for cls in range(k):
        for run,marker in [(1,'o'),(2,'s')]:
            mask=(truth==cls)&(d.run_number.to_numpy()==run)
            ax.scatter(x[mask],y[mask],color=palette[cls],marker=marker,s=28,edgecolors='white',linewidths=.4,zorder=3)
    ax.set(title=title,xlabel=feature_label(cols[0]),ylabel='Recorded class' if single else feature_label(cols[1]))
    if single:ax.set_yticks(range(k),['H','F'] if k==2 else ['H','I','B'])
    ax.set_xlabel(ax.get_xlabel().replace(' signed maximum','\nsigned maximum'))
    cm=np.zeros((k,k),dtype=int);np.add.at(cm,(truth,pred),1)
    expected=CACHE['metrics'].loc[(scenario,selection,model)]
    assert cm.tolist()==json.loads(expected.confusion_matrix)
    counts=cm.sum(1)
    return dict(scenario=scenario,selection=selection,model=model,input1=feature_label(cols[0]),input2=feature_label(cols[1]) if len(cols)>1 else '',n=len(d),
        intermediate_iterations={'three_all':'2,4,6,8,12','three146':'2,8,12','three4':'8'}.get(scenario,''),
        healthy=f'{cm[0,0]}/{counts[0]}',intermediate=f'{cm[1,1]}/{counts[1]}' if k==3 else '',
        blunt=f'{cm[2,2]}/{counts[2]}' if k==3 else '',TP=int(cm[1:,1:].sum()),FN=int(cm[1:,0].sum()),
        TN=int(cm[0,0]),FP=int(cm[0,1:].sum()),correct=int(np.trace(cm)),
        balanced_accuracy=float(np.mean(np.diag(cm)/counts)),confusion_matrix=json.dumps(cm.tolist()))

def lda_figures():
    source=OUT/'combinations';best=pd.read_csv(source/'best_models_summary.csv');records=[]
    binary=best[(best.scenario=='healthy_all')&(best.objective=='fault_detection')].set_index('selection')
    binary.reset_index().to_csv(MAIN/'binary_selected_counts.csv',index=False)
    current=M[(M.scenario=='healthy_all')&(M.selection=='later')&(M.weight==1)&M.model.isin([
        'closing_M3_phase_mean','opening_M3_phase_mean','cycle_M3_phase_mean',
        'directions_M3_phase_mean','opening_paired_phase_mean'])].copy()
    extra=best[(best.scenario=='healthy_all')&(best.selection=='later')&(best.objective=='fault_detection')]
    definitions={r['model']:r['parameters'] for r in json.loads((OUT/'model_registry.json').read_text())}
    rows=[]
    for _,r in current.iterrows():
        cols=definitions[r.model]
        rows.append(dict(model=r.model,input1=feature_label(cols[0]),input2=feature_label(cols[1]) if len(cols)>1 else '',
            cuts='2,3',n=int(r.n),TP=int(r.TP),FN=int(r.FN),TN=int(r.TN),FP=int(r.FP)))
    r=extra.iloc[0]
    rows.append(dict(model=r.model,input1=feature_label(r.feature1),input2=feature_label(r.feature2),
        cuts='2,3',n=int(r.n),TP=int(r.TP),FN=int(r.FN),TN=int(r.TN),FP=int(r.FP)))
    pd.DataFrame(rows).to_csv(MAIN/'binary_model_counts.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.5,3.1));fig.subplots_adjust(left=.18,right=.97,bottom=.18,top=.74)
    records.append(panel(ax,'healthy_all','later',binary.loc['later','model'],'Healthy versus after abrasion: cuts 2 and 3'))
    class_legend(fig,binary=True);save(fig,'scissors_binary_cuts23')
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.2));fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.73,wspace=.65)
    for ax,selection in zip(axes,SELECTIONS):
        panel(ax,'healthy_all',selection,binary.loc[selection,'model'],SELN[selection])
    class_legend(fig,binary=True);save(fig,'scissors_binary_cut_selections')
    scenarios={'three_all':('F0722','(a) Intermediate: 2,4,6,8,12'),
               'three146':('F0716','(b) Intermediate: 2,8,12'),
               'three4':('F1092','(c) Intermediate: 8')}
    comparisons=[]
    for selection,name in [('all','scissors_severity_selections'),('later','scissors_severity_selections_cuts23')]:
        fig,axes=plt.subplots(1,3,figsize=(7.16,3.2));fig.subplots_adjust(left=.10,right=.99,bottom=.25,top=.73,wspace=.55)
        for ax,(scenario,(model,title)) in zip(axes,scenarios.items()):
            r=panel(ax,scenario,selection,model,title);records.append(r);comparisons.append(r)
            ax.set_ylabel(ax.get_ylabel().replace(' fundamental amplitude',' fundamental\namplitude'))
        class_legend(fig);save(fig,name)
        fig,axes=plt.subplots(1,3,figsize=(7.16,2.55));fig.subplots_adjust(left=.09,right=.98,top=.82,bottom=.22,wspace=.35)
        for ax,r in zip(axes,comparisons[-3:]):
            cm=np.array(json.loads(r['confusion_matrix']));norm=cm/cm.sum(1)[:,None]
            ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
            for i in range(3):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.55 else '#222',fontsize=9)
            label={'three_all':'2,4,6,8,12','three146':'2,8,12','three4':'8'}[r['scenario']]
            ax.set(title='Intermediate: '+label,xticks=range(3),yticks=range(3),xticklabels=['H','I','B'],yticklabels=['H','I','B'],xlabel='Predicted class',ylabel='Recorded class')
        save(fig,'severity_confusion_'+selection)
    pd.DataFrame(comparisons).to_csv(MAIN/'severity_cut_selection_comparison.csv',index=False)
    pd.DataFrame(records).to_csv(MAIN/'plotted_models.csv',index=False)
    if (source/'nested_metrics.csv').exists():shutil.copy2(source/'nested_metrics.csv',MAIN/'nested_selection_counts.csv')

def run(root=ROOT):
    global ROOT,OUT,MAIN,APP,FIGURES,D,M,P
    ROOT=Path(root);OUT=ROOT/'outputs';MAIN=OUT/'tables';APP=MAIN;FIGURES=OUT/'figures'
    CACHE.clear()
    MAIN.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True);apply_style()
    D=pd.read_csv(OUT/'cut_features.csv');M=pd.read_csv(OUT/'loocv_metrics.csv');P=pd.read_csv(OUT/'loocv_predictions.csv')
    summarise(ROOT);tables();profiles();closing_profiles()
    boxplots(['closing_M3_phase_mean','opening_M3_phase_mean','closing_M3_peak_to_peak','opening_M3_peak_to_peak','closing_M3_abs_max','closing_M3_rc30_80'],'02_main_features')
    requested=[f'closing_M3_{m}' for m in METRICS]+['closing_joint_wave_corr','closing_joint_ellipse_area','closing_M3_rc30_80']
    boxplots(requested[:6],'additional_parameters_1');boxplots(requested[6:],'additional_parameters_2')
    performance();order_and_correlations();lda_figures()
    cols=pd.read_csv(OUT/'feature_definitions.csv').column.tolist()
    groups=[('pooled',LEVELS),('healthy',[0]),('medium_all',[1,2,3,4,6]),('medium146',[1,4,6]),('blunt',[8])]
    pd.concat([D[D.iteration.isin(levels)][cols].corr() for _,levels in groups],
        keys=[name for name,_ in groups]).to_csv(OUT/'feature_correlations_by_group.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);run(p.parse_args().root)