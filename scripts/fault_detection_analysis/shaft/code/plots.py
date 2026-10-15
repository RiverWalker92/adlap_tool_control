"""Shaft distributions, correlations and LDA figures (PNG only)."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shaft_features as f
from shaft_analysis import estimator,models
from plot_style import *
RED=ORANGE
COLORS=CLASS_COLOURS
SHORT={'M1_absolute_maximum':'M1 absolute peak','M1_rms':'M1 RMS','M2_rms':'M2 RMS',
 'M2_mean':'M2 phase mean','M2_a2':'M2 second harmonic amplitude','M2_rc':'M2 RC coefficient',
 'gain_difference':'Absolute gain difference','gain_positive':'Positive gain','gain_negative':'Negative gain',
 'video_mae':'Centred video MAE','video_relative_mae':'Relative video MAE'}
TITLES={'small':'Small amplitude','medium':'Medium amplitude','full':'Full amplitude'}
SELECT=['M1_absolute_maximum','M1_rms','M2_mean','M2_rms','M2_a1','M2_a2','M2_rc',
 'gain_positive','gain_negative','gain_difference','video_mae','video_relative_mae']
def save(fig,path):save_png(fig,path)
def info(frame):return run_info(frame)
def legend(fig,frame):run_legend(fig,frame,boundary=False,ncol=4)
def one_box(ax,frame,feature):
    defs=f.definitions().set_index('feature');unit=defs.loc[feature,'unit']
    for cls in [0,1]:
        v=frame.loc[frame.true_class==cls,feature].to_numpy()
        ax.boxplot([v],positions=[cls],widths=.6,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':COLORS[cls],'alpha':.13,'edgecolor':'#444'},
                   medianprops={'color':'#222'},whiskerprops={'color':'#444'},capprops={'color':'#444'})
    for r,(_,marker,_) in info(frame).items():
        part=frame[frame.run==r];cls=int(part.true_class.iloc[0]);n=len(part)
        jitter=np.linspace(-.16,.16,n) if n>1 else np.array([0.])
        ax.scatter(cls+jitter,part[feature],c=COLORS[cls],marker=marker,s=13,alpha=.8,zorder=4,linewidths=.25)
    ax.set_xticks([0,1],['Healthy','Degraded']);ax.set_ylabel(unit);ax.set_xlim(-.65,1.65)
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.set_title(SHORT.get(feature,defs.loc[feature,'label']),pad=5)


def selected_boxes(frame,path):
    features=['M1_absolute_maximum','M2_mean','M2_a2','M2_rc','gain_difference','video_relative_mae']
    fig,axes=plt.subplots(2,3,figsize=(10,6.2));legend(fig,frame)
    for ax,feature in zip(axes.flat,features):
        defs=f.definitions().set_index('feature')
        for j,rg in enumerate(f.RANGES):
            part=frame[frame['range']==rg]
            for cls in [0,1]:
                x=j*3+cls
                ax.boxplot([part.loc[part.true_class==cls,feature]],positions=[x],widths=.65,showfliers=False,
                   patch_artist=True,boxprops={'facecolor':COLORS[cls],'alpha':.13,'edgecolor':'#444'},
                   medianprops={'color':'#222'},whiskerprops={'color':'#444'},capprops={'color':'#444'})
            for r,(_,marker,_) in info(frame).items():
                p=part[part.run==r];cls=int(p.true_class.iloc[0]);ax.scatter(j*3+cls+np.linspace(-.17,.17,len(p)),p[feature],
                                  c=COLORS[cls],marker=marker,s=15,alpha=.85,zorder=4,linewidths=.25)
        ax.set_xticks([.5,3.5,6.5],['Small','Medium','Full']);ax.set_xlim(-.7,7.7)
        ax.set_ylabel(defs.loc[feature,'unit']);ax.set_title(SHORT.get(feature,feature));ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.subplots_adjust(top=.90,hspace=.42,wspace=.37,bottom=.08);save(fig,path)


def all_boxes(frame,features,path):
    nr=int(np.ceil(len(features)/3));fig,axes=plt.subplots(nr,3,figsize=(9,nr*2.2+.45));legend(fig,frame)
    for ax,feature in zip(axes.flat,features):one_box(ax,frame,feature)
    for ax in list(axes.flat)[len(features):]:ax.axis('off')
    fig.subplots_adjust(top=.92,hspace=.52,wspace=.4,bottom=.06);save(fig,path)


def matrix(corr,features,path,title):
    c=corr.loc[features,features].to_numpy();n=len(features);large=n>15
    fig,ax=plt.subplots(figsize=(11,9) if large else (9.2,7.6))
    mask=np.triu(np.ones((n,n),bool),k=1);c=np.ma.array(c,mask=mask)
    im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r');labels=[SHORT.get(x,x.replace('_',' ')) for x in features]
    ax.set_xticks(range(n),labels,rotation=55,ha='right',fontsize=6 if large else 8)
    ax.set_yticks(range(n),labels,fontsize=6 if large else 8)
    if not large:
        for i in range(n):
            for j in range(i+1):
                v=c[i,j]
                if not np.ma.is_masked(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.5,color='white' if abs(v)>.6 else '#222')
    fig.colorbar(im,ax=ax,shrink=.73,label='Pearson feature correlation');ax.set_title(title,pad=13)
    ax.spines[['top','right','left','bottom']].set_visible(False);save(fig,path)



def plane(ax,frame,columns,fit=None,alpha=1):
    X=frame[columns].to_numpy();fit=estimator().fit(X,frame.true_class) if fit is None else fit
    co,bias=raw_coefficients(fit);lda_regions(ax,co,bias,limits(X[:,0],X[:,1],.13))
    for run,(cls,marker,name) in info(frame).items():
        q=frame[frame.run==run];ax.scatter(q[columns[0]],q[columns[1]],color=COLORS[cls],marker=marker,
            s=25,edgecolors='white',linewidths=.4,alpha=alpha,zorder=3)
    defs=f.definitions().set_index('feature')
    for setter,col in [(ax.set_xlabel,columns[0]),(ax.set_ylabel,columns[1])]:
        label=SHORT.get(col,defs.loc[col,'label'])
        setter(label+' ['+defs.loc[col,'unit']+']')
    return co,bias

def pooled(frame,path,title):
    fig,ax=plt.subplots(figsize=(3.5,3.1));fig.subplots_adjust(left=.18,right=.97,bottom=.17,top=.76)
    plane(ax,frame,['M1_absolute_maximum','M2_rms']);ax.set_title(title)
    run_legend(fig,frame,ncol=3);save(fig,path)

def lda_planes(frame,path):
    fig,axes=plt.subplots(2,3,figsize=(7.16,5.3));fig.subplots_adjust(top=.84,bottom=.13,wspace=.55,hspace=.6)
    for j,rg in enumerate(f.RANGES):
        q=frame[frame['range']==rg]
        for i,cols in enumerate([['M1_absolute_maximum','M2_rms'],['M2_a2','gain_difference']]):
            plane(axes[i,j],q,cols)
            axes[i,j].set_title(TITLES[rg]+(' - current' if i==0 else ' - current and video'),fontsize=8)
            axes[i,j].set_xlabel(axes[i,j].get_xlabel().replace(' harmonic amplitude',' harmonic\namplitude'))
    run_legend(fig,frame,ncol=3);save(fig,path)

def video_and_scores(frame,out):
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.1));fig.subplots_adjust(left=.09,right=.99,top=.74,bottom=.2,wspace=.55)
    for ax,rg in zip(axes,f.RANGES):
        plane(ax,frame[frame['range']==rg],['gain_positive','gain_negative']);ax.set_title(TITLES[rg])
    run_legend(fig,frame,ncol=3);save(fig,out/'video_only_lda')
    pred=pd.read_csv(out.parent/'loocv_predictions.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.1));fig.subplots_adjust(left=.09,right=.99,top=.74,bottom=.18,wspace=.45)
    for ax,model,title in zip(axes,['M1_absolute_maximum','gain_pair','M2_a2_gain'],['Current only','Video only','Current and video']):
        q=pred[(pred.window_cycles==3)&(pred.strategy=='pooled')&(pred.model==model)]
        for j,rg in enumerate(f.RANGES):
            for run,(cls,marker,name) in info(frame).items():
                v=q[(q['range']==rg)&(q.run==run)]
                ax.scatter(j+np.linspace(-.2,.2,len(v)),v.score,color=COLORS[cls],marker=marker,s=20,edgecolors='white',linewidths=.3)
        ax.axhline(0,color=GREY,ls='--',lw=1.1);ax.set(xticks=[0,1,2],xticklabels=['Small','Medium','Full'],ylabel='Held-out LDA score',title=title);grid(ax)
    run_legend(fig,frame,ncol=3);save(fig,out/'held_out_scores')

def folds(frame,out):
    cols=['M2_a2','gain_difference'];part=frame[frame['range']=='small']
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.4));fig.subplots_adjust(top=.83,bottom=.12,wspace=.43,hspace=.6)
    for ax,(run,(cls,marker,name)) in zip(axes.flat,info(frame).items()):
        train=part[part.run!=run];test=part[part.run==run]
        fit=estimator().fit(train[cols].to_numpy(),train.true_class)
        plane(ax,part,cols,fit,alpha=.15)
        ax.scatter(test[cols[0]],test[cols[1]],color=COLORS[cls],marker=marker,s=30,
                   edgecolors=GREY,linewidths=.5,zorder=5)
        correct=int(np.sum((fit.decision_function(test[cols].to_numpy())>0)==test.true_class.to_numpy()))
        ax.set_title(f'Held out: {name} ({correct}/{len(test)})',fontsize=8)
    run_legend(fig,frame,ncol=3);save(fig,out/'held_out_boundaries_small')

def run(output):
    out=Path(output);p=out/'figures';p.mkdir(exist_ok=True);apply_style()
    frame=pd.read_csv(out/'condition_features.csv');cycles=pd.read_csv(out/'cycle_features.csv')
    selected_boxes(frame,p/'01_parameter_boxplots')
    matrix(frame[SELECT].corr(),SELECT,p/'02_parameter_correlations','Three-cycle feature means')
    pooled(cycles,p/'03_pooled_current_lda','Pooled one-cycle observations')
    pooled(frame,p/'lda_three_cycle','Pooled three-cycle feature means')
    lda_planes(frame,p/'lda_by_amplitude')
    for m in range(4):all_boxes(frame,[f'M{m}_{x}' for x in f.CURRENT],p/f'current_M{m}_all_parameters')
    all_boxes(frame,f.VIDEO,p/'video_all_parameters')
    active=[f'M{m}_{x}' for m in [1,2] for x in f.CURRENT]+f.VIDEO
    for cls,name in [(None,'pooled'),(0,'healthy'),(1,'degraded')]:
        q=frame if cls is None else frame[frame.true_class==cls]
        matrix(q[active].corr(),active,p/('active_correlation_'+name),'M1 / M2 and video: '+name)
    for rg in f.RANGES:
        q=frame[frame['range']==rg];matrix(q[SELECT].corr(),SELECT,p/('selected_correlation_'+rg),TITLES[rg])
    centred=frame[SELECT]-frame.groupby('condition')[SELECT].transform('mean')
    matrix(centred.corr(),SELECT,p/'condition_centred_correlation','Condition-centred feature correlations')
    video_and_scores(frame,p);folds(frame,p)
    export_tables(out)

def export_tables(out):
    out=Path(out)
    metrics=pd.read_csv(out/'loocv_metrics.csv');tables=out/'tables';tables.mkdir(exist_ok=True)
    registry=models()
    metrics['input_parameters']=metrics.model.map(lambda model:'; '.join(registry[model]['features']))
    metrics['model_label']=metrics.model.map(lambda model:registry[model]['label'])
    metrics[(metrics.window_cycles==1)&(metrics.strategy=='pooled')&(metrics.evaluated_range=='all')&(metrics.threshold=='standard')].to_csv(tables/'main_one_cycle_counts.csv',index=False)
    metrics[(metrics.window_cycles==3)&(metrics.strategy=='pooled')&(metrics.evaluated_range=='all')&(metrics.threshold=='standard')].to_csv(tables/'supplementary_three_cycle_counts.csv',index=False)
    metrics[(metrics.window_cycles==3)&(metrics.strategy=='pooled')&(metrics.evaluated_range!='all')&(metrics.threshold=='standard')].to_csv(tables/'pooled_amplitude_counts.csv',index=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');run(a.parse_args().output_dir)
