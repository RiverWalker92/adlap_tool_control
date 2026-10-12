#!/usr/bin/env python3

"""Export report figures from the gearbox RC analysis outputs."""

import argparse

import json

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyse import PARAMETERS, NAMES, COLUMNS, MAIN_MODELS, fit_classifier

from grouped_boxplots import draw


COLOURS = ['#176b9c', '#479cbb', '#263e62']

DEGRADED = '#c34525'


def tex_name(column):
    if column == 'shape_corr': return 'Waveform correlation'
    if column == 'ellipse_area': return 'Ellipse area'
    if column.startswith('M0_M3_'): return NAMES[column[len('M0_M3_'):]]
    motor, parameter = column.split('_', 1)
    return motor + ' ' + NAMES[parameter]


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / (name+'.pending.png')
    fig.savefig(pending, format='png', dpi=220, bbox_inches='tight')
    pending.replace(directory/(name+'.png'))
    plt.close(fig)


def scatter(ax, observations):
    for i, run in enumerate(sorted(observations.loc[observations.label==0,'run'].unique())):
        group = observations[observations.run==run]
        ax.scatter(group.M0_rms,group.M3_rms,c=COLOURS[i],marker=['o','s','^'][i],
                   s=22,label=f'Healthy run {i+1}',zorder=3)
    group=observations[observations.label==1].sort_values('start')
    ax.scatter(group.M0_rms,group.M3_rms,c=DEGRADED,marker='D',s=24,
               label='Degraded blocks',zorder=3)
    for i,(_,row) in enumerate(group.iterrows()):
        ax.annotate(str(i+1),(row.M0_rms,row.M3_rms),xytext=(3,4),
                    textcoords='offset points',fontsize=6.5)
    last=group.iloc[-1]
    ax.scatter(last.M0_rms,last.M3_rms,c=DEGRADED,marker='*',s=100,
               edgecolor='black',linewidth=.4,zorder=4)
    ax.set(xlabel='M0 RMS [mA]',ylabel='M3 RMS [mA]')
    ax.grid(alpha=.15)


def boundary(ax, estimator, observations, **kwargs):
    weights=estimator['lda'].coef_[0]/estimator['scale'].scale_
    intercept=estimator['lda'].intercept_[0]-weights@estimator['scale'].mean_
    lo,hi=observations.M0_rms.min(),observations.M0_rms.max()
    xx=np.linspace(lo-.5,hi+.5,100)
    ax.plot(xx,(-intercept-weights[0]*xx)/weights[1],**kwargs)
    ax.set_xlim(xx[0],xx[-1]);ax.set_ylim(observations.M3_rms.min()-.5,
                                        observations.M3_rms.max()+.5)


def correlations(five, directory):
    short={'peak':'max','abs_peak':'abs max','mean':'mean','abs_auc':'abs int',
           'pos_auc':'pos int','rms':'RMS','phase_sd':'SD','ptp':'range',
           'a1':'1f','a2':'2f','rc':'RC'}
    labels=[('Wave corr' if c=='shape_corr' else 'Ellipse' if c=='ellipse_area'
             else c.split('_',1)[0]+' '+short[c.split('_',1)[1]]) for c in COLUMNS]
    for scope,group in [('pooled',five),('healthy',five[five.label==0]),
                        ('degraded',five[five.label==1])]:
        matrix=group[COLUMNS].corr().to_numpy()
        fig,ax=plt.subplots(figsize=(10.2,9))
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(labels)),labels,rotation=70,ha='right',fontsize=7)
        ax.set_yticks(range(len(labels)),labels,fontsize=7)
        for row in range(len(labels)):
            for col in range(len(labels)):
                value=matrix[row,col]
                ax.text(col,row,f'{value:.2f}',ha='center',va='center',fontsize=4.6,
                        color='white' if abs(value)>.65 else '#222222')
        fig.colorbar(im,ax=ax,shrink=.8,label='Pearson correlation coefficient')
        fig.tight_layout();save(fig,directory,'correlation_'+scope)


def rc_fit_figure(output, directory):
    profiles=np.load(output/'rotation_profiles.npz')
    inventory=pd.read_csv(output/'inventory.csv')
    phase=profiles['phase'][:-1]
    fig,axes=plt.subplots(1,2,figsize=(7,2.8))
    for ax,i,motor in zip(axes,[0,1],['M0','M3']):
        for j,(_,run) in enumerate(inventory[inventory.label==0].iterrows()):
            residual=profiles[run.run][-5:,:-1,i].mean(axis=0)
            ax.plot(phase,residual,color=COLOURS[j],alpha=.4,lw=.8,
                    label=f'Healthy run {j+1}')
        run=inventory[inventory.label==1].iloc[0].run
        residual=profiles[run][-5:,:-1,i].mean(axis=0)
        slope,intercept=np.linalg.lstsq(np.column_stack([phase,np.ones(len(phase))]),
                                        residual,rcond=None)[0]
        ax.plot(phase,residual,color=DEGRADED,lw=1.2,label='Final degraded block')
        ax.plot(phase,slope*phase+intercept,color='#222222',ls='--',lw=1,
                label=f'RC = {slope:.3f} mA/rad')
        ax.set(title=motor,xlabel='Rotational phase [rad]',ylabel='Current residual [mA]')
        ax.grid(alpha=.15);ax.legend(fontsize=6,frameon=False)
    fig.tight_layout();save(fig,directory,'rc_final_block_fit')


def existing_report_figures(output, directory):
    """Regenerate the evolution and final-block figures from measured data."""
    features=pd.read_csv(output/'features.csv');five=features[features.n==5]
    runs=sorted(five.loc[five.label==0,'run'].unique())
    fig,axes=plt.subplots(2,2,figsize=(7,4.8))
    motor_styles = {
        'M0': {'colour': DEGRADED, 'line': '-', 'marker': 'o'},
        'M3': {'colour': '#7B3294', 'line': '--', 'marker': 's'},
    }
    for ax,panel,param in zip(axes.flat,'ABCD',['peak','rms','phase_sd','mean']):
        for motor,style in motor_styles.items():
            for i,run in enumerate(runs):
                group=five[five.run==run].sort_values('start')
                ax.plot(group.centre,group[motor+'_'+param],c=COLOURS[i],
                        ls=style['line'],alpha=.4,lw=.9,zorder=1)
            group=five[five.label==1].sort_values('start')
            ax.plot(group.centre,group[motor+'_'+param],c=style['colour'],
                    ls=style['line'],marker=style['marker'],ms=4,lw=1.7,
                    markeredgecolor='white',markeredgewidth=.3,zorder=3)
        ax.set(title=panel+'  '+NAMES[param],xlabel='Centre of block [rotation]',
               ylabel='mA',xticks=[3,8,13,18,23,28])
        ax.grid(alpha=.15)
    legend_handles = [
        Line2D([],[],color=motor_styles[m]['colour'],
               linestyle=motor_styles[m]['line'],marker=motor_styles[m]['marker'],
               linewidth=1.7,markersize=4,label='Degraded '+m)
        for m in ('M0','M3')
    ]
    legend_handles += [
        Line2D([],[],color=COLOURS[0],linestyle=motor_styles[m]['line'],
               alpha=.4,linewidth=.9,label='Healthy '+m)
        for m in ('M0','M3')
    ]
    fig.legend(handles=legend_handles,loc='upper center',ncol=4,
               frameon=False,fontsize=7,handlelength=2.8,columnspacing=1.2)
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,directory,'04_parameter_evolution')
    definitions=json.loads((output/'model_definitions.json').read_text())
    predictions=pd.read_csv(output/'cv_predictions.csv')
    predictions=predictions[(predictions.n==5)&(predictions.model=='M0_M3_rms')&
                            (predictions.held_degraded=='block5_6')]
    fig,axes=plt.subplots(2,2,figsize=(7,5))
    scatter(axes[0,0],five)
    for i,run in enumerate(runs):
        train=five[(five.run!=run)&~((five.label==1)&(five.window=='block5_6'))]
        estimator=fit_classifier(train,definitions['M0_M3_rms'])
        boundary(axes[0,0],estimator,five,c=COLOURS[i],ls='--',lw=.8,
                 label=f'Healthy run {i+1} withheld')
        group=predictions[predictions.held_nominal==run]
        healthy=group[group.label==0].score.to_numpy()
        faulty=group[group.label==1].score.to_numpy()
        offsets=np.linspace(-.08,.08,len(healthy))
        axes[0,1].scatter(i+offsets,healthy,c=COLOURS[i],s=12)
        axes[0,1].scatter([i]*len(faulty),faulty,c=DEGRADED,marker='*',s=60)
    axes[0,0].set_title('A  Final block withheld')
    axes[0,0].legend(fontsize=5,frameon=False)
    axes[0,1].axhline(0,c='#333333',ls='--',lw=.8)
    axes[0,1].set(title='B  Withheld test scores',xticks=[0,1,2],
                  xticklabels=['Run 1','Run 2','Run 3'],xlabel='Withheld healthy run',
                  ylabel='LDA score')
    profiles=np.load(output/'rotation_profiles.npz');phase=profiles['phase']
    degraded=five.loc[five.label==1,'run'].iloc[0]
    for ax,index,motor,panel in zip(axes[1],[0,1],['M0','M3'],'CD'):
        for i,run in enumerate(runs):
            ax.plot(phase,profiles[run][-5:,:,index].mean(axis=0),c=COLOURS[i],
                    lw=.8,alpha=.65,label=f'Healthy run {i+1}')
        residual=profiles[degraded][-5:,:,index]
        for trace in residual:ax.plot(phase,trace,c=DEGRADED,alpha=.2,lw=.5)
        average=residual.mean(axis=0)
        ax.plot(phase,average,c=DEGRADED,lw=1.2,label='Degraded block mean')
        ax.axhline(average[:-1].mean(),c=DEGRADED,ls=':',lw=1)
        ax.set(title=panel+'  '+motor,xlabel='Rotational phase [rad]',
               ylabel='Current residual [mA]')
        ax.grid(alpha=.15)
    fig.tight_layout();save(fig,directory,'last_five_validation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    args=parser.parse_args();output=args.output;figures=args.figures
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    features=pd.read_csv(output/'features.csv');five=features[features.n==5]
    draw(five,['M0_'+p for p in PARAMETERS],figures,'boxplots_M0',3)
    draw(five,['M3_'+p for p in PARAMETERS]+['shape_corr','ellipse_area'],
         figures,'boxplots_M3_joint',4)
    correlations(five,figures)
    definitions=json.loads((output/'model_definitions.json').read_text())
    estimator=fit_classifier(five,definitions['M0_M3_rms'])
    fig,ax=plt.subplots(figsize=(3.5,3.1));scatter(ax,five)
    boundary(ax,estimator,five,c='#333333',ls='--',lw=1,label='Descriptive LDA boundary')
    ax.legend(fontsize=6,frameon=False,loc='best');fig.tight_layout();save(fig,figures,'lda_rms')
    rc_fit_figure(output,figures)
    existing_report_figures(output,figures)
    print('Grouped boxplots, correlation matrices, RMS LDA, RC illustration exported as PNG.')


if __name__=='__main__':main()