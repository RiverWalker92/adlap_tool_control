"""Gearbox parameter, validation and LDA figures (PNG only)."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyse import PARAMETERS,NAMES,COLUMNS,fit_classifier
from plot_style import *
COLOURS=[BLUE]*3
DEGRADED=ORANGE

def save(fig,directory,name):save_png(fig,Path(directory)/name)
def title(column):
    if column == 'shape_corr': return 'Waveform correlation'
    if column == 'ellipse_area': return 'Ellipse area'
    motor, parameter = column.split('_', 1)
    return motor + ' ' + NAMES[parameter]


def unit(column):
    if column == 'shape_corr': return 'Dimensionless'
    if column == 'ellipse_area': return r'mA$^2$'
    if column.endswith('_rc'): return 'mA/rad'
    return 'mA rad' if column.endswith('_auc') else 'mA'


def draw(observations, columns, output, name, ncols=3):
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.75*nrows+.38),
                             squeeze=False)
    healthy_runs = sorted(observations.loc[observations.label == 0, 'run'].unique())
    if len(healthy_runs) > len(COLOURS):
        raise ValueError('Extend run colours/markers for more than three healthy runs.')
    rng = np.random.default_rng(17)
    slots=list(range(len(columns)))
    if ncols==4 and columns[-2:]==['shape_corr','ellipse_area'] and len(columns)==13:
        # Keep the two joint parameters together, centred on the final row.
        slots=list(range(11))+[13,14]
    for slot, column in zip(slots, columns):
        ax=axes.flat[slot]
        groups = [observations.loc[observations.label == label, column].to_numpy()
                  for label in (0, 1)]
        if any(not len(g) or not np.isfinite(g).all() for g in groups):
            raise ValueError(f'Missing or nonfinite values for {column}')
        boxes = ax.boxplot(groups, positions=[0,1], widths=.45,
                           showfliers=False, patch_artist=True,
                           medianprops={'color':'#222222','linewidth':.9})
        for box, colour in zip(boxes['boxes'], [COLOURS[0], DEGRADED]):
            box.set(facecolor=colour, alpha=.15)
        for i, run in enumerate(healthy_runs):
            values = observations.loc[(observations.label == 0) &
                                      (observations.run == run), column]
            ax.scatter(rng.uniform(-.12,.12,len(values)), values, s=12,
                       color=COLOURS[i], marker=MARKERS[i], zorder=3)
        values = observations.loc[observations.label == 1, column]
        ax.scatter(1+rng.uniform(-.12,.12,len(values)), values, s=14,
                   color=DEGRADED, marker='D', zorder=3)
        # Long titles wrap so that neighbouring panels stay separate.
        label = title(column).replace('Fundamental amplitude', 'Fundamental\namplitude')
        label = label.replace('Second harmonic amplitude', 'Second harmonic\namplitude')
        ax.set(xticks=[0,1], xticklabels=['Healthy','Degraded'],
               ylabel=unit(column), title=label)
        ax.grid(axis='y', alpha=.15)
    for i,ax in enumerate(axes.flat):
        if i not in slots:ax.set_visible(False)
    handles = [Line2D([],[],color=COLOURS[i],marker=MARKERS[i],linestyle='',
                      label=f'Healthy run {i+1}',markersize=3.5)
               for i in range(len(healthy_runs))]
    handles.append(Line2D([],[],color=DEGRADED,marker='D',linestyle='',
                          label='Degraded blocks',markersize=3.5))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               fontsize=7, handletextpad=.3, columnspacing=1)
    fig.tight_layout(rect=(0,0,1,.95))
    save_png(fig,output/name)


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
        'M3': {'colour': ORANGE, 'line': '--', 'marker': 's'},
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


def scatter(ax,observations):
    for run,(cls,marker,label) in run_info(observations,run='run',label='label').items():
        group=observations[observations.run==run]
        ax.scatter(group.M0_rms,group.M3_rms,color=CLASS_COLOURS[cls],marker=marker if cls==0 else 'D',
                   s=28,edgecolors='white',linewidths=.4,label=label,zorder=3)
    group=observations[observations.label==1].sort_values('start')
    for i,(_,row) in enumerate(group.iterrows()):
        ax.annotate(str(i+1),(row.M0_rms,row.M3_rms),xytext=(4,4),textcoords='offset points',fontsize=7)
    last=group.iloc[-1]
    ax.scatter(last.M0_rms,last.M3_rms,color=ORANGE,marker='*',s=100,
               edgecolors='#5A2413',linewidths=.6,label='Final degraded block',zorder=4)
    ax.set(xlabel='M0 residual RMS [mA]',ylabel='M3 residual RMS [mA]');grid(ax)

def boundary(ax,estimator,observations,**kwargs):
    co,bias=raw_coefficients(estimator);lo,hi,yl,yh=limits(observations.M0_rms.to_numpy(),observations.M3_rms.to_numpy())
    x=np.linspace(lo,hi,400);y=-(co[0,0]*x+bias[0])/co[0,1]
    ax.plot(x,y,color=GREY,ls='--',lw=1,label=kwargs.get('label'))
    ax.set_xlim(lo,hi);ax.set_ylim(yl,yh)

def lda_panel(ax,d,columns,co,bias,title):
    x,y=d[columns[0]].to_numpy(),d[columns[1]].to_numpy()
    lda_regions(ax,co,bias,limits(x,y))
    for run,(cls,marker,label) in run_info(d,label='label').items():
        q=d[d.run==run];ax.scatter(q[columns[0]],q[columns[1]],color=CLASS_COLOURS[cls],
            marker=marker if cls==0 else 'D',s=28,edgecolors='white',linewidths=.4,zorder=3,label=label)
    ax.set(title=title,xlabel=display_name(columns[0]),ylabel=display_name(columns[1]))

def display_name(col):
    if col=='shape_corr':return 'M0-M3 waveform correlation'
    if col=='ellipse_area':return 'M0-M3 ellipse area [mA squared]'
    motor,param=col.split('_',1)
    unit='mA rad' if param in ['abs_auc','pos_auc'] else 'mA/rad' if param=='rc' else 'mA'
    return f'{motor} {NAMES[param].lower().replace("rms","RMS").replace("phase sd","phase SD")} [{unit}]'

def final_block_validation(five,figures):
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.2))
    fig.subplots_adjust(left=.08,right=.99,bottom=.20,top=.73,wspace=.4)
    cols=['M0_rms','M3_rms']
    lim=limits(five.M0_rms.to_numpy(),five.M3_rms.to_numpy())
    for i,(ax,healthy) in enumerate(zip(axes,sorted(five.loc[five.label==0,'run'].unique()))):
        train=five[(five.run!=healthy)&~((five.label==1)&(five.window=='block5_6'))]
        test=five[(five.run==healthy)|((five.label==1)&(five.window=='block5_6'))]
        fitted=fit_classifier(train,{'features':cols});co,bias=raw_coefficients(fitted)
        lda_regions(ax,co,bias,lim)
        for run,(cls,marker,label) in run_info(five,label='label').items():
            group=five[five.run==run]
            ax.scatter(group.M0_rms,group.M3_rms,color=CLASS_COLOURS[cls],marker=marker if cls==0 else 'D',
                       s=20,edgecolors='white',linewidths=.3,alpha=.15,zorder=3)
        q=test[test.label==0]
        ax.scatter(q.M0_rms,q.M3_rms,color=BLUE,marker=MARKERS[i],s=30,edgecolors='white',linewidths=.4,zorder=4)
        q=test[test.label==1]
        ax.scatter(q.M0_rms,q.M3_rms,color=ORANGE,marker='*',s=85,edgecolors='#5A2413',linewidths=.5,zorder=5)
        ax.set(title=f'Healthy run {i+1} withheld',xlabel='M0 residual RMS [mA]',ylabel='M3 residual RMS [mA]')
    handles=[marker_handle(f'Healthy run {i+1}',BLUE,MARKERS[i]) for i in range(3)]
    handles += [marker_handle('Degraded blocks',ORANGE,'D'),marker_handle('Final degraded block',ORANGE,'*'),boundary_handle()]
    figure_legend(fig,handles,3);save(fig,figures,'last_five_validation')

def run(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    apply_style();features=pd.read_csv(output/'features.csv');five=features[features.n==5]
    draw(five,['M0_'+p for p in PARAMETERS],figures,'boxplots_M0',3)
    draw(five,['M3_'+p for p in PARAMETERS]+['shape_corr','ellipse_area'],figures,'boxplots_M3_joint',4)
    correlations(five,figures)
    fit=fit_classifier(five,{'features':['M0_rms','M3_rms']});co,bias=raw_coefficients(fit)
    fig,ax=plt.subplots(figsize=(3.5,3.3));fig.subplots_adjust(left=.17,right=.98,bottom=.16,top=.70)
    lda_regions(ax,co,bias,limits(five.M0_rms.to_numpy(),five.M3_rms.to_numpy()))
    scatter(ax,five);ax.set_title('Paired residual RMS: five-rotation blocks')
    handles,labels=ax.get_legend_handles_labels();figure_legend(fig,handles+[boundary_handle()],2)
    save(fig,figures,'lda_rms')
    registry=pd.read_csv(output/'combinations/model_registry.csv').fillna('')
    full=np.load(output/'combinations/full_models.npz')
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.2));fig.subplots_adjust(left=.08,right=.99,bottom=.24,top=.73,wspace=.55)
    for ax,model,title in zip(axes,['G075','G083','G102'],['(a) Same motor','(b) Mean and SD','(c) Integral and RMS']):
        index=registry.index[registry.model==model][0];row=registry.loc[index]
        lda_panel(ax,five,[row.feature1,row.feature2],full['n5_coef'][index],full['n5_intercept'][index],title)
        ax.set_xlabel(ax.get_xlabel().replace(' second harmonic amplitude',' second harmonic\namplitude'))
    handles=[marker_handle(f'Healthy run {i+1}',BLUE,MARKERS[i]) for i in range(3)]
    handles += [marker_handle('Degraded run',ORANGE,'D'),boundary_handle()]
    figure_legend(fig,handles,3);save(fig,figures,'lda_mixed_parameters')
    rc_fit_figure(output,figures);existing_report_figures(output,figures);final_block_validation(five,figures)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');run(p.parse_args().output)
