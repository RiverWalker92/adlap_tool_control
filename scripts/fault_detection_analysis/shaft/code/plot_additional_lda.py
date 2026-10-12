"""Generate video-only descriptive LDA and actual out-of-run score plots.
No LOOCV model is refitted. Reads the results from shaft_analysis.py."""
import argparse,json,re,shutil,zipfile,io
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from PIL import Image

BLUE='#2075a1';RED='#c94324'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,path):
    from artifact_io import save_figure
    save_figure(fig,path)


def create(output_dir):
    output_dir=Path(output_dir)
    figures=output_dir/'figures';figures.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(output_dir/'condition_features.csv')
    pred=pd.read_csv(output_dir/'loocv_predictions.csv')
    # Additional video-only LDA geometry, fitted descriptively to all points.
    runs=list(frame.run.unique());markers=['o','s','D','^'];names=['Healthy 09:47','Healthy 11:40','Degraded 9 Sep','Degraded 10 Sep'];colors=[BLUE,BLUE,RED,RED]
    fig,axes=plt.subplots(1,3,figsize=(11,3.65))
    for ax,rg,title in zip(axes,['small','medium','full'],['Small: ±5.73°','Medium: ±11.46°','Full: ±22.92°']):
        q=frame[frame['range']==rg];X=q[['gain_positive','gain_negative']].to_numpy();y=q.true_class.to_numpy()
        fit=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5])).fit(X,y)
        lim=[]
        for j in range(2):
            lo,hi=X[:,j].min(),X[:,j].max();margin=(hi-lo)*.15;lim.append((lo-margin,hi+margin))
        xx,yy=np.meshgrid(np.linspace(*lim[0],140),np.linspace(*lim[1],140));z=fit.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx,yy,z,levels=[-1e12,0,1e12],colors=[BLUE,RED],alpha=.09)
        if z.min()<0<z.max():ax.contour(xx,yy,z,levels=[0],colors='black',linestyles='--',linewidths=1)
        for i,r in enumerate(runs):
            q1=q[q.run==r];ax.scatter(q1.gain_positive,q1.gain_negative,color=colors[i],marker=markers[i],s=23,edgecolors='white',linewidths=.3)
        ax.set_title(title+' · video only');ax.set_xlabel('Positive gain g+ [–]');ax.set_ylabel('Negative gain g− [–]');ax.grid(alpha=.15)
    handles=[Line2D([],[],color=c,marker=m,ls='',label=n,markersize=4) for c,m,n in zip(colors,markers,names)]
    fig.legend(handles=handles,ncol=4,loc='upper center',frameon=False,fontsize=8);fig.subplots_adjust(top=.77,bottom=.2,wspace=.4);save(fig,figures/'video_only_lda')
    # Held-out scalar scores: no classifier is refitted here.
    fig,axes=plt.subplots(1,3,figsize=(11,3.9))
    for ax,model,title in zip(axes,['M1_absolute_maximum','gain_pair','M2_a2_gain'],['Current only: M1 peak','Video only: g+ and g−','Current + video: M2 A2 + Δg']):
        q=pred[(pred.window_cycles==3)&(pred.strategy=='pooled')&(pred.model==model)]
        for j,rg in enumerate(['small','medium','full']):
            for i,r in enumerate(runs):
                v=q[(q['range']==rg)&(q.run==r)];jitter=np.linspace(-.22,.22,len(v));ax.scatter(j+jitter,v.score,color=colors[i],marker=markers[i],s=21,edgecolors='white',linewidths=.3,zorder=4)
        ax.axhline(0,color='black',ls='--',lw=1);ax.set_xticks([0,1,2],['Small','Medium','Full']);ax.set_title(title);ax.set_ylabel('Held-out LDA score d');ax.grid(axis='y',alpha=.17)
    fig.legend(handles=handles,ncol=4,loc='upper center',frameon=False,fontsize=8);fig.subplots_adjust(top=.77,bottom=.15,wspace=.4);save(fig,figures/'held_out_scores')
    inventory_path=output_dir/'figures/figure_inventory.json'
    inventory=json.loads(inventory_path.read_text())
    for name in ['video_only_lda','held_out_scores']:
        if name not in inventory['appendix']:inventory['appendix'].append(name)
    inventory_path.write_text(json.dumps(inventory,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');v=a.parse_args();create(v.output_dir)
