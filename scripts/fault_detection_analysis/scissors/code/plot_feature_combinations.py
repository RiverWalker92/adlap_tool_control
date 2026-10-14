"""Compact candidate summaries and descriptive fits of leading combinations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classify_cuts import SCENARIOS,SELECTIONS,dataset
from search_feature_combinations import ranking

ROOT=Path(__file__).resolve().parents[1]
SELECTION_NAMES={'all':'All cuts','first':'Cut 1','later':'Cuts 2,3 separately'}
BLUE='#397d9e';RED='#c45860';ORANGE='#d49a3e'

def run(root=ROOT):
    root=Path(root);out=root/'outputs/combinations';figs=out/'figures';figs.mkdir(exist_ok=True)
    data=pd.read_csv(root/'outputs/cut_features.csv');allm=pd.read_csv(out/'cv_metrics.csv').fillna('')
    registry=pd.read_csv(out/'model_registry.csv').fillna('');defs=pd.read_csv(root/'outputs/feature_definitions.csv').set_index('column')
    models=np.load(out/'full_models.npz');decisions=np.load(out/'cv_decisions.npz')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def label(col):
        r=defs.loc[col];motor=f'M{r.motor}' if str(r.motor) in ['0','3'] else 'M0–M3'
        return f'{motor} {r.phase.replace("cycle","active cycle")} {r["name"].lower()} [{r.unit}]'
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(figs/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    selected=[]
    for scenario in SCENARIOS:
        available=allm[allm.scenario==scenario]
        if available.empty:continue
        k=SCENARIOS[scenario][1];objectives=['fault_detection'] if k==2 else ['fault_detection','severity']
        for objective in objectives:
            fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),layout='constrained')
            for ax,selection in zip(axes,SELECTIONS):
                chosen=ranking(available[available.selection==selection],severity=objective=='severity').iloc[0]
                selected.append({**chosen.to_dict(),'scenario':scenario,'selection':selection,'objective':objective})
                index=int(registry.index[registry.model==chosen.model][0]);cols=[chosen.feature1]+([chosen.feature2] if chosen.feature2 else [])
                d,k=dataset(data,scenario,selection);key=f'{scenario}__{selection}'
                co=models[key+'__coef'][index];b=models[key+'__intercept'][index]
                palette=[BLUE,RED] if k==2 else [BLUE,ORANGE,RED]
                x=d[cols[0]].to_numpy();dx=max(float(np.ptp(x)),.1);lo=x.min()-.08*dx;hi=x.max()+.08*dx
                xx=np.linspace(lo,hi,300)
                if len(cols)==2:
                    y=d[cols[1]].to_numpy();dy=max(float(np.ptp(y)),.1);yl=y.min()-.08*dy;yh=y.max()+.08*dy
                    yy=np.linspace(yl,yh,240);X,Y=np.meshgrid(xx,yy);points=np.c_[X.ravel(),Y.ravel()]
                    z=points@co.T+b;pred=z.argmax(1).reshape(X.shape)
                    ax.contourf(X,Y,pred,levels=np.arange(k+1)-.5,colors=palette,alpha=.13)
                    for a in range(k):
                        for c in range(a+1,k):
                            w=co[a]-co[c];bias=b[a]-b[c]
                            if abs(w[1])>1e-14:bx=xx;by=-(w[0]*xx+bias)/w[1]
                            elif abs(w[0])>1e-14:by=yy;bx=np.full(len(yy),-bias/w[0])
                            else:continue
                            valid=(bx>=lo)&(bx<=hi)&(by>=yl)&(by<=yh)
                            if k==3:
                                test=np.c_[bx,by]@co.T+b;valid&=test[:,3-a-c]<=test[:,a]+1e-8
                            ax.plot(np.where(valid,bx,np.nan),np.where(valid,by,np.nan),color='#57666d',lw=.9,ls='--')
                    ax.set_ylim(yl,yh);ax.set_ylabel(label(cols[1]))
                else:
                    y=d.truth.to_numpy()+np.linspace(-.07,.07,len(d))
                    zz=xx[:,None]*co[:,0][None,:]+b[None,:];pred=zz.argmax(1)
                    start=0
                    for pos in range(1,len(xx)+1):
                        if pos==len(xx) or pred[pos]!=pred[start]:
                            ax.axvspan(xx[start],xx[pos-1],color=palette[pred[start]],alpha=.13);start=pos
                    ax.set(yticks=list(range(k)),yticklabels=SCENARIOS[scenario][2],ylabel='Recorded condition',ylim=(-.4,k-.6))
                for cls in range(k):
                    for run,marker in [(1,'o'),(2,'^')]:
                        mask=(d.truth.to_numpy()==cls)&(d.run_number.to_numpy()==run)
                        ax.scatter(x[mask],y[mask],color=palette[cls],marker=marker,s=28,edgecolor='white',lw=.4,
                                   label=SCENARIOS[scenario][2][cls] if run==1 else None,zorder=3)
                ax.set_xlim(lo,hi);ax.set_xlabel(label(cols[0]));ax.set_title(SELECTION_NAMES[selection]);ax.grid(alpha=.1)
            fig.suptitle(f'{scenario} · leading {objective.replace("_"," ")} candidates',fontsize=11)
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.08),ncol=k,fontsize=8)
            fig.supxlabel('Descriptive full-data fits; run-wise counts are separate. Candidate ranking is exploratory.',fontsize=8)
            save(fig,f'best_{scenario}_{objective}')
            # Withheld confusion matrices, separated from full-data boundary illustrations.
            fig,axes=plt.subplots(1,3,figsize=(8.8,3.2),layout='constrained')
            for ax,selection in zip(axes,SELECTIONS):
                chosen=ranking(available[available.selection==selection],severity=objective=='severity').iloc[0]
                cm=np.asarray(json.loads(chosen.confusion_matrix));norm=cm/cm.sum(1)[:,None]
                ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
                for a in range(k):
                    for c in range(k):ax.text(c,a,str(cm[a,c]),ha='center',va='center',color='white' if norm[a,c]>.55 else '#243645')
                labels=['Reference','Positive'] if k==2 else ['Healthy','Intermediate','Blunt']
                ax.set(title=SELECTION_NAMES[selection],xticks=range(k),yticks=range(k),xticklabels=labels,yticklabels=labels,xlabel='Predicted',ylabel='Recorded')
            fig.suptitle(f'{scenario} · withheld decisions for selected candidates',fontsize=11)
            save(fig,f'confusion_{scenario}_{objective}')
    best=pd.DataFrame(selected);best.to_csv(out/'best_models_summary.csv',index=False)
    binary=best[(best.scenario=='healthy_all')&(best.objective=='fault_detection')].copy()
    binary[['selection','feature1','feature2','TP','FN','TN','FP']].to_csv(out/'binary_compact.csv',index=False)
    severity=best[(best.scenario.str.startswith('three'))&(best.objective=='severity')].copy()
    severity[['scenario','selection','feature1','feature2','class0_correct','class0_n','class1_correct','class1_n','class2_correct','class2_n','alarm_FN','alarm_FP']].to_csv(out/'severity_compact.csv',index=False)
    if (out/'nested_metrics.csv').exists():
        nested=pd.read_csv(out/'nested_metrics.csv')
        comparison=binary[['selection','TP','FN','TN','FP']].merge(nested[nested.scenario=='healthy_all'][['selection','TP','FN','TN','FP']],on='selection',suffixes=('_best_candidate','_nested_selection'))
        comparison.to_csv(out/'candidate_vs_nested.csv',index=False)
    print('Combination figures and compact summaries written.',flush=True)

if __name__=='__main__':run()
