#!/usr/bin/env python3
"""Plot top five video, single-current, current-pair and mixed classifiers.
Rank pooled one-cycle, standard-threshold leave-one-run-out results.
Uses the saved metadata, never a potentially changed model registry.
"""
import argparse,json,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def run(out):
    out=Path(out); dest=out/'figures_top5';dest.mkdir(parents=True,exist_ok=True)
    registry=json.loads((out/'analysis_metadata.json').read_text())['models']
    defs=pd.read_csv(out/'feature_definitions.csv').set_index('feature')
    metrics=pd.read_csv(out/'loocv_metrics.csv')
    selected=metrics[(metrics.window_cycles==1)&(metrics.strategy=='pooled')&(metrics.fit_range=='all')&(metrics.evaluated_range=='all')&(metrics.threshold=='standard')].copy()
    def category(key):
        cols=registry[key]['features'];families=[defs.loc[c,'family'] for c in cols]
        if len(cols)>2:return None
        if all(t=='video' for t in families):return 'video_only'
        if families==['current']:return 'current_single'
        if families==['current','current']:return 'current_pair'
        if sorted(families)==['current','video']:return 'current_video'
        return None
    selected['category']=selected.model.map(category)
    selected['n_inputs']=selected.model.map(lambda k:len(registry[k]['features']))
    selected['errors']=selected.fn+selected.fp
    # Equal BA remains a tie. Simplicity then model name give reproducible display order.
    selected=selected.sort_values(['balanced_accuracy','n_inputs','model'],ascending=[False,True,True])
    predictions=pd.read_csv(out/'loocv_predictions.csv')
    predictions=predictions[(predictions.window_cycles==1)&(predictions.strategy=='pooled')&(predictions.fit_range=='all')]
    cycles=pd.read_csv(out/'cycle_features.csv')
    titles={'video_only':'Video only (one or two parameters)','current_single':'One current parameter','current_pair':'Two current parameters','current_video':'Current + video'}
    colours={0:'#326BC1',1:'#C66C35'};markers=['o','s','^','D','v','P']
    runs=list(cycles.run.unique());labels={c:defs.loc[c,'label']+' ['+defs.loc[c,'unit']+']' for c in defs.index}
    tables=[]
    for group,title in titles.items():
        all_group=selected[selected.category==group].copy();top=all_group.head(5).copy()
        if top.empty:raise ValueError('No models in '+group)
        cutoff=top.balanced_accuracy.iloc[-1]
        tied=int(np.isclose(all_group.balanced_accuracy,cutoff).sum())
        top['display_order']=range(1,len(top)+1)
        top['input_parameters']=top.model.map(lambda k:'; '.join(registry[k]['features']))
        top['tied_at_cutoff']=tied;tables.append(top)
        fig,axes=plt.subplots(len(top),2,figsize=(13,3.25*len(top)),squeeze=False)
        for row,(_,item) in enumerate(top.iterrows()):
            cols=registry[item.model]['features'];X=cycles[cols].to_numpy();y=cycles.true_class.to_numpy()
            fit=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5])).fit(X,y)
            ax,score_ax=axes[row]
            for i,r in enumerate(runs):
                mask=(cycles.run==r).to_numpy();cls=int(y[mask][0])
                if len(cols)==2:ax.scatter(X[mask,0],X[mask,1],s=20,color=colours[cls],marker=markers[i%len(markers)],alpha=.7)
                else:
                    jitter=np.linspace(-.13,.13,mask.sum())
                    ax.scatter(X[mask,0],cls+jitter,s=20,color=colours[cls],marker=markers[i%len(markers)],alpha=.7)
            ax.set_xlabel(labels[cols[0]])
            if len(cols)==2:
                ax.set_ylabel(labels[cols[1]])
                xl=ax.get_xlim();yl=ax.get_ylim();xx,yy=np.meshgrid(np.linspace(*xl,160),np.linspace(*yl,160))
                zz=fit.decision_function(np.column_stack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
                if zz.min()<0<zz.max():ax.contour(xx,yy,zz,levels=[0],colors='#555555',linestyles='--',linewidths=1)
                ax.set_xlim(xl);ax.set_ylim(yl)
            else:
                ax.set_yticks([0,1],['Healthy','Degraded']);ax.set_ylim(-.35,1.35)
                sc=fit.named_steps['standardscaler'];lda=fit.named_steps['lineardiscriminantanalysis'];beta=lda.coef_[0,0]/sc.scale_[0]
                intercept=lda.intercept_[0]-beta*sc.mean_[0]
                if abs(beta)>1e-12:ax.axvline(-intercept/beta,color='#555555',ls='--',lw=1)
            ax.set_title(textwrap.fill(' + '.join(defs.loc[c,'label'] for c in cols),64),fontsize=10)
            q=predictions[predictions.model==item.model]
            for i,r in enumerate(runs):
                part=q[q.run==r];cls=int(part.true_class.iloc[0]);jitter=np.linspace(-.16,.16,len(part))
                score_ax.scatter(cls+jitter,part.score,s=20,color=colours[cls],marker=markers[i%len(markers)],alpha=.7)
            score_ax.axhline(0,color='#555555',ls='--',lw=1)
            score_ax.set(xticks=[0,1],xticklabels=['Healthy','Degraded'],xlim=(-.4,1.4),ylabel='Held-out LDA score')
            score_ax.set_title(f'BA {item.balanced_accuracy:.1%} | FN {int(item.fn)} | FP {int(item.fp)}',fontsize=11)
            for a in [ax,score_ax]:a.grid(alpha=.15);a.set_axisbelow(True)
        fig.suptitle(title+' — top five',fontsize=17,y=.995)
        fig.text(.5,.005,'Left: boundary fitted to all observations for illustration. Right: predictions from held-out runs.\nRanking: pooled individual cycles, standard threshold. Equal BA = tied; display order uses fewer inputs, then model name.\nExploratory selection from the same validation results; independent confirmation is needed.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.975));fig.savefig(dest/(group+'_top5.png'),dpi=180);plt.close(fig)
        print(group,':',len(all_group),'candidates;',tied,'tied at cutoff',flush=True)
    pd.concat(tables,ignore_index=True).to_csv(dest/'top5_selection.csv',index=False)
    print('Saved:',dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);run(ap.parse_args().output_dir)
