"""Publication figures. All points are shown; colour = class, marker = run.

Boxplots/correlations use three-cycle feature means. LDA plots fitted to all
points are descriptive only; held-out fold boundaries are plotted separately.
"""
import io,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import shaft_features as f
from shaft_analysis import estimator,models

BLUE='#2075a1';RED='#c94324';COLORS=[BLUE,RED]
SHORT={'M1_absolute_maximum':'M1 absolute maximum','M1_rms':'M1 RMS','M2_rms':'M2 RMS','M2_mean':'M2 phase mean',
'M2_a2':'M2 second harmonic A2','M2_rc':'M2 RC coefficient','gain_difference':'Absolute gain difference',
'gain_positive':'Positive gain','gain_negative':'Negative gain','video_mae':'Centred video MAE',
'video_relative_mae':'Relative video MAE'}
TITLES={'small':'Small: ±5.73°','medium':'Medium: ±11.46°','full':'Full: ±22.92°'}
SELECT=['M1_absolute_maximum','M1_rms','M2_mean','M2_rms','M2_a1','M2_a2','M2_rc',
'gain_positive','gain_negative','gain_difference','video_mae','video_relative_mae']
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})


def save(fig,path):
    from artifact_io import save_figure
    save_figure(fig,path)


def info(frame):
    return {r:(i%2,['o','s','D','^'][i],('Healthy 09:47','Healthy 11:40','Degraded 9 Sep','Degraded 10 Sep')[i])
            for i,r in enumerate(frame.run.unique())}


def legend(fig,frame):
    handles=[Line2D([],[],marker=v[1],ls='',color=COLORS[int(frame.loc[frame.run==r,'true_class'].iloc[0])],
                    markersize=4,label=v[2]) for r,v in info(frame).items()]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,1.0))


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


def plane(ax,part,columns,model=None):
    X=part[columns].to_numpy();y=part.true_class.to_numpy();fit=estimator().fit(X,y) if model is None else model
    limits=[]
    for k in range(2):
        lo,hi=X[:,k].min(),X[:,k].max();pad=max((hi-lo)*.13,.03);limits.append((lo-pad,hi+pad))
    xx,yy=np.meshgrid(np.linspace(*limits[0],140),np.linspace(*limits[1],140));score=fit.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,score,levels=[-1e10,0,1e10],colors=[BLUE,RED],alpha=.09)
    if score.min()<0<score.max():ax.contour(xx,yy,score,levels=[0],colors='k',linestyles='--',linewidths=1)
    for r,(_,marker,_) in info(part).items():
        p=part[part.run==r];cls=int(p.true_class.iloc[0]);ax.scatter(p[columns[0]],p[columns[1]],c=COLORS[cls],marker=marker,s=24,edgecolors='white',linewidths=.4,zorder=4)
    defs=f.definitions().set_index('feature')
    for setter,col in zip([ax.set_xlabel,ax.set_ylabel],columns):setter(SHORT.get(col,col)+' ['+defs.loc[col,'unit']+']')
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.grid(alpha=.13)


def lda_planes(frame,path):
    fig,axes=plt.subplots(2,3,figsize=(11,6.6));legend(fig,frame)
    for j,rg in enumerate(f.RANGES):
        part=frame[frame['range']==rg]
        for i,cols in enumerate([['M1_absolute_maximum','M2_rms'],['M2_a2','gain_difference']]):
            plane(axes[i,j],part,cols);axes[i,j].set_title(TITLES[rg]+(' · current only' if i==0 else ' · current + video'))
    fig.subplots_adjust(top=.89,wspace=.42,hspace=.50,bottom=.09);save(fig,path)


def folds(frame,out):
    cols=['M2_a2','gain_difference'];part=frame[frame['range']=='small'];fig,axes=plt.subplots(2,2,figsize=(9,7))
    for ax,(r,(_,_,name)) in zip(axes.flat,info(frame).items()):
        train=part[part.run!=r];test=part[part.run==r];fit=estimator().fit(train[cols].to_numpy(),train.true_class)
        # Use all points only to set common axes. Draw actual held-out points over training outlines.
        plane(ax,part,cols,fit)
        for collection in ax.collections[2:]:collection.set_alpha(.12)
        cls=int(test.true_class.iloc[0]);ax.scatter(test[cols[0]],test[cols[1]],s=40,c=COLORS[cls],marker=info(frame)[r][1],edgecolors='black',linewidths=.4,zorder=6)
        z=fit.decision_function(test[cols].to_numpy());correct=np.sum((z>0)==test.true_class.to_numpy());ax.set_title(f'Held out: {name} · {correct}/6 correct')
    fig.subplots_adjust(hspace=.45,wspace=.4);save(fig,out/'held_out_boundaries_small')


def create(frame,out):
    out=Path(out);p=out/'figures';(p).mkdir(parents=True,exist_ok=True);(p).mkdir(parents=True,exist_ok=True)
    selected_boxes(frame,p/'01_parameter_boxplots');matrix(frame[SELECT].corr(),SELECT,p/'02_parameter_correlations','Three-cycle feature means · all amplitudes')
    lda_planes(frame,p/'03_descriptive_lda')
    for m in range(4):all_boxes(frame,[f'M{m}_{x}' for x in f.CURRENT],p/f'appendix/current_M{m}_all_parameters')
    all_boxes(frame,f.VIDEO,p/'video_all_parameters')
    active=[f'M{m}_{x}' for m in [1,2] for x in f.CURRENT]+f.VIDEO
    for cls,name in [(None,'pooled'),(0,'healthy'),(1,'degraded')]:
        q=frame if cls is None else frame[frame.true_class==cls];matrix(q[active].corr(),active,p/('appendix/active_correlation_'+name),'M1 / M2 + video · '+name)
    for rg in f.RANGES:
        q=frame[frame['range']==rg];matrix(q[SELECT].corr(),SELECT,p/('appendix/selected_correlation_'+rg),TITLES[rg]+' · three-cycle feature means')
    centred=frame[SELECT]-frame.groupby('condition')[SELECT].transform('mean');matrix(centred.corr(),SELECT,p/'condition_centred_correlation','Feature correlation after subtracting condition means')
    folds(frame,p)
    # Descriptive full-data models are exported for reproducibility, not deployed.
    import joblib
    mod=out/'descriptive_models';mod.mkdir(parents=True,exist_ok=True)
    for name in ['M1_absolute_maximum','M2_a2_gain','gain_pair']:
        cols=models()[name]['features'];joblib.dump(dict(features=cols,model=estimator().fit(frame[cols].to_numpy(),frame.true_class),
             scope='all amplitudes, three-cycle feature means; exploratory full-data fit'),mod/(name+'.joblib'))
    (p/'figure_inventory.json').write_text(json.dumps(dict(main=['01_parameter_boxplots','02_parameter_correlations','03_descriptive_lda'],
              appendix=[x.stem for x in sorted((p).glob('*.png'))],descriptive_lda=True,point_unit='mean of three per-cycle features'),indent=2))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs');v=a.parse_args();create(pd.read_csv(v.output_dir/'condition_features.csv'),v.output_dir)
