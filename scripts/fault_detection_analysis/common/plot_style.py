"""Shared figure appearance and exact linear LDA decision regions."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

BLUE='#2E7DB2'
ORANGE='#C95A2A'
YELLOW='#D49A3E'
GREY='#4A4A4A'
CLASS_COLOURS=[BLUE,ORANGE]
THREE_COLOURS=[BLUE,YELLOW,ORANGE]
REGIONS=['#DCE8F2','#F4EBD6','#F2E4E1']
MARKERS=['o','s','^','D']

def apply_style():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,
        'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,
        'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'axes.titleweight':'normal',
        'figure.titleweight':'normal','savefig.dpi':300})

def grid(ax):
    ax.grid(color='#BFBFBF',lw=.35,alpha=.45)
    ax.set_axisbelow(True)

def save_png(fig,path):
    path=Path(path).with_suffix('.png');path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.stem+'.pending.png')
    fig.savefig(pending,dpi=300,bbox_inches='tight',facecolor='white')
    pending.replace(path)
    import csv
    legend_text=[t.get_text() for legend in list(fig.legends)+[a.get_legend() for a in fig.axes if a.get_legend() is not None] for t in legend.get_texts()]
    record={'figure':path.name,'width_inches':round(fig.get_figwidth(),3),'height_inches':round(fig.get_figheight(),3),
            'legend_count':len(fig.legends)+sum(a.get_legend() is not None for a in fig.axes),
            'legend_labels':'; '.join(legend_text)}
    inventory=path.parent/'figure_inventory.csv';previous=[]
    if inventory.exists():
        with inventory.open(newline='',encoding='utf-8') as stream:previous=list(csv.DictReader(stream))
    previous=[r for r in previous if r['figure']!=path.name]+[record]
    with inventory.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(record));writer.writeheader();writer.writerows(sorted(previous,key=lambda r:r['figure']))
    plt.close(fig)

def marker_handle(label,colour,marker='o'):
    return Line2D([],[],marker=marker,color=colour,ls='',markersize=4.5,
                  markeredgecolor='white',markeredgewidth=.4,label=label)

def boundary_handle():return Line2D([],[],color=GREY,ls='--',lw=1.1,label='LDA boundary')

def figure_legend(fig,handles,ncol=3):
    # One figure legend, including both class and marker keys.
    for legend in list(fig.legends):legend.remove()
    labels={};
    for h in handles:labels.setdefault(h.get_label(),h)
    fig.legend(handles=list(labels.values()),loc='upper center',
               bbox_to_anchor=(.5,.99),ncol=ncol,frameon=False,
               columnspacing=1.2,handletextpad=.4,handlelength=1.7)

def class_legend(fig,binary=False,positive='After filing'):
    entries=[('Healthy',BLUE),(positive,ORANGE)] if binary else [('Healthy',BLUE),('Intermediate',YELLOW),('Blunt',ORANGE)]
    handles=[marker_handle(entries[0][0],entries[0][1]),marker_handle('Run 1',GREY,'o'),
             marker_handle(entries[1][0],entries[1][1]),marker_handle('Run 2',GREY,'s')]
    if not binary:handles.append(marker_handle(entries[2][0],entries[2][1]))
    handles.append(boundary_handle())
    figure_legend(fig,handles,3)

def run_info(frame,run='run',label='true_class'):
    info={}
    for cls in sorted(frame[label].unique()):
        for i,key in enumerate(sorted(frame.loc[frame[label]==cls,run].unique())):
            info[key]=(int(cls),MARKERS[i],f'{"Healthy" if cls==0 else "Degraded"} run {i+1}')
    return info

def run_legend(fig,frame,run='run',label='true_class',boundary=True,ncol=3):
    handles=[marker_handle(name,CLASS_COLOURS[cls],marker) for cls,marker,name in run_info(frame,run,label).values()]
    if boundary:handles.append(boundary_handle())
    figure_legend(fig,handles,ncol)

def raw_coefficients(fitted):
    steps=fitted.named_steps
    scaler=steps.get('scale',steps.get('standardscaler'))
    lda=steps.get('lda',steps.get('lineardiscriminantanalysis'))
    co=lda.coef_/scaler.scale_;bias=lda.intercept_-co@scaler.mean_
    return co,bias

def clip(poly,w,b):
    result=[]
    for a,z in zip(poly,poly[1:]+poly[:1]):
        va,vz=a@w+b,z@w+b
        if va>=0:result.append(a)
        if (va>=0)!=(vz>=0):result.append(a+(z-a)*va/(va-vz))
    return result

def lda_regions(ax,coef,bias,limits):
    coef=np.asarray(coef,float);bias=np.asarray(bias,float)
    if coef.shape[0]==1:
        coef=np.vstack([np.zeros_like(coef),coef]);bias=np.r_[0,bias]
    k=len(coef);colours=[REGIONS[0],REGIONS[2]] if k==2 else REGIONS
    lo,hi,yl,yh=limits;edges={}
    for a in range(k):
        poly=[np.array(p,dtype=float) for p in [(lo,yl),(hi,yl),(hi,yh),(lo,yh)]]
        for j in range(k):
            if j!=a and poly:poly=clip(poly,coef[a]-coef[j],bias[a]-bias[j])
        if not poly:continue
        ax.add_patch(Polygon(poly,facecolor=colours[a],edgecolor='none',alpha=.9,zorder=0))
        for u,v in zip(poly,poly[1:]+poly[:1]):
            outer=any(np.isclose(u[axis],edge) and np.isclose(v[axis],edge) for axis,edge in [(0,lo),(0,hi),(1,yl),(1,yh)])
            if not outer:edges[tuple(sorted([tuple(np.round(u,7)),tuple(np.round(v,7))]))]=(u,v)
    for u,v in edges.values():ax.plot([u[0],v[0]],[u[1],v[1]],'--',color=GREY,lw=1.1,zorder=2)
    ax.set_xlim(lo,hi);ax.set_ylim(yl,yh);grid(ax)

def limits(x,y,pad=.09):
    dx=max(np.ptp(x),.01);dy=max(np.ptp(y),.01)
    return x.min()-pad*dx,x.max()+pad*dx,y.min()-pad*dy,y.max()+pad*dy

apply_style()
