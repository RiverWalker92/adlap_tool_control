"""Gearbox figures; RMS rotation-window comparison exports PNG and SVG."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyse import PARAMETERS,NAMES,COLUMNS,fit_classifier
try:
    from plot_style import *
    _PLOT_STYLE_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != 'plot_style':
        raise
    # The standalone rotation-window figure does not need this external helper.
    _PLOT_STYLE_AVAILABLE = False
    BLUE, ORANGE = '#3976A8', '#C75B3E'
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


ROTATION_WINDOWS = [
    ('all30', 'All 30\n1–30', 0, 30),
    ('first20', 'First 20\n1–20', 0, 20),
    ('last20', 'Last 20\n11–30', 10, 30),
    ('first10', 'First 10\n1–10', 0, 10),
    ('middle10', 'Middle 10\n11–20', 10, 20),
    ('last10', 'Last 10\n21–30', 20, 30),
] + [
    (f'block5_{i+1}',
     ('First 5 rotations' if i == 0 else 'Last 5 rotations' if i == 5 else '5 rotations')
     + f'\n{5*i+1}–{5*i+5}', 5*i, 5*i+5)
    for i in (0, 3, 5)
]

FIVE_ROTATION_BLOCKS = [
    (f'block5_{i+1}', f'Block {i+1}\n{5*i+1}–{5*i+5}', 5*i, 5*i+5)
    for i in range(6)
]


def rotation_window_scores(features, windows=None, estimator=None):
    """Apply one fixed, descriptive RMS model to the selected windows.

    Fit on all non-overlapping five-rotation blocks, exactly as the
    M0_M3_rms descriptive model in analyse.py. No window-specific refitting,
    CV-fold averaging, peak parameters, or true-label sign reversal.
    """
    windows = ROTATION_WINDOWS if windows is None else windows
    columns = ['M0_rms', 'M3_rms']
    required = ['run', 'label', 'window', 'n', 'start', 'end'] + columns
    missing = sorted(set(required) - set(features.columns))
    if missing:
        raise ValueError('features.csv is missing columns: ' + ', '.join(missing))
    if features.duplicated(['run', 'window']).any():
        raise ValueError('features.csv contains duplicate run/window observations.')
    if set(features.label) != {0, 1} or (features.groupby('run').label.nunique() != 1).any():
        raise ValueError('Each run must have one consistent label: healthy=0 or degraded=1.')
    if not np.isfinite(features[columns].to_numpy()).all():
        raise ValueError('RMS features must be finite.')
    runs = sorted(features.run.unique())
    # Require a complete common basis: six disjoint five-rotation blocks/run.
    train = features[features.n == 5]
    for run in runs:
        group = train[train.run == run].sort_values('start')
        if (list(group.window) != [f'block5_{i+1}' for i in range(6)] or
                list(group.start) != list(range(0, 30, 5)) or
                list(group.end) != list(range(5, 31, 5))):
            raise ValueError(f'Run {run} needs six complete five-rotation training blocks.')
    selected = []
    for order, (window, label, start, end) in enumerate(windows):
        group = features[features.window == window].copy()
        if set(group.run) != set(runs):
            raise ValueError(f'Missing {window} observations for one or more runs.')
        if not ((group.start == start) & (group.end == end) & (group.n == end-start)).all():
            raise ValueError(f'Incorrect rotation bounds for window {window}.')
        group['window_order'] = order
        group['window_label'] = label.replace('\n', ': rotations ')
        selected.append(group)
    scores = pd.concat(selected, ignore_index=True)
    if estimator is None:
        estimator = fit_classifier(train, {'features': columns})
    scores['score'] = estimator.decision_function(scores[columns])
    scores['predicted'] = (scores.score > 0).astype(int)
    return scores, estimator


def _draw_rms_window_figure(scores, estimator, windows, directory, filename,
                            heading, y_limits, section_breaks=()):
    """Shared layout and score scale for the overview and complete block plot."""
    healthy_runs = sorted(scores.loc[scores.label == 0, 'run'].unique())
    degraded_runs = sorted(scores.loc[scores.label == 1, 'run'].unique())
    blue, orange, grey = '#326F9F', '#BC5036', '#53616B'
    wide = len(windows) > 8
    scale = 1.25 if wide else 1.
    style = {
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'font.size': 10.5*scale, 'font.weight': 'normal',
        'mathtext.fontset': 'stix', 'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'axes.labelsize': 11*scale, 'axes.titlesize': 14*scale,
        'xtick.labelsize': 10*scale, 'ytick.labelsize': 10*scale,
        'axes.grid': False, 'axes.linewidth': .8,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'axes.unicode_minus': True,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(12.8, 5.6) if wide else (9.2, 4.9))
        fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.73)
        fig.text(.5, .95, heading,
                 fontsize=14*scale, weight='normal', ha='center', va='top')
        fig.text(.5, .89, 'LDA using the residual RMS of M0 and M3',
                 fontsize=11*scale, weight='normal',
                 ha='center', va='top')
        low, high = y_limits
        ax.set_ylim(low, high)
        ax.set_xlim(-.55, len(windows)-.45)
        ax.axhspan(low, 0, color='#EAF2F8', zorder=0)
        ax.axhspan(0, high, color='#FBEDE9', zorder=0)
        ax.axhline(0, color=grey, lw=1.3, zorder=2)
        for cut in section_breaks:
            ax.axvline(cut, color='white', lw=2, zorder=1)
        ax.grid(axis='y', color='#87929A', alpha=.15, lw=.7, zorder=1)
        markers = ['o', 's', '^', 'v', 'P', 'X']
        offsets = np.linspace(-.14, .14, len(healthy_runs)) if len(healthy_runs) > 1 else [0.]
        handles = []
        for index, (run, offset) in enumerate(zip(healthy_runs, offsets)):
            group = scores[scores.run == run].sort_values('window_order')
            marker = markers[index % len(markers)]
            ax.scatter(group.window_order+offset, group.score, color=blue,
                       marker=marker, s=31*scale**2, edgecolors='white', linewidths=.6, zorder=4)
            handles.append(Line2D([], [], marker=marker, linestyle='', color=blue,
                                  markersize=5*scale, label=f'Healthy run {index+1}'))
        for index, run in enumerate(degraded_runs):
            group = scores[scores.run == run].sort_values('window_order')
            offset = (index-(len(degraded_runs)-1)/2)*.12
            ax.scatter(group.window_order+offset, group.score, color=orange,
                       marker='D', s=49*scale**2, edgecolors='white', linewidths=.7, zorder=5)
            for row in group.itertuples():
                ax.annotate(f'{row.score:+.2f}', (row.window_order+offset, row.score),
                            xytext=(0, (9 if row.score >= 0 else -14)*scale),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=9*scale, color=orange, weight='normal', zorder=6)
            label = 'Degraded run' if len(degraded_runs) == 1 else f'Degraded run {index+1}'
            handles.append(Line2D([], [], marker='D', linestyle='', color=orange,
                                  markersize=5.5*scale, label=label))
        handles.append(Line2D([], [], color=grey, lw=1.3, label='Decision boundary'))
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.095, .835),
                   ncol=min(5, len(handles)), frameon=False, fontsize=9*scale,
                   handletextpad=.45, columnspacing=1.1, borderaxespad=0)
        ax.set_xticks(range(len(windows)), [item[1] for item in windows])
        ax.set_xlabel('Rotation windows', labelpad=10)
        ax.set_ylabel('LDA decision score', labelpad=8)
        ax.tick_params(axis='both', length=3, color='#7F8991')
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#7F8991')
        stem = directory/filename
        for extension in ['png', 'svg']:
            fig.savefig(stem.with_suffix('.'+extension), dpi=600, bbox_inches='tight', pad_inches=.08)
        plt.close(fig)
    export = ['run', 'label', 'window', 'window_order', 'window_label',
              'n', 'start', 'end', 'M0_rms', 'M3_rms', 'score', 'predicted']
    scores[export].to_csv(stem.with_suffix('.csv'), index=False)
    weights = estimator['lda'].coef_[0] / estimator['scale'].scale_
    metadata = {
        'model': 'M0_M3_rms', 'features': ['M0_rms', 'M3_rms'],
        'fit_scope': 'All six non-overlapping five-rotation blocks of every run; descriptive fit.',
        'score': 'Raw LDA decision_function; no true-label sign reversal or normalization.',
        'decision_boundary': 0,
        'y_axis_limits': [float(low), float(high)],
        'windows': [{'window': item[0], 'first_rotation': item[2]+1,
                     'last_rotation': item[3]} for item in windows],
        'positive_region': 'Predicted degraded', 'negative_region': 'Predicted healthy',
        'feature_extraction': 'RMS of the averaged residual profile, as stored in features.csv.',
        'run_labels': {f'Healthy run {i+1}': str(run) for i, run in enumerate(healthy_runs)},
        'degraded_runs': [str(run) for run in degraded_runs],
        'raw_feature_weights': weights.tolist(),
        'raw_feature_intercept': float(estimator['lda'].intercept_[0] - weights @ estimator['scale'].mean_),
        'interpretation': 'Descriptive window comparison, not independent cross-validation. Model training includes these measurements. The overview includes overlapping windows; five-rotation blocks are disjoint within each run but are not independent runs.'}
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'RMS rotation-window figure: {stem.with_suffix(".png")}')
    print(f'Vector export: {stem.with_suffix(".svg")}')


def rms_rotation_windows(output, directory=None):
    """Create BOTH RMS figures with one fitted model and identical y limits."""
    output = Path(output)
    directory = Path(directory) if directory is not None else output/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(output/'features.csv')
    scores, estimator = rotation_window_scores(features)
    five, _ = rotation_window_scores(features, FIVE_ROTATION_BLOCKS, estimator)
    values = np.r_[scores.score.to_numpy(), five.score.to_numpy(), 0.]
    span = max(float(np.ptp(values)), 1.)
    limits = (min(float(values.min())-.16*span, -.10*span),
              max(float(values.max())+.20*span, .12*span))
    _draw_rms_window_figure(
        scores, estimator, ROTATION_WINDOWS, directory, 'lda_rms_rotation_windows',
        'Gearbox fault detection across rotation windows', limits, [.5, 2.5, 5.5])
    _draw_rms_window_figure(
        five, estimator, FIVE_ROTATION_BLOCKS, directory, 'lda_rms_five_rotation_blocks',
        'Gearbox fault detection in five-rotation blocks', limits)
    return scores


def run(output):
    if not _PLOT_STYLE_AVAILABLE:
        raise ModuleNotFoundError('The full figure set needs plot_style.py. Use --rotation-windows-only for the standalone RMS window figure.')
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    apply_style();features=pd.read_csv(output/'features.csv');five=features[features.n==5]
    rms_rotation_windows(output, figures)
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs')
    p.add_argument('--rotation-windows-only',action='store_true',
                   help='Create both RMS figures (window overview and all five-rotation blocks) from features.csv.')
    args=p.parse_args()
    if args.rotation_windows_only:
        rms_rotation_windows(args.output)
    else:
        run(args.output)
