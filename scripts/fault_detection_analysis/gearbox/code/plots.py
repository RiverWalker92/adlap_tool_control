#!/usr/bin/env python3
"""Complete gearbox figures for multiple runs and leave-one-run-out validation."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyse import (fit_classifier, PARAMETERS, NAMES, COLUMNS,
                     combination_model_definitions, validation_folds, validate_feature_table)


ROTATION_WINDOWS = [
    ('all30', 'All 30\n1–30', 0, 30),
    ('first10', 'First 10\n1–10', 0, 10),
    ('middle10', 'Middle 10\n11–20', 10, 20),
    ('last10', 'Last 10\n21–30', 20, 30),
] + [
    (f'block5_{i+1}',
     ('First 5' if i == 0 else 'Last 5' if i == 5 else '5 rotations')
     + f'\n{5*i+1}–{5*i+5}', 5*i, 5*i+5)
    for i in range(6)
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
            marker = ['D', 'P', 'X', 'v', '*', 'h'][index % 6]
            group = scores[scores.run == run].sort_values('window_order')
            offset = (index-(len(degraded_runs)-1)/2)*.12
            ax.scatter(group.window_order+offset, group.score, color=orange,
                       marker=marker, s=49*scale**2, edgecolors='white', linewidths=.7, zorder=5)
            for row in group.itertuples():
                above = (row.score >= 0) if index % 2 == 0 else (row.score < 0)
                ax.annotate(f'{row.score:+.2f}', (row.window_order+offset, row.score),
                            xytext=(0, (9 if above else -9)*scale),
                            textcoords='offset points', ha='center', va='bottom' if above else 'top',
                            fontsize=9*scale, color=orange, weight='normal', zorder=6)
            label = 'Degraded run' if len(degraded_runs) == 1 else f'Degraded run {index+1}'
            handles.append(Line2D([], [], marker=marker, linestyle='', color=orange,
                                  markersize=5.5*scale, label=label))
        handles.append(Line2D([], [], color=grey, lw=1.3, label='Decision boundary'))
        legend_columns = min(3, len(handles))
        legend_rows = (len(handles)+legend_columns-1)//legend_columns
        legend_handles = [handles[row*legend_columns+col]
                          for col in range(legend_columns)
                          for row in range(legend_rows)
                          if row*legend_columns+col < len(handles)]
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .835),
                   ncol=legend_columns, frameon=False, fontsize=9*scale,
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
        'Gearbox fault detection across rotation windows', limits, [.5, 3.5])
    _draw_rms_window_figure(
        five, estimator, FIVE_ROTATION_BLOCKS, directory, 'lda_rms_five_rotation_blocks',
        'Gearbox fault detection in five-rotation blocks', limits)
    return scores



# Reuse the same shared helper as the original plotter.
import sys
_common = Path(__file__).resolve().parents[2]/'common'
if _common.is_dir() and str(_common) not in sys.path:
    sys.path.insert(0, str(_common))
try:
    import plot_style as _original_style
except ModuleNotFoundError as error:
    if error.name != 'plot_style':
        raise
    _original_style = None

BLUE = getattr(_original_style, 'BLUE', '#3976A8')
ORANGE = getattr(_original_style, 'ORANGE', '#C75B3E')
HEALTHY_MARKERS = list(getattr(_original_style, 'MARKERS', ['o', 's', '^', 'v', 'X', 'h']))
DEGRADED_MARKERS = ['D', 'P', 'X', 'v', '*', 'h']


def apply_original_style():
    if _original_style is not None:
        _original_style.apply_style()
        print('Original shared plot style:', _original_style.__file__, flush=True)
    else:
        # Runnable fallback; the shared helper is authoritative for exact typography.
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
            'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
            'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
            'legend.fontsize': 7, 'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
            'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
        print('NOTICE: common/plot_style.py was not found. Original dimensions are used with fallback typography. Copy the original helper to common/plot_style.py for the exact original shared style.', flush=True)


def original_legend(fig, handles, ncols):
    if _original_style is not None:
        _original_style.figure_legend(fig, handles, ncols)
    else:
        fig.legend(handles=handles, loc='upper center', ncol=ncols, frameon=False,
                   fontsize=7, handletextpad=.3, columnspacing=1)


def boundary_handle():
    if _original_style is not None:
        return _original_style.boundary_handle()
    return Line2D([], [], color='#53616b', ls='--', lw=1, label='Decision boundary')


def run_styles(features):
    styles = {}
    for label, colour, markers, word in [(0, BLUE, HEALTHY_MARKERS, 'Healthy'),
                                          (1, ORANGE, DEGRADED_MARKERS, 'Degraded')]:
        for index, run in enumerate(sorted(features.loc[features.label == label, 'run'].unique())):
            styles[run] = dict(label=label, colour=colour, marker=markers[index % len(markers)],
                               name=f'{word} run {index+1}', index=index)
    return styles


def run_handle(style):
    if _original_style is not None:
        return _original_style.marker_handle(style['name'], style['colour'], style['marker'])
    return Line2D([], [], color=style['colour'], marker=style['marker'], linestyle='',
                  markersize=3.5, label=style['name'])


def save_figure(fig, directory, stem):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # The shared helper controls the PNG settings exactly as in the old plotter.
    if _original_style is not None:
        _original_style.save_png(fig, directory/stem)
    else:
        fig.savefig(directory/(stem+'.png'), dpi=600, bbox_inches='tight', pad_inches=.08)
    fig.savefig(directory/(stem+'.svg'), bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    print('Figure:', directory/(stem+'.png'), flush=True)


def feature_title(column):
    if column == 'shape_corr': return 'Waveform correlation'
    if column == 'ellipse_area': return 'Ellipse area'
    motor, parameter = column.split('_', 1)
    return motor + ' ' + NAMES[parameter]


def feature_unit(column):
    if column == 'shape_corr': return 'Dimensionless'
    if column == 'ellipse_area': return r'mA$^2$'
    if column.endswith('_rc'): return 'mA/rad'
    return 'mA rad' if column.endswith('_auc') else 'mA'


def axis_label(col):
    if col=='shape_corr':return 'M0-M3 waveform correlation'
    if col=='ellipse_area':return 'M0-M3 ellipse area [mA squared]'
    motor,param=col.split('_',1)
    unit='mA rad' if param in ['abs_auc','pos_auc'] else 'mA/rad' if param=='rc' else 'mA'
    return f'{motor} {NAMES[param].lower().replace("rms","RMS").replace("phase sd","phase SD")} [{unit}]'


def boxplots(observations, columns, output, name, ncols=3):
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.75*nrows+.38),
                             squeeze=False)
    healthy_runs = sorted(observations.loc[observations.label == 0, 'run'].unique())
    styles = run_styles(observations)
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
        for box, colour in zip(boxes['boxes'], [BLUE, ORANGE]):
            box.set(facecolor=colour, alpha=.15)
        for i, run in enumerate(healthy_runs):
            values = observations.loc[(observations.label == 0) &
                                      (observations.run == run), column]
            ax.scatter(rng.uniform(-.12,.12,len(values)), values, s=12,
                       color=BLUE, marker=styles[run]['marker'], zorder=3)
        for run, style in styles.items():
            if style['label'] != 1:
                continue
            values = observations.loc[observations.run == run, column]
            ax.scatter(1+rng.uniform(-.12,.12,len(values)), values, s=14,
                       color=ORANGE, marker=style['marker'], zorder=3)
        # Long titles wrap so that neighbouring panels stay separate.
        label = feature_title(column).replace('Fundamental amplitude', 'Fundamental\namplitude')
        label = label.replace('Second harmonic amplitude', 'Second harmonic\namplitude')
        ax.set(xticks=[0,1], xticklabels=['Healthy','Degraded'],
               ylabel=feature_unit(column), title=label)
        ax.grid(axis='y', alpha=.15)
    for i,ax in enumerate(axes.flat):
        if i not in slots:ax.set_visible(False)
    handles = [run_handle(style) for style in styles.values()]
    fig.legend(handles=handles, loc='upper center', ncol=min(5,len(handles)), frameon=False,
               fontsize=7, handletextpad=.3, columnspacing=1)
    fig.tight_layout(rect=(0,0,1,.95))
    save_figure(fig,output,name)


def correlation_figures(five, directory):
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
        fig.tight_layout();save_figure(fig,directory,'correlation_'+scope)


def scatter_runs(ax, observations, columns, styles, alpha=1., annotate=False):
    for run, style in styles.items():
        group = observations[observations.run == run].sort_values('start')
        if group.empty:
            continue
        ax.scatter(group[columns[0]], group[columns[1]], color=style['colour'],
                   marker=style['marker'], s=28, edgecolors='white', linewidths=.4,
                   label=style['name'], alpha=alpha, zorder=3)
        if annotate and style['label'] == 1:
            for block, row in enumerate(group.itertuples(), start=1):
                ax.annotate(str(block), (getattr(row, columns[0]), getattr(row, columns[1])),
                            xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set(xlabel=axis_label(columns[0]), ylabel=axis_label(columns[1]))
    if _original_style is not None:
        _original_style.grid(ax)
    else:
        ax.grid(alpha=.15)


def feature_limits(observations, columns):
    if _original_style is not None:
        lo, hi, yl, yh = _original_style.limits(observations[columns[0]].to_numpy(),
                                               observations[columns[1]].to_numpy())
        return [(lo, hi), (yl, yh)]
    result = []
    for col in columns:
        values = observations[col].to_numpy()
        span = max(float(np.ptp(values)), max(float(abs(values).max()), 1.)*.05)
        result.append((float(values.min())-.15*span, float(values.max())+.15*span))
    return result


def decision_regions(ax, fitted, observations, columns, bounds=None):
    bounds = feature_limits(observations, columns) if bounds is None else bounds
    if _original_style is not None:
        scale, lda = fitted['scale'], fitted['lda']
        co = lda.coef_ / scale.scale_
        bias = lda.intercept_ - co @ scale.mean_
        _original_style.lda_regions(ax, co, bias, (*bounds[0], *bounds[1]))
        return
    xs, ys = [np.linspace(lo, hi, 140) for lo, hi in bounds]
    xx, yy = np.meshgrid(xs, ys)
    frame = pd.DataFrame({columns[0]: xx.ravel(), columns[1]: yy.ravel()})
    z = fitted.decision_function(frame).reshape(xx.shape)
    ax.contourf(xx, yy, (z > 0).astype(int), levels=[-.5, .5, 1.5],
                colors=['#eaf2f8', '#fbece7'], alpha=.8, zorder=0)
    if z.min() < 0 < z.max():
        ax.contour(xx, yy, z, levels=[0], colors=['#53616b'], linestyles='--', linewidths=1.)
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])


def descriptive_lda_figures(five, directory):
    styles = run_styles(five)
    columns = ['M0_rms', 'M3_rms']
    fitted = fit_classifier(five, {'features': columns})
    fig, ax = plt.subplots(figsize=(3.5,3.3))
    fig.subplots_adjust(left=.17,right=.98,bottom=.16,top=.70)
    decision_regions(ax, fitted, five, columns)
    scatter_runs(ax, five, columns, styles, annotate=True)
    # Preserve the old last-block emphasis, now for each degraded run.
    for run, style in styles.items():
        if style['label'] == 1:
            row = five[five.run == run].sort_values('start').iloc[-1]
            ax.scatter(row.M0_rms, row.M3_rms, color=ORANGE, marker='*', s=100,
                       edgecolors='#5A2413',linewidths=.6,zorder=4)
    ax.set(xlabel='M0 residual RMS [mA]', ylabel='M3 residual RMS [mA]',
           title='Paired residual RMS: five-rotation blocks')
    handles = [run_handle(st) for st in styles.values()]
    handles += [Line2D([],[],color=ORANGE,marker='*',ls='',label='Final degraded blocks'), boundary_handle()]
    original_legend(fig,handles,2)
    save_figure(fig,directory,'lda_rms')
    registry = combination_model_definitions()
    fig, axes = plt.subplots(1,3,figsize=(7.16,3.2))
    fig.subplots_adjust(left=.08,right=.99,bottom=.24,top=.73,wspace=.55)
    for ax, model, heading in zip(axes,['G075','G083','G102'],
                                  ['(a) Same motor','(b) Mean and SD','(c) Integral and RMS']):
        columns = registry[model]['features']
        fitted = fit_classifier(five, {'features': columns})
        decision_regions(ax,fitted,five,columns)
        scatter_runs(ax,five,columns,styles)
        ax.set_title(heading)
        ax.set_xlabel(ax.get_xlabel().replace(' second harmonic amplitude',' second harmonic\namplitude'))
    handles = [run_handle(st) for st in styles.values()]+[boundary_handle()]
    original_legend(fig,handles,3)
    save_figure(fig,directory,'lda_mixed_parameters')


def rc_figure(output, five, directory):
    styles = run_styles(five)
    with np.load(output/'rotation_profiles.npz',allow_pickle=False) as profiles:
        phase=profiles['phase'][:-1]
        fig,axes=plt.subplots(1,2,figsize=(7,2.8))
        for ax,index,motor in zip(axes,[0,1],['M0','M3']):
            for run,style in styles.items():
                residual=profiles[run][-5:,:-1,index].mean(axis=0)
                if style['label']==0:
                    ax.plot(phase,residual,color=BLUE,alpha=.4,lw=.8,label=style['name'])
                else:
                    linestyle=['-',':','-.'][style['index']%3]
                    ax.plot(phase,residual,color=ORANGE,ls=linestyle,lw=1.2,
                            label=style['name']+' final block')
                    slope,intercept=np.linalg.lstsq(np.column_stack([phase,np.ones(len(phase))]),residual,rcond=None)[0]
                    ax.plot(phase,slope*phase+intercept,color='#222222',
                            ls=['--',':','-.'][style['index']%3],lw=1,
                            label=f"RC {style['index']+1} = {slope:.3f} mA/rad")
            ax.set(title=motor,xlabel='Rotational phase [rad]',ylabel='Current residual [mA]')
            ax.grid(alpha=.15);ax.legend(fontsize=6,frameon=False)
        fig.tight_layout()
        save_figure(fig,directory,'rc_final_block_fit')


def parameter_evolution(five, directory):
    styles=run_styles(five)
    fig,axes=plt.subplots(2,2,figsize=(7,4.8))
    for ax,panel,param in zip(axes.flat,'ABCD',['peak','rms','phase_sd','mean']):
        for run,style in styles.items():
            group=five[five.run==run].sort_values('start')
            for motor,linestyle,marker in [('M0','-','o'),('M3','--','s')]:
                ax.plot(group.centre,group[motor+'_'+param],c=style['colour'],ls=linestyle,
                        alpha=.4 if style['label']==0 else (.95 if style['index']==0 else .55),
                        lw=.9 if style['label']==0 else 1.7,
                        marker=None if style['label']==0 else marker,
                        ms=4,markeredgecolor='white',markeredgewidth=.3,
                        zorder=1 if style['label']==0 else 3)
        ax.set(title=panel+'  '+NAMES[param],xlabel='Centre of block [rotation]',
               ylabel='mA',xticks=[3,8,13,18,23,28])
        ax.grid(alpha=.15)
    handles=[Line2D([],[],color=ORANGE,ls=ls,marker=marker,lw=1.7,ms=4,label='Degraded '+motor)
             for motor,ls,marker in [('M0','-','o'),('M3','--','s')]]
    handles += [Line2D([],[],color=BLUE,ls=ls,alpha=.4,lw=.9,label='Healthy '+motor)
                for motor,ls in [('M0','-'),('M3','--')]]
    if sum(st['label']==1 for st in styles.values())>1:
        handles += [Line2D([],[],color=ORANGE,lw=1.7,alpha=.95 if st['index']==0 else .55,
                           label=st['name']) for st in styles.values() if st['label']==1]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,
               fontsize=7,handlelength=2.8,columnspacing=1.2)
    fig.tight_layout(rect=(0,0,1,.90 if len(handles)>4 else .93))
    save_figure(fig,directory,'04_parameter_evolution')


def loocv_figures(features, directory):
    styles = run_styles(features)
    cols = ['M0_rms', 'M3_rms']
    for length in (5, 10):
        observations = features[features.n == length]
        folds = list(validation_folds(features, length))
        ncols = min(3, len(folds)); nrows = (len(folds)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 2.7*nrows+.35), squeeze=False)
        fig.subplots_adjust(left=.08,right=.99,bottom=.10,top=.80,wspace=.4,hspace=.65)
        bounds = feature_limits(observations, cols)
        exported = []
        for ax, (train, test, meta) in zip(axes.flat, folds):
            held = meta['held_out_run']
            fitted = fit_classifier(train, {'features': cols})
            decision_regions(ax, fitted, observations, cols, bounds)
            scatter_runs(ax, train, cols, styles, alpha=.2)
            scatter_runs(ax, test, cols, styles)
            predicted = (fitted.decision_function(test[cols]) > 0).astype(int)
            failed = test[predicted != test.label.to_numpy()]
            if not failed.empty:
                ax.scatter(failed[cols[0]], failed[cols[1]], marker='o', facecolors='none',
                           edgecolors='#111111', linewidths=1.2, s=75, zorder=4)
            ax.set_title(f"Withheld: {styles[held]['name']}\n{int((predicted == test.label.to_numpy()).sum())}/{len(test)} correct; {length}-rotation blocks", fontsize=7)
            for row, score, prediction in zip(test.itertuples(), fitted.decision_function(test[cols]), predicted):
                exported.append(dict(n=length, held_out_run=held, run=row.run, window=row.window,
                                     label=row.label, score=float(score), predicted=int(prediction)))
        for ax in list(axes.flat)[len(folds):]:
            ax.set_visible(False)
        handles = [run_handle(st) for st in styles.values()]
        handles += [Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='#111111',
                           linestyle='', markersize=8, label='Incorrect test prediction')]
        original_legend(fig,handles,3)
        stem = f'leave_one_run_out_validation_{length}'
        pd.DataFrame(exported).to_csv(directory/(stem+'.csv'), index=False)
        save_figure(fig, directory, stem)


def margin_figures(output, directory):
    path = output/'cv_margin_ranking.csv'
    if not path.exists():
        print('Margin ranking figures skipped: cv_margin_ranking.csv not found.', flush=True)
        return
    ranking = pd.read_csv(path)
    for length, group in ranking.groupby('n', sort=False):
        top = group[group.margin_rank_eligible == True].head(12).iloc[::-1]
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, max(3.5, .36*len(top)+1.8)))
        pos = np.arange(len(top))
        ax.set_facecolor('#f6f8fa')
        ax.axvline(0, color='#52616b', lw=1.2)
        ax.scatter(top.healthy_margin_min, pos-.09, color=BLUE, label='Healthy: minimum', s=35, zorder=3)
        ax.scatter(top.faulty_margin_min, pos+.09, color=ORANGE, marker='D', label='Degraded: minimum', s=35, zorder=3)
        labels = [r.model+': '+r.features.replace(';', ' + ')+
                  (' *' if r.perfect_classification else '') for r in top.itertuples()]
        ax.set_yticks(pos, labels)
        ax.set_xlabel('Signed margin to LDA boundary [training within-class SD units]')
        ax.set_title(f'Parameter comparison: {int(length)}-rotation blocks\nLeave-one-run-out predictions; * = 100% sensitivity and specificity', fontsize=12)
        ax.grid(axis='x', alpha=.2)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.16), ncol=2, frameon=False)
        fig.tight_layout()
        save_figure(fig, directory, f'lda_margin_ranking_{int(length)}')


def run(output):
    output = Path(output)
    directory = output/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(output/'features.csv')
    validate_feature_table(features)
    five = features[features.n == 5]
    styles = run_styles(features)
    pd.DataFrame([dict(run=run, display_name=st['name'], label=st['label'], marker=st['marker'])
                  for run, st in styles.items()]).to_csv(directory/'run_labels.csv', index=False)
    apply_original_style()
    rms_rotation_windows(output, directory)
    boxplots(five, ['M0_'+p for p in PARAMETERS], directory, 'boxplots_M0', 3)
    boxplots(five, ['M3_'+p for p in PARAMETERS]+['shape_corr', 'ellipse_area'],
             directory, 'boxplots_M3_joint', 4)
    correlation_figures(five, directory)
    descriptive_lda_figures(five, directory)
    rc_figure(output, five, directory)
    parameter_evolution(five, directory)
    loocv_figures(features, directory)
    margin_figures(output, directory)
    (directory/'figure_protocol.json').write_text(json.dumps(dict(
        descriptive_figures=['lda_rms', 'lda_mixed_parameters', 'lda_rms_rotation_windows',
                             'lda_rms_five_rotation_blocks'],
        descriptive_fit='All five-rotation blocks of every run. Not held-out validation.',
        validation_figures=['leave_one_run_out_validation_5', 'leave_one_run_out_validation_10'],
        validation_fit='Exactly one entire run excluded from scaler and LDA fitting in every fold.',
        replaced_legacy_figure='last_five_validation is replaced by run-wise LOOCV figures.',
        run_mapping='run_labels.csv', formats=['png', 'svg'],
        style_helper=str(_original_style.__file__) if _original_style is not None else None,
        style_reference='Original plots.py dimensions, subplot margins, labels and shared plot_style.py; LOOCV gets one panel per complete held-out run.'), indent=2))
    print('Full figure set completed:', directory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs')
    parser.add_argument('--rotation-windows-only', action='store_true',
                        help='Only generate the two descriptive RMS rotation-window figures.')
    args = parser.parse_args()
    print('Gearbox figures — complete set; 30/10/all 5-rotation windows', flush=True)
    print('Running script:', Path(__file__).resolve(), flush=True)
    if args.rotation_windows_only:
        apply_original_style()
        rms_rotation_windows(args.output)
    else:
        run(args.output)
