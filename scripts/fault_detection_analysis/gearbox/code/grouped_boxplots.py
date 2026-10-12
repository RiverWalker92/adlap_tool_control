#!/usr/bin/env python3
"""Draw grouped gearbox boxplots directly from the analysis features.csv.

This script reproduces the grouped layout from numerical observations.
It does not calculate a new RC parameter or refit classifiers.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PARAMETERS = ['peak', 'abs_peak', 'mean', 'abs_auc', 'pos_auc', 'rms',
              'phase_sd', 'ptp', 'a1', 'a2', 'rc']
NAMES = dict(zip(PARAMETERS, ['Signed maximum', 'Absolute maximum',
    'Phase mean', 'Absolute integral', 'Positive integral', 'RMS',
    'Phase SD', 'Peak to peak range', 'Fundamental amplitude',
    'Second harmonic amplitude', 'RC coefficient']))
COLOURS = ['#176b9c', '#479cbb', '#263e62']
MARKERS = ['o', 's', '^']
DEGRADED = '#c34525'


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
    output.mkdir(parents=True, exist_ok=True)
    pending = output / (name+'.pending.png')
    fig.savefig(pending, format='png', dpi=220, bbox_inches='tight')
    pending.replace(output/(name+'.png'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--columns', type=int, choices=[2,3,4], default=None,
                        help='Override the defaults: three M0 columns and four M3 columns.')
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    if not {'label','run','n'}.issubset(frame.columns):
        raise ValueError('Expected label, run and n columns from the gearbox analysis.')
    observations = frame.loc[frame.n == 5].copy()
    if not set(observations.label.unique()).issubset({0,1}):
        raise ValueError('Expected labels 0=healthy and 1=degraded.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,
                         'axes.titlesize':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    draw(observations, ['M0_'+p for p in PARAMETERS], args.output,
         'boxplots_M0', args.columns or 3)
    draw(observations, ['M3_'+p for p in PARAMETERS]+['shape_corr','ellipse_area'],
         args.output, 'boxplots_M3_joint', args.columns or 4)


if __name__ == '__main__':
    main()
