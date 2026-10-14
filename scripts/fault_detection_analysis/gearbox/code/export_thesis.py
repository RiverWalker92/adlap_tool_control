#!/usr/bin/env python3
"""Export the thesis RMS figure and verify all pooled CV counts.

python export_thesis.py --output /path/to/gearbox/outputs --destination thesis
Requires matplotlib; uses recorded coefficients without refitting LDA.
"""
import argparse
import csv
import json
from pathlib import Path

def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path('thesis'))
    args = parser.parse_args()
    source, dest = args.output, args.destination
    dest.mkdir(parents=True, exist_ok=True)
    predictions = read(source / 'cv_predictions.csv')
    summaries = read(source / 'cv_summary.csv')
    for row in summaries:
        group = [p for p in predictions if p['n'] == row['n'] and p['model'] == row['model']]
        for key, actual, predicted in [('TP', 1, 1), ('FN', 1, 0), ('TN', 0, 0), ('FP', 0, 1)]:
            count = sum(int(p['label']) == actual and int(p['predicted']) == predicted for p in group)
            assert count == int(row[key]), (row['n'], row['model'], key, count, row[key])
    models = [('M0_M3_'+key, name) for key, name in [
        ('peak', 'Signed maximum'), ('abs_peak', 'Absolute maximum'),
        ('mean', 'Phase mean'), ('abs_auc', 'Absolute integral'),
        ('pos_auc', 'Positive integral'), ('rms', 'RMS'), ('phase_sd', 'Phase SD'),
        ('ptp', 'Peak to peak range'), ('a1', 'Fundamental amplitude'),
        ('a2', 'Second harmonic amplitude'), ('rc', 'RC coefficient')]]
    models += [('shape_corr', 'Waveform correlation'), ('ellipse_area', 'Ellipse area')]
    descriptive = read(source / 'descriptive_predictions.csv')
    lookup = {(r['n'], r['model']): r for r in summaries}
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Gearbox LDA results using paired M0 and M3 parameters or individual joint parameters. Thirty-rotation counts are descriptive; ten- and five-rotation counts are pooled cross-validation decisions with repeated testing. Degraded is the positive class.}',
             r'\label{tab:lda_m0_m3}', r'\label{tab:gearbox150_cv}',
             r'\scriptsize', r'\setlength{\tabcolsep}{2.5pt}',
             r'\providecommand{\gbcorrect}[1]{\cellcolor{green!15}#1}',
             r'\providecommand{\gbincorrect}[1]{\cellcolor{red!15}#1}',
             r'\resizebox{\columnwidth}{!}{%',
             r'\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm} cccc !{\color{black!25}\vrule width 0.4pt} cccc !{\color{black!25}\vrule width 0.4pt} cccc}',
             r'\toprule',
             r'& \multicolumn{4}{c}{\textbf{30 rotations}} & \multicolumn{4}{c}{\textbf{10 rotations}} & \multicolumn{4}{c}{\textbf{5 rotations}} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}',
             r'\textbf{Parameters} & TP & FP & TN & FN & TP & FP & TN & FN & TP & FP & TN & FN \\',
             r'\midrule']
    for model_id, name in models:
        full = [r for r in descriptive if r['model'] == model_id and r['n'] == '30' and r['window'] == 'all30']
        assert len(full) == 4 and len({r['run'] for r in full}) == 4
        counts = {key: sum(int(r['label']) == actual and int(r['predicted']) == predicted for r in full)
                  for key, actual, predicted in [('TP', 1, 1), ('FP', 0, 1), ('TN', 0, 0), ('FN', 1, 0)]}
        values = []
        for n in ('30', '10', '5'):
            row = counts if n == '30' else lookup[n, model_id]
            for key in ('TP', 'FP', 'TN', 'FN'):
                value = str(row[key])
                if model_id == 'M0_M3_rms':
                    value = r'\gbcorrect{' + value + '}'
                elif model_id == 'M0_M3_rc' and n == '30':
                    value = r'\gbincorrect{' + value + '}'
                values.append(value)
        lines.append(name + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}']
    (dest / 'cv_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    blocks = [r for r in read(source / 'features.csv') if r['n'] == '5' and r['window'].startswith('block5_')]
    healthy = [r for r in blocks if r['label'] == '0']
    degraded = sorted([r for r in blocks if r['label'] == '1'], key=lambda r: int(r['start']))
    assert len(healthy) == 18 and len(degraded) == 6
    model = json.loads((source / 'descriptive_coefficients.json').read_text())['M0_M3_rms']
    assert model['features'] == ['M0_rms', 'M3_rms']
    w0, w3 = model['weights']
    intercept = model['intercept']
    for row in blocks:
        score = w0 * float(row['M0_rms']) + w3 * float(row['M3_rms']) + intercept
        assert int(score > 0) == int(row['label'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none'
    })

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    fig.subplots_adjust(left=.15, right=.98, bottom=.14, top=.72)

    fig.suptitle('Paired residual RMS: pooled 5-rotation windows', fontsize=11, y=.97)

    # kleuren zoals in je voorbeeldfiguur
    healthy_color = '#2E7DB2'
    degraded_color = '#C95A2A'
    healthy_bg = '#DCE8F2'
    degraded_bg = '#F2E4E1'
    boundary_color = '#4A4A4A'

    xlim = (5.2, 13.3)
    ylim = (2.9, 8.0)

    # achtergrondregio's op basis van LDA-score
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 400),
        np.linspace(ylim[0], ylim[1], 400)
    )
    zz = w0 * xx + w3 * yy + intercept

    ax.contourf(
        xx, yy, zz,
        levels=[-1e9, 0, 1e9],
        colors=[healthy_bg, degraded_bg],
        alpha=0.9,
        zorder=0
    )

    # healthy runs
    healthy_runs = sorted({r['run'] for r in healthy})
    healthy_markers = ['o', 's', '^']

    for index, run in enumerate(healthy_runs):
        rows = [r for r in healthy if r['run'] == run]
        ax.scatter(
            [float(r['M0_rms']) for r in rows],
            [float(r['M3_rms']) for r in rows],
            marker=healthy_markers[index % len(healthy_markers)],
            color=healthy_color,
            s=42,
            edgecolors='white',
            linewidths=.5,
            label=f'Healthy run {index+1}',
            zorder=3
        )

    # degraded windows
    offsets = [(6, 4), (6, -2), (-12, -2), (6, -1), (6, 2), (8, 6)]

    for index, row in enumerate(degraded):
        x, y = float(row['M0_rms']), float(row['M3_rms'])

        if index == 5:
            ax.scatter(
                x, y,
                marker='*',
                s=180,
                color=degraded_color,
                edgecolors='#5A2413',
                linewidths=.7,
                zorder=5
            )
        else:
            ax.scatter(
                x, y,
                marker='D',
                s=46,
                color=degraded_color,
                edgecolors='white',
                linewidths=.5,
                zorder=4
            )

        ax.annotate(
            str(index + 1),
            (x, y),
            xytext=offsets[index],
            textcoords='offset points',
            fontsize=8,
            zorder=6
        )

    # LDA-boundary
    x_line = np.linspace(xlim[0], xlim[1], 200)
    y_line = (-intercept - w0 * x_line) / w3
    ax.plot(
        x_line, y_line,
        '--',
        color=boundary_color,
        lw=1.4,
        zorder=2
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('M0 residual RMS [mA]', fontsize=9)
    ax.set_ylabel('M3 residual RMS [mA]', fontsize=9)

    ax.set_xticks(range(6, 14))
    ax.set_yticks(range(3, 9))
    ax.grid(color='#BFBFBF', lw=.35, alpha=.45)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker='o', color=healthy_color, ls='', markersize=6, markeredgecolor='white'),
        Line2D([], [], marker='s', color=healthy_color, ls='', markersize=6, markeredgecolor='white'),
        Line2D([], [], marker='^', color=healthy_color, ls='', markersize=6, markeredgecolor='white'),
        Line2D([], [], marker='D', color=degraded_color, ls='', markersize=6, markeredgecolor='white'),
        Line2D([], [], marker='*', color=degraded_color, markeredgecolor='#5A2413', ls='', markersize=11),
        Line2D([], [], color=boundary_color, ls='--', lw=1.4)
    ]
    labels = [
        'Healthy run 1',
        'Healthy run 2',
        'Healthy run 3',
        'Degraded run',
        'Final degraded window',
        'LDA boundary'
    ]

    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(.5, .91),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handlelength=1.6
    )

    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(dest / f'gearbox150_lda.{extension}', dpi=300)

    plt.close(fig)

    (dest / 'verification.txt').write_text(
        f'All {len(summaries)} pooled CV summaries match the individual predictions.\n'
        'Positive class: degraded (1); negative class: healthy (0).\n'
        'Figure: 18 healthy blocks and 6 degraded blocks, each representing a 5-rotation window.\n'
        'Recorded coefficients used directly; no model refitting or digitisation.\n',
        encoding='utf-8'
    )

    print('Verified counts and exported figure to', dest.resolve())
if __name__ == '__main__':
    main()