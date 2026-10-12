#!/usr/bin/env python3
"""Export CV performance and descriptive full-run counts to cv_table.tex.

Reads existing numerical outputs; it does not fit classifiers.
"""
import argparse
from pathlib import Path
import pandas as pd

PARAMETERS = ['peak', 'abs_peak', 'mean', 'abs_auc', 'pos_auc', 'rms',
              'phase_sd', 'ptp', 'a1', 'a2', 'rc']
NAMES = dict(zip(PARAMETERS, ['Signed maximum', 'Absolute maximum',
    'Phase mean', 'Absolute integral', 'Positive integral', 'RMS',
    'Phase SD', 'Peak to peak range', 'Fundamental amplitude',
    'Second harmonic amplitude', 'RC coefficient']))
MAIN_MODELS = ['M0_M3_'+p for p in PARAMETERS]+['shape_corr','ellipse_area']


def display_name(model):
    if model == 'shape_corr': return 'Waveform correlation'
    if model == 'ellipse_area': return 'Ellipse area'
    return NAMES[model.removeprefix('M0_M3_')]


def full_run_counts(predictions, model):
    required={'model','n','window','run','label','predicted'}
    if not required.issubset(predictions.columns):
        raise ValueError('descriptive_predictions.csv lacks required columns: '+
                         ', '.join(sorted(required-set(predictions.columns))))
    group=predictions[(predictions.model==model)&(predictions.n==30)&
                      (predictions.window=='all30')].copy()
    if group.empty:
        raise ValueError(f'No full thirty rotation predictions for {model}.')
    if group.run.duplicated().any():
        raise ValueError('Full-run counts require one prediction per run and classifier.')
    if not set(group.label).issubset({0,1}) or not set(group.predicted).issubset({0,1}):
        raise ValueError('Expected binary labels and predictions: 0=healthy, 1=degraded.')
    healthy=group[group.label==0]
    degraded=group[group.label==1]
    if healthy.empty or degraded.empty:
        raise ValueError('Both experimental conditions are required for the full-run summary.')
    return (int((degraded.predicted==1).sum()),len(degraded),
            int((healthy.predicted==0).sum()),len(healthy))


def write_main_table(output, target, descriptive_models=('M0_M3_rms',)):
    output,target=Path(output),Path(target)
    summary=pd.read_csv(output/'cv_summary.csv')
    if summary.duplicated(['n','model']).any():
        raise ValueError('Duplicate cross-validation summaries.')
    cv=summary.set_index(['n','model'])
    predictions=pd.read_csv(output/'descriptive_predictions.csv')
    rows=[r'\begin{table}[htbp]',r'\centering\small',
          r'\setlength{\tabcolsep}{3pt}',
          r'\caption{Classification performance for five and ten rotation blocks. Motor parameter rows use M0 and M3 together; joint parameters were evaluated individually. Cross-validation percentages describe pooled, repeated test decisions. The thirty rotation counts describe fixed classifiers fitted to all five rotation blocks and reuse training measurements. They are descriptive outcomes.}',
          r'\label{tab:gearbox150_cv}',
          r'\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrrrr@{}}',
          r'\toprule & \multicolumn{2}{c}{Five rotations} & \multicolumn{2}{c}{Ten rotations}\\',
          r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
          r'LDA input & Sens. [\%] & Spec. [\%] & Sens. [\%] & Spec. [\%]\\\midrule']
    for model in MAIN_MODELS:
        if model=='shape_corr':rows.append(r'\midrule')
        values=[]
        for length in (5,10):
            if (length,model) not in cv.index:
                raise ValueError(f'Missing CV results for {model}, {length} rotations.')
            record=cv.loc[(length,model)]
            for metric in ('sensitivity','specificity'):
                value=float(record[metric])
                if not 0<=value<=1:raise ValueError(f'Invalid {metric} for {model}.')
                values.append(f'{100*value:.1f}')
        rows.append(display_name(model)+' & '+' & '.join(values)+r'\\')
    rows.extend([r'\midrule',
                 r'\multicolumn{5}{@{}l}{\textit{Thirty rotations: descriptive evaluation}}\\'])
    for model in descriptive_models:
        if model not in MAIN_MODELS:raise ValueError('Unknown descriptive classifier: '+model)
        tp,total_degraded,tn,total_healthy=full_run_counts(predictions,model)
        name=display_name(model)
        if model.startswith('M0_M3_'):name += ', M0 and M3'
        rows.append(name+f' & \\multicolumn{{2}}{{c}}{{Degraded: {tp}/{total_degraded}}}'
                    +f' & \\multicolumn{{2}}{{c@{{}}}}{{Healthy: {tn}/{total_healthy}}}'+r'\\')
    rows.extend([r'\bottomrule\end{tabularx}',r'\end{table}'])
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(rows)+'\n')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output'))
    parser.add_argument('--overleaf',type=Path,default=Path('overleaf'))
    parser.add_argument('--descriptive-models',nargs='+',default=['M0_M3_rms'])
    args=parser.parse_args()
    target=write_main_table(args.output,args.overleaf/'cv_table.tex',args.descriptive_models)
    print('Generated',target,'from recorded prediction outputs.')


if __name__=='__main__':main()
