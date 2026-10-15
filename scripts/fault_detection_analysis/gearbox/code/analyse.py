#!/usr/bin/env python3
"""Reproduce the gearbox parameter study from an explicit run manifest.

Run from any directory: python3 code/analyse.py --config config.json
All residuals come from the supplied Motor DT replays; the DT is not refitted.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr, theilslopes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PARAMETERS = ['peak', 'abs_peak', 'mean', 'abs_auc', 'pos_auc', 'rms',
              'phase_sd', 'ptp', 'a1', 'a2', 'rc']
NAMES = dict(zip(PARAMETERS, ['Signed maximum', 'Absolute maximum', 'Phase mean',
    'Absolute integral', 'Positive integral', 'RMS', 'Phase SD',
    'Peak to peak range', 'Fundamental amplitude', 'Second harmonic amplitude',
    'RC coefficient']))
COLUMNS = [f'{m}_{p}' for m in ('M0', 'M3') for p in PARAMETERS] + ['shape_corr', 'ellipse_area']
MAIN_MODELS = ['M0_M3_' + p for p in PARAMETERS] + ['shape_corr', 'ellipse_area']
FORWARD_MODELS = ['M0_M3_peak', 'M0_M3_abs_auc', 'M0_M3_rms',
                  'M0_M3_phase_sd', 'M0_M3_rc']
TRAP = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def model_definitions():
    definitions = {}
    for parameter in PARAMETERS:
        for motors in [('M0',), ('M3',), ('M0', 'M3')]:
            definitions['_'.join(motors) + '_' + parameter] = {
                'features': [f'{m}_{parameter}' for m in motors]}
    for column in ['shape_corr', 'ellipse_area']:
        definitions[column] = {'features': [column]}
    assert len(definitions) == 35
    return definitions


def fit_classifier(train, definition):
    """Fit standardisation and shrinkage LDA on this training partition."""
    if set(train.label) != {0, 1}:
        raise ValueError('Binary LDA requires both classes in training.')
    steps = [('scale', StandardScaler())]
    steps.append(('lda', LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])))
    estimator = Pipeline(steps)
    estimator.fit(train[definition['features']], train.label)
    return estimator


def load_rotations(path, config):
    records = [json.loads(line) for line in path.open() if line.strip()]
    records = [r for r in records if r['offline_commanded_target'][0] is not None
               and r['measured_positions'][0] is not None]
    t = np.array([r['time'] for r in records], float)
    positions = np.array([r['measured_positions'] for r in records], float)
    commands = np.array([r['offline_commanded_target'] for r in records], float)
    residuals = np.array([r['filtered_current_deviation'] for r in records], float)[:, [0, 3]]
    if not np.all(np.diff(t) > 0):
        raise ValueError(f'Non-increasing time in {path}')
    moving = np.flatnonzero(np.diff(commands[:, 0]) / np.diff(t) < config['motion_speed_threshold'])
    if not len(moving):
        raise ValueError(f'No eligible rotational movement in {path}')
    start, end = t[moving[0]], t[moving[-1] + 1]
    low, high = config['command_limits']
    mask = ((t >= start + config['edge_margin_seconds']) &
            (t <= end - config['edge_margin_seconds']) &
            (np.max(abs(positions[:, [0, 3]] - commands[:, [0, 3]]), axis=1) < config['position_error_limit']) &
            (commands[:, 0] > low) & (commands[:, 0] < high))
    if not np.isfinite(residuals[mask]).all():
        raise ValueError(f'Non-finite current residuals in retained data: {path}')
    # np.unique retains the first measurement at repeated positions, matching
    # the original analysis. Log the duplicate counts instead of hiding them.
    u, indices = np.unique(-positions[mask, 0], return_index=True)
    values = residuals[mask][indices]
    first, count = config['first_rotation_index'], config['retained_rotations']
    pulses, samples = config['encoder_pulses'], config['phase_samples']
    if u.min() > first * pulses or u.max() < (first + count) * pulses:
        raise ValueError(f'Insufficient encoder coverage in {path}; extrapolation is forbidden.')
    rotations, sampling = [], []
    for index in range(first, first + count):
        grid = np.linspace(index * pulses, (index + 1) * pulses, samples + 1)
        rotations.append(np.column_stack([np.interp(grid, u, values[:, m]) for m in range(2)]))
        selection = mask & (-positions[:, 0] >= index * pulses) & (-positions[:, 0] < (index + 1) * pulses)
        sampling.append({'rotation': index - first + 1, 'logged_samples': int(selection.sum()),
            'unique_encoder_positions': len(np.unique(positions[selection, 0]))})
    return np.array(rotations), sampling, {'valid_replay_rows': len(records),
        'median_log_interval_s': float(np.median(np.diff(t))), 'motion_start_s': start,
        'motion_end_s': end, 'retained_input_samples': int(mask.sum()),
        'unique_retained_positions': len(u), 'duplicate_positions_removed': int(mask.sum()) - len(u)}


def extract_parameters(profile, phase):
    """Extract from the averaged profile, not by averaging cycle parameters.

    The closing endpoint is used for trapezoidal integration. It is excluded
    from RMS, population SD, FFT, correlation and covariance calculations.
    """
    result = {}
    for i, motor in enumerate(['M0', 'M3']):
        full, unique = profile[:, i], profile[:-1, i]
        fourier = np.fft.rfft(unique) / len(unique)
        numbers = [full.max(), abs(full).max(), TRAP(full, phase) / (2 * np.pi),
            TRAP(abs(full), phase), TRAP(np.maximum(full, 0), phase),
            np.sqrt(np.mean(unique ** 2)), np.std(unique, ddof=0), np.ptp(full),
            2 * abs(fourier[1]), 2 * abs(fourier[2]),
            residual_phase_slope(unique, phase[:-1])]
        result.update({f'{motor}_{p}': float(value) for p, value in zip(PARAMETERS, numbers)})
    result['shape_corr'] = float(np.corrcoef(profile[:-1].T)[0, 1])
    covariance = np.cov(profile[:-1].T, ddof=0)
    result['ellipse_area'] = float(np.pi * np.sqrt(max(0, np.linalg.det(covariance))))
    if not all(np.isfinite(list(result.values()))):
        raise ValueError('Undefined parameter, e.g. correlation of a constant profile.')
    return result


def residual_phase_slope(residual, phase):
    """OLS slope with an intercept; residual as y, phase as x (mA/rad).

    Use the 150 unique phase points, excluding the duplicated closing
    endpoint. The phase origin is fixed by the existing encoder segmentation.
    This is a waveform parameter, not a trend across successive rotations.
    """
    x, y = np.asarray(phase, float), np.asarray(residual, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError('The phase and residual vectors must have equal lengths.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite phase or residual values.')
    centred_phase = x - x.mean()
    denominator = centred_phase @ centred_phase
    if denominator <= 0:
        raise ValueError('A slope requires variation in phase.')
    return float(centred_phase @ (y-y.mean()) / denominator)


def observation_windows():
    # Five- and ten-rotation partitions are disjoint. The twenty-rotation
    # windows overlap and are descriptive, as is the full thirty-rotation window.
    windows = [(f'block5_{i+1}', 5*i, 5*(i+1)) for i in range(6)]
    windows += [('first10', 0, 10), ('middle10', 10, 20), ('last10', 20, 30),
                ('first20', 0, 20), ('last20', 10, 30), ('all30', 0, 30)]
    return windows


def predict_rows(estimator, test, definition, metadata):
    rows = []
    for (_, observation), score in zip(test.iterrows(), estimator.decision_function(test[definition['features']])):
        rows.append(dict(metadata, run=observation.run, window=observation.window,
            label=int(observation.label), score=float(score), predicted=int(score > 0)))
    return rows


def summarise_predictions(predictions):
    rows = []
    for (length, name), group in predictions.groupby(['n', 'model']):
        faulty, healthy = group[group.label == 1], group[group.label == 0]
        tp, tn = int(faulty.predicted.sum()), int((healthy.predicted == 0).sum())
        rows.append({'n': length, 'model': name, 'TP': tp, 'FN': len(faulty)-tp,
            'TN': tn, 'FP': len(healthy)-tn, 'degraded_total': len(faulty),
            'healthy_total': len(healthy), 'sensitivity': tp/len(faulty),
            'specificity': tn/len(healthy),
            'balanced_accuracy': (tp/len(faulty)+tn/len(healthy))/2})
    return pd.DataFrame(rows)


def blocked_validation(features, definitions):
    predictions = []
    for length in [5, 10]:
        observations = features[features.n == length]
        for name, definition in definitions.items():
            for nominal_run in observations[observations.label == 0].run.unique():
                for _, withheld in observations[observations.label == 1].iterrows():
                    overlap = (observations.label == 1) & (observations.start < withheld.end) & (observations.end > withheld.start)
                    train = observations[(observations.run != nominal_run) & ~overlap]
                    test = observations[(observations.run == nominal_run) | ((observations.label == 1) & (observations.window == withheld.window))]
                    assert nominal_run not in train.run.values
                    assert not ((train.label == 1) & (train.start < withheld.end) & (train.end > withheld.start)).any()
                    estimator = fit_classifier(train, definition)
                    metadata = {'n': length, 'model': name, 'held_nominal': nominal_run, 'held_degraded': withheld.window}
                    predictions.extend(predict_rows(estimator, test, definition, metadata))
    return pd.DataFrame(predictions)


def chronological_validation(features, definitions):
    predictions = []
    for length in [5, 10]:
        observations = features[features.n == length]
        faulty = observations[observations.label == 1].sort_values('start')
        for name in FORWARD_MODELS:
            for nominal_run in observations[observations.label == 0].run.unique():
                for _, withheld in faulty.iloc[2:].iterrows():
                    train = observations[((observations.label == 0) & (observations.run != nominal_run)) |
                        ((observations.label == 1) & (observations.end <= withheld.start))]
                    assert (train[train.label == 1].end <= withheld.start).all()
                    test = observations[(observations.run == nominal_run) | ((observations.label == 1) & (observations.window == withheld.window))]
                    estimator = fit_classifier(train, definitions[name])
                    predictions.extend(predict_rows(estimator, test, definitions[name],
                        {'n': length, 'model': name, 'held_nominal': nominal_run, 'held_degraded': withheld.window}))
    return pd.DataFrame(predictions)


def run(config_path, output):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(-np.pi, np.pi, config['phase_samples'] + 1)
    rotations, observations, inventory, sampling = {}, [], [], []
    hashes, ids = set(), set()
    for spec in config['runs']:
        path = (config_path.parent / spec['file']).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes or spec['id'] in ids:
            raise ValueError('Duplicate measurement run in manifest.')
        hashes.add(digest); ids.add(spec['id'])
        cycles, counts, metadata = load_rotations(path, config)
        rotations[spec['id']] = cycles
        inventory.append(dict(metadata, run=spec['id'], label=spec['label'], file=spec['file'], sha256=digest))
        sampling.extend(dict(row, run=spec['id'], label=spec['label']) for row in counts)
        for window, start, end in observation_windows():
            profile = cycles[start:end].mean(axis=0)
            observations.append(dict(run=spec['id'], label=spec['label'], window=window,
                n=end-start, start=start, end=end, centre=(start+end+1)/2,
                **extract_parameters(profile, phase)))
    features = pd.DataFrame(observations)
    assert len(features) == len(config['runs']) * 12
    features.to_csv(output/'features.csv', index=False)
    pd.DataFrame(inventory).to_csv(output/'inventory.csv', index=False)
    pd.DataFrame(sampling).to_csv(output/'sampling_per_rotation.csv', index=False)
    np.savez_compressed(output/'rotation_profiles.npz', phase=phase, **rotations)
    five = features[features.n == 5]
    distributions = []
    for column in COLUMNS:
        row = {'parameter': column}
        for label, prefix in [(0, 'healthy'), (1, 'degraded')]:
            values = five[five.label == label][column].values
            row.update({prefix+'_'+stat: float(value) for stat, value in zip(
                ['min', 'q1', 'median', 'q3', 'max'], np.quantile(values, [0, .25, .5, .75, 1]))})
        distributions.append(row)
    pd.DataFrame(distributions).to_csv(output/'parameter_distributions.csv', index=False)
    for scope, group in [('pooled', five), ('healthy', five[five.label == 0]), ('degraded', five[five.label == 1])]:
        group[COLUMNS].corr().to_csv(output/f'correlation_{scope}.csv')
    definitions = model_definitions()
    (output/'model_definitions.json').write_text(json.dumps(definitions, indent=2))
    predictions = blocked_validation(features, definitions)
    predictions.to_csv(output/'cv_predictions.csv', index=False)
    summarise_predictions(predictions).to_csv(output/'cv_summary.csv', index=False)
    votes = predictions.groupby(['n','model','run','window','label']).predicted.agg(['mean','size']).reset_index()
    votes['predicted'] = (votes['mean'] > .5).astype(int)
    votes['tie'] = votes['mean'] == .5
    votes.to_csv(output/'cv_unique_block_votes.csv', index=False)
    forward = chronological_validation(features, definitions)
    forward.to_csv(output/'chronological_predictions.csv', index=False)
    summarise_predictions(forward).to_csv(output/'chronological_summary.csv', index=False)
    descriptive, coefficients = [], {}
    models_dir = output/'models'; models_dir.mkdir(exist_ok=True)
    for name, definition in definitions.items():
        estimator = fit_classifier(five, definition)
        descriptive.extend(predict_rows(estimator, features, definition, {'model': name}))
        if name == 'M0_M3_rms':
            joblib.dump({'pipeline': estimator, 'features': definition['features'],
                'phase_samples': config['phase_samples'], 'fit_scope': 'All five-rotation blocks; descriptive fit'}, models_dir/f'{name}.joblib')
        weights = estimator['lda'].coef_[0] / estimator['scale'].scale_
        coefficients[name] = {'features': definition['features'], 'weights': weights.tolist(),
            'intercept': float(estimator['lda'].intercept_[0] - weights @ estimator['scale'].mean_)}
    descriptive = pd.DataFrame(descriptive).merge(features[['run','window','n','start','end']], on=['run','window'])
    descriptive.to_csv(output/'descriptive_predictions.csv', index=False)
    (output/'descriptive_coefficients.json').write_text(json.dumps(coefficients, indent=2))
    trends = []
    for run_id, group in five.groupby('run'):
        group = group.sort_values('start')
        for column in COLUMNS:
            values = group[column].values
            trends.append({'run': run_id, 'label': int(group.label.iloc[0]), 'parameter': column,
                'first': values[0], 'last': values[-1], 'change': values[-1]-values[0],
                'percent_change': 100*(values[-1]-values[0])/abs(values[0]) if abs(values[0]) > 1e-8 else np.nan,
                'theil_sen_per_rotation': theilslopes(values, group.centre)[0],
                'spearman_block_order': spearmanr(group.centre, values).statistic})
    pd.DataFrame(trends).to_csv(output/'trends.csv', index=False)
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}
    (output/'analysis_metadata.json').write_text(json.dumps({'config': config, 'versions': versions,
        'RC': 'OLS slope of current residual against phase with intercept, using unique phase points; mA/rad. This coefficient depends on the phase origin.',
        'validation': 'Healthy run and degraded block withheld; 5 and 10 rotations separately',
        'limitations': 'One degraded run. Repeated fold predictions are dependent. Twenty- and thirty-rotation predictions are descriptive.'}, indent=2))
    print(summarise_predictions(predictions).query("model == 'M0_M3_rms'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    run(config_path, args.output.resolve() if args.output else config_path.parent/'output')
