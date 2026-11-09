#!/usr/bin/env python3
"""Verify extracted profiles, fold isolation and counts for complete run-wise LOOCV."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyse import (COLUMNS, PARAMETERS, residual_phase_slope, extract_parameters,
                     observation_windows, validation_folds, validate_feature_table,
                     summarise_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs')
    parser.add_argument('--figures', type=Path,
                        help='Verify all 15 figures in this directory, in PNG and SVG format.')
    args = parser.parse_args()
    output = args.output
    phase = np.linspace(-np.pi, np.pi, 151)[:-1]
    np.testing.assert_allclose(residual_phase_slope(2.3*phase+5, phase), 2.3, atol=1e-12)
    np.testing.assert_allclose(residual_phase_slope(np.full_like(phase, 5), phase), 0, atol=1e-12)
    features = pd.read_csv(output/'features.csv')
    validate_feature_table(features)
    assert len(COLUMNS) == 24 and len(PARAMETERS) == 11
    with np.load(output/'rotation_profiles.npz') as profiles:
        grid = profiles['phase']
        for run, group in features.groupby('run'):
            assert profiles[run].shape == (30, len(grid), 2)
            for window, start, end in observation_windows():
                profile = profiles[run][start:end].mean(axis=0)
                row = group[group.window == window].iloc[0]
                for column, value in extract_parameters(profile, grid).items():
                    np.testing.assert_allclose(row[column], value, rtol=1e-11, atol=1e-11)
                for index, motor in enumerate(('M0', 'M3')):
                    direct = np.linalg.lstsq(np.column_stack([grid[:-1], np.ones(len(grid)-1)]),
                                            profile[:-1, index], rcond=None)[0][0]
                    np.testing.assert_allclose(row[motor+'_rc'], direct, atol=1e-12)
    definitions = json.loads((output/'model_definitions.json').read_text())
    assert len(definitions) == 35
    expected = Counter()
    n_runs = features.run.nunique()
    for length in (5, 10):
        folds = list(validation_folds(features, length))
        assert len(folds) == n_runs
        observations = features[features.n == length]
        for train, test, meta in folds:
            held = meta['held_out_run']
            assert set(train.run).isdisjoint(test.run)
            assert set(test.run) == {held}
            assert train.run.nunique() == n_runs-1
            assert set(train.label) == {0, 1}
            assert len(test) == (6 if length == 5 else 3)
            pd.testing.assert_frame_equal(test, observations[observations.run == held])
            for row in test.itertuples():
                expected[(length, held, row.run, row.window, row.label)] += 1
    assert all(count == 1 for count in expected.values())
    keys = ['n', 'held_out_run', 'run', 'window', 'label']
    for filename, summary, count in [('cv_predictions.csv', 'cv_summary.csv', 35),
                                     ('cv_margin_predictions.csv', 'cv_margin_ranking.csv', 300)]:
        predictions = pd.read_csv(output/filename)
        assert predictions.model.nunique() == count
        assert not predictions.duplicated(['n', 'model', 'run', 'window']).any()
        assert (predictions.held_out_run == predictions.run).all()
        for _, group in predictions.groupby('model'):
            assert Counter(group[keys].itertuples(index=False, name=None)) == expected
        assert ((predictions.score > 0).astype(int) == predictions.predicted).all()
        actual = summarise_predictions(predictions)
        reported = pd.read_csv(output/summary)
        joined = actual.merge(reported, on=['n', 'model'], suffixes=('_actual', '_reported'),
                              validate='one_to_one')
        assert len(joined) == 2*count
        for key in ['TP', 'FN', 'TN', 'FP']:
            assert (joined[key+'_actual'] == joined[key+'_reported']).all()
    for scope in ('pooled', 'healthy', 'degraded'):
        matrix = pd.read_csv(output/f'correlation_{scope}.csv', index_col=0).to_numpy()
        assert matrix.shape == (24, 24)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    if args.figures:
        for stem in ('boxplots_M0', 'boxplots_M3_joint', 'correlation_pooled',
                     'correlation_healthy', 'correlation_degraded', 'lda_rms',
                     'lda_mixed_parameters', 'rc_final_block_fit', '04_parameter_evolution',
                     'leave_one_run_out_validation_5', 'leave_one_run_out_validation_10',
                     'lda_margin_ranking_5', 'lda_margin_ranking_10',
                     'lda_rms_rotation_windows', 'lda_rms_five_rotation_blocks'):
            for extension in ('png', 'svg'):
                assert (args.figures/(stem+'.'+extension)).stat().st_size > 1000
    print(f'Verified {features.run.nunique()} runs, profiles, 35 direct and 300 candidate classifiers.')
    print('Verified complete run exclusion and exactly one held-out prediction per block/model/length; classification errors are allowed.')


if __name__ == '__main__':
    main()
