#!/usr/bin/env python3
"""V7 run-level LOOCV for Motor DTs and the seven-input Hybrid Bend DT.

Import feature/model factories from the user's training scripts. The default
nested mode repeats model-family selection on an inner hold-out drawn ONLY
from each outer training fold. Models and instruments are evaluated separately.
No production model packages are written. Error = measured - predicted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
import yaml

VERSION = '4.0.0'
FAMILIES = ['ridge', 'polynomial_ridge', 'random_forest', 'gradient_boosting', 'hist_gradient_boosting']
METRICS = ['mae', 'rmse', 'bias', 'max_abs_error']


def path_of(value, base=None):
    p = Path(value).expanduser()
    return ((Path(base) / p) if base is not None and not p.is_absolute() else p).resolve()


def import_script(path, name, required):
    path = path_of(path)
    if not path.is_file():
        raise FileNotFoundError(f'Training script not found: {path}')
    # Mirror the training layout: training/ next to digital_twins/.
    for folder in [path.parent, path.parent / 'digital_twins', path.parent.parent / 'digital_twins']:
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    missing = [key for key in required if not hasattr(module, key)]
    if missing:
        raise RuntimeError(f'{path}: missing functions/constants: {missing}')
    return module


def find_training_script(explicit, kind):
    if explicit:
        return path_of(explicit)
    here = Path(__file__).resolve().parent
    folders = [here, here / 'training', here.parent / 'training']
    candidates = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in folder.glob('train*.py'):
            text = p.read_text(encoding='utf-8')
            marker = 'def make_current_models(' if kind == 'motor' else 'def select_feature_columns('
            second = 'def load_segments_from_files(' if kind == 'motor' else 'def build_trial_table_from_files('
            if marker in text and second in text:
                candidates.add(p.resolve())
    if len(candidates) != 1:
        option = '--motor-training-script' if kind == 'motor' else '--hybrid-training-script'
        raise RuntimeError(f'Found {len(candidates)} possible {kind} training scripts. Set {option} PATH.\n'
                           + '\n'.join(str(p) for p in sorted(candidates)))
    return next(iter(candidates))


def metrics(y, pred):
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if y.shape != pred.shape or not len(y):
        raise ValueError('Empty or mismatched measured/predicted arrays.')
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError('Nonfinite measured/predicted values; no samples were silently excluded.')
    error = y - pred
    return dict(n_samples=len(y), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))), bias=float(np.mean(error)),
                max_abs_error=float(np.max(np.abs(error))))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def source_of(sample):
    return str(path_of(sample[4]['source_file']))


def training_path(native, root, value):
    """Rebase an old configured training root to the explicitly requested V7."""
    path = path_of(value, root)
    if path.is_relative_to(root):
        return path
    old_value = native.get('training_data_root')
    if old_value:
        old_root = path_of(old_value, path_of(native.get('test_data_root', root.parent)))
        if path.is_relative_to(old_root):
            return root / path.relative_to(old_root)
    raise ValueError(f'Configured input is outside the requested data root: {path}. '
                     'Use an input_dir relative to the V7 root in training_config.yaml.')


def prepare_motor(module, files, mode, source, partition):
    segments = module.load_segments_from_files(
        file_paths=files, coupling_mode=mode, input_source=source,
        conversion_parameters=None, controller_starting_positions=None,
    )
    for segment in segments:
        segment['dataset_split'] = partition
        segment['split_group'] = module.get_split_group(segment, mode, source)
    samples = module.build_dataset(segments=segments, coupling_mode=mode, input_source=source)
    allowed = {str(path_of(p)) for p in files}
    for motor_samples in samples.values():
        for sample in motor_samples:
            if source_of(sample) not in allowed:
                raise RuntimeError('Feature builder returned a source outside its fold partition.')
    return samples


def inner_motor_split(samples_by_motor, seed):
    """Original trainer policy: one complete held-out run per protocol."""
    groups = defaultdict(set)
    for samples in samples_by_motor.values():
        for sample in samples:
            groups[sample[4]['protocol']].add(source_of(sample))
    rng = np.random.default_rng(seed)
    held = {}
    for protocol in sorted(groups):
        files = sorted(groups[protocol])
        if len(files) < 2:
            raise RuntimeError(
                f'Inner model selection: protocol {protocol!r} has only {len(files)} run(s) '
                'after excluding the outer test run. Nested evaluation needs at least three '
                'runs per protocol in the original dataset. For a model family fixed independently '
                'in advance, use --selection fixed --model-family NAME.'
            )
        held[protocol] = files[int(rng.integers(0, len(files)))]
    training, validation = {}, {}
    for motor, samples in samples_by_motor.items():
        training[motor] = [s for s in samples if source_of(s) != held[s[4]['protocol']]]
        validation[motor] = [s for s in samples if source_of(s) == held[s[4]['protocol']]]
    return training, validation, held


def choose_motor_family(module, samples_by_motor, candidates):
    train, validation, held = inner_motor_split(samples_by_motor, getattr(module, 'RUN_SPLIT_SEED', 42))
    tasks = []
    for motor in sorted(samples_by_motor):
        if not validation[motor]:
            continue
        if not train[motor]:
            raise RuntimeError(f'No inner training samples for M{motor}.')
        Xet, Xct, yet, yct = module.stack_samples(train[motor])
        Xev, Xcv, yev, ycv = module.stack_samples(validation[motor])
        dynamic, stationary = module.stack_evaluation_masks(validation[motor])
        for output, factory, Xt, yt, Xv, yv, kind, score in [
            ('encoder_position', module.make_encoder_models, Xet, yet, Xev, yev, 'encoder', module.encoder_model_selection_score),
            ('motor_current', module.make_current_models, Xct, yct, Xcv, ycv, 'current', module.current_model_selection_score),
        ]:
            results = {}
            models = factory()
            for family in candidates:
                model = clone(models[family]).fit(Xt, yt)
                pred = model.predict(Xv)
                overall = metrics(yv, pred)
                results[family] = {
                    'encoder_metrics': overall, 'overall_metrics': overall,
                    'by_protocol': module.compute_metrics_by_protocol(validation[motor], pred, response_kind=kind),
                    'zone_metrics': module.compute_current_zone_metrics(yv, pred, dynamic, stationary) if kind == 'current' else {},
                }
                if not np.isfinite(float(score(results[family]))):
                    raise RuntimeError(f'Nonfinite selection score: M{motor}, {output}, {family}.')
            tasks.append(dict(motor_index=motor, output=output, results=results, score_function=score))
    selected, report = module.select_common_model_family(tasks)
    report['inner_validation_file_by_protocol'] = held
    report['inner_training_files'] = sorted({source_of(s) for v in train.values() for s in v})
    return selected, report


def motor_fold(module, job, held_out, selection, fixed_family, candidates, fold_dir):
    files = job['files']
    train_files = [p for p in files if p != held_out]
    train = prepare_motor(module, train_files, job['coupling_mode'], job['input_source'], 'train')
    test = prepare_motor(module, [held_out], job['coupling_mode'], job['input_source'], 'test')
    if selection == 'nested':
        family, report = choose_motor_family(module, train, candidates)
    else:
        family, report = fixed_family, {'policy': 'family_fixed_before_evaluation', 'selected_common_model_family': fixed_family}
    report.update(outer_test_file=str(held_out), outer_training_files=[str(p) for p in train_files])
    dump(fold_dir / 'selection.json', report)
    rows = []
    details = {}
    for motor, test_samples in sorted(test.items()):
        if not test_samples:
            continue
        train_samples = train.get(motor, [])
        if not train_samples:
            raise RuntimeError(f'No outer training samples for M{motor}: {held_out}')
        train_sources = sorted({source_of(s) for s in train_samples})
        if str(held_out) in train_sources:
            raise RuntimeError('Outer test file found in training samples.')
        Xe, Xc, ye, yc = module.stack_samples(train_samples)
        Xet, Xct, yet, yct = module.stack_samples(test_samples)
        dyn, stationary = module.stack_evaluation_masks(test_samples)
        for output, factory, X, y, Xt, yt, kind, unit in [
            ('encoder_position', module.make_encoder_models, Xe, ye, Xet, yet, 'encoder', 'pulses'),
            ('motor_current', module.make_current_models, Xc, yc, Xct, yct, 'current', 'mA'),
        ]:
            model = clone(factory()[family]).fit(X, y)
            pred = model.predict(Xt)
            phases = {'overall': np.ones(len(yt), dtype=bool), 'dynamic': dyn, 'stationary': stationary}
            protocols = sorted({s[4]['protocol'] for s in test_samples})
            protocol = '|'.join(protocols)
            for phase, mask in phases.items():
                if not np.any(mask):
                    continue
                rows.append(dict(dataset=job['name'], configuration=job['coupling_mode'],
                    input_source=job['input_source'], variant=job['variant'], instrument_config=job['instrument'],
                    run_id=held_out.stem, held_out_file=str(held_out), protocol=protocol, motor=motor,
                    model_family=family, output=output, unit=unit, phase=phase,
                    n_training_runs=len(train_sources), training_files_json=json.dumps(train_sources),
                    **metrics(yt[mask], pred[mask])))
            details[f'motor_{motor}/{output}'] = module.compute_metrics_by_protocol(test_samples, pred, response_kind=kind)
    if not rows:
        raise RuntimeError(f'No outer test samples: {held_out}')
    dump(fold_dir / 'per_protocol_metrics.json', details)
    return rows


def resolve_jobs(module, native, root, gearbox_config, overrides=None):
    """Use the supplied trainer's configuration keys, without mixing jobs."""
    entries = native['motor_dt']['configurations']
    jobs, skipped = [], []
    def add(name, mode, source, folder, variant='', instrument=''):
        if not folder.is_dir():
            skipped.append(dict(dataset=name, reason=f'Missing directory: {folder}'))
            return
        files = sorted({path_of(p) for p in module.find_motor_jsonl_files(input_dir=folder, coupling_mode=mode)})
        if len(files) < 2:
            raise RuntimeError(f'{name}: need at least two complete runs; found {len(files)}.')
        jobs.append(dict(name=name, coupling_mode=mode, input_source=source, folder=folder,
                         files=files, variant=variant, instrument=instrument))
    for key, source in [('motor_only_commands', 'commands'), ('motor_only_targets', 'targets')]:
        if key in entries:
            add(key, 'motor_only', source, training_path(native, root, entries[key]['input_dir']))
    if 'gearbox_only' in entries or 'full_setup' in entries:
        variants, active = module.load_available_gearbox_variants(gearbox_config)
        if 'gearbox_only' in entries:
            base = training_path(native, root, entries['gearbox_only']['input_dir'])
            for variant in variants:
                add(f'gearbox_only/{variant}', 'gearbox_only', 'targets', base / variant, variant=variant)
        if 'full_setup' in entries:
            base = training_path(native, root, entries['full_setup']['input_dir'])
            for instrument in ('gripper', 'scissors'):
                folder = base / instrument
                if not folder.exists():
                    if instrument == 'gripper' and base.is_dir() and not (base / 'scissors').exists():
                        # Match legacy trainer fallback, but do not treat an already
                        # explicitly named scissors directory as gripper.
                        if base.name.lower() not in {'scissors', 'scissors_withoutcutting'}:
                            folder = base
                    elif instrument == 'scissors':
                        folder = next((p for p in [root/'scissors', root/'scissors_withoutcutting'] if p.is_dir()), folder)
                add(f'full_setup/{instrument}', 'full_setup', 'targets', folder, variant=active, instrument=instrument)
    if overrides:
        chosen = set(overrides)
        unknown = chosen - {j['name'] for j in jobs}
        if unknown:
            raise ValueError(f'Unknown/unavailable --jobs: {sorted(unknown)}. Available: {[j["name"] for j in jobs]}')
        jobs = [j for j in jobs if j['name'] in chosen]
    if not jobs:
        raise RuntimeError('No motor evaluation jobs found.')
    return jobs, skipped


def load_hybrid_data(module, native, root, csv_path, raw_dir):
    if csv_path:
        df = pd.read_csv(csv_path, dtype={'trial_id': 'string'})
        origin = str(path_of(csv_path))
    else:
        folder = path_of(raw_dir) if raw_dir else training_path(native, root, native['hybrid_bend_dt']['input_dir'])
        df = module.build_dataset(folder)
        origin = str(folder)
    required = {'trial_id', 'video_angle', 'dt_prediction'}
    if required - set(df.columns):
        raise ValueError(f'Hybrid dataset missing: {sorted(required-set(df.columns))}')
    # Authoritative schema from instrument_digital_twin, as imported by trainer.
    feature_columns = list(module.select_feature_columns(df))
    if len(feature_columns) != 7:
        raise ValueError(f'Expected the supplied seven-input Hybrid Bend schema; got {feature_columns}')
    if df['trial_id'].isna().any():
        raise ValueError('Missing trial_id values.')
    df = df.copy()
    df['trial_id'] = df['trial_id'].astype(str)
    trial_ids = sorted(df['trial_id'].unique())
    if len(trial_ids) < 2:
        raise ValueError('Hybrid evaluation needs at least two complete trials.')
    columns = ['video_angle', 'dt_prediction', *feature_columns]
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite Hybrid Bend inputs/targets. Rebuild the synchronized dataset with the training script.')
    df['prediction_error'] = df['video_angle'] - df['dt_prediction']
    if 'time' in df.columns:
        if not np.isfinite(pd.to_numeric(df['time'], errors='coerce')).all():
            raise ValueError('Hybrid dataset contains invalid time values.')
        df['time'] = pd.to_numeric(df['time'])
        df = df.sort_values(['trial_id', 'time'], kind='stable').reset_index(drop=True)
    return df, feature_columns, origin


def evaluate_hybrid(module, df, features, selection, fixed_family, candidates, output):
    rows, normalized_rows = [], []
    trials = sorted(df['trial_id'].unique())
    if selection == 'nested' and len(trials) < 3:
        raise ValueError('Nested Hybrid Bend evaluation requires at least three trials. '
                         'Use --selection fixed --hybrid-model-family NAME for a family fixed in advance.')
    for fold, trial in enumerate(trials, 1):
        print(f'Hybrid Bend fold {fold}/{len(trials)}: {trial}', flush=True)
        training_trials = [t for t in trials if t != trial]
        train = df[df['trial_id'].isin(training_trials)]
        test = df[df['trial_id'] == trial]
        report = dict(outer_test_trial=trial, outer_training_trials=training_trials, feature_columns=features)
        if selection == 'nested':
            # Exact original hybrid policy within the OUTER TRAINING trials:
            # latest trial by lexicographic ID is the inner validation trial.
            inner_trial = training_trials[-1]
            inner_train = train[train['trial_id'] != inner_trial]
            inner_val = train[train['trial_id'] == inner_trial]
            scores = {}
            factories = module.make_models()
            for name in candidates:
                model = clone(factories[name]).fit(inner_train[features], inner_train['prediction_error'])
                pred = inner_val['dt_prediction'].to_numpy() + model.predict(inner_val[features])
                scores[name] = metrics(inner_val['video_angle'], pred)['mae']
            # Original model iteration order breaks exact score ties.
            family = min(candidates, key=lambda name: scores[name])
            report.update(policy='inner_latest_trial_hold_out', inner_validation_trial=inner_trial,
                          inner_training_trials=[t for t in training_trials if t != inner_trial],
                          inner_hybrid_mae_deg=scores)
        else:
            family = fixed_family
            report['policy'] = 'family_fixed_before_evaluation'
        report['selected_model_family'] = family
        fold_dir = output / 'hybrid_folds' / f'fold_{fold:03d}'
        fold_dir.mkdir(parents=True)
        dump(fold_dir/'selection.json', report)
        model = clone(module.make_models()[family]).fit(train[features], train['prediction_error'])
        video = test['video_angle'].to_numpy(dtype=float)
        physics = test['dt_prediction'].to_numpy(dtype=float)
        hybrid = physics + model.predict(test[features])
        # Preserve the training script's exact angle convention. No additional
        # per-fold zeroing is applied to video, physics, or hybrid predictions.
        angle_range = float(np.ptp(video))
        predictions = test[['trial_id', *(['time'] if 'time' in test else [])]].copy()
        predictions['video_angle_deg'] = video
        predictions['physics_deg'] = physics
        predictions['hybrid_deg'] = hybrid
        predictions.to_csv(fold_dir/'held_out_predictions.csv', index=False)
        for name, prediction in [('physics', physics), ('hybrid', hybrid)]:
            errors_deg = metrics(video, prediction)
            rows.append(dict(dof=2, run_id=trial, model=name, model_family=family if name == 'hybrid' else 'physics',
                             unit='rad', n_training_runs=len(training_trials) if name == 'hybrid' else 0,
                             training_trials_json=json.dumps(training_trials if name == 'hybrid' else []),
                             **metrics(np.deg2rad(video), np.deg2rad(prediction))))
            normalized_rows.append(dict(run_id=trial, model=name, model_family=family if name == 'hybrid' else 'physics',
                video_angle_range_deg=angle_range, **{f'{k}_deg':errors_deg[k] for k in METRICS},
                nmae_percent=100*errors_deg['mae']/angle_range if angle_range>0 else None,
                nrmse_percent=100*errors_deg['rmse']/angle_range if angle_range>0 else None))
    return rows, normalized_rows


def pool_motor_runs(rows):
    groups = defaultdict(list)
    keys = ['dataset','configuration','input_source','variant','instrument_config','run_id','held_out_file','protocol','model_family','output','unit','phase']
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        weights = np.asarray([r['n_samples'] for r in values], dtype=float)
        result.append(dict(zip(keys, key), n_motors=len(values), n_samples=int(weights.sum()),
            mae=float(np.average([r['mae'] for r in values],weights=weights)),
            bias=float(np.average([r['bias'] for r in values],weights=weights)),
            rmse=float(np.sqrt(np.average([r['rmse']**2 for r in values],weights=weights))),
            max_abs_error=max(r['max_abs_error'] for r in values)))
    return result


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        summary = dict(zip(keys, key), n_runs=len(values))
        for metric in METRICS:
            array = np.asarray([r[metric] for r in values])
            summary[f'{metric}_mean'] = float(np.mean(array))
            summary[f'{metric}_sd'] = float(np.std(array,ddof=1)) if len(array)>1 else None
        output.append(summary)
    return output


def latex_tables(motor, hybrid):
    lines = ['% Requires booktabs. Generated from held-out predictions only.']
    def fmt(row, name, decimals):
        mean, sd = row[f'{name}_mean'], row[f'{name}_sd']
        return f'{mean:.{decimals}f} $\\pm$ ' + (f'{sd:.{decimals}f}' if sd is not None else '--')
    def escape(text):
        return str(text).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
    if motor:
        lines += [r'\begin{table*}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
            r'\caption{Motor Digital Twin run-level cross-validation. Separate datasets and phases are reported. Errors are pooled across motor samples within each held-out run; values are mean $\pm$ sample SD of run-level errors. Model-family selection, when enabled, occurs within the outer training data.}',
            r'\label{tab:motor_dt_loocv}',r'\begin{tabular}{lllccccc}',r'\toprule',
            r'Dataset & Output & Phase & $n$ & MAE & RMSE & Bias & Max. abs. error \\',r'\midrule']
        for row in motor:
            if row['phase'] not in {'overall','dynamic'}:
                continue
            label = 'Position [pulses]' if row['output']=='encoder_position' else 'Current [mA]'
            lines.append(f"{escape(row['dataset'])} & {label} & {row['phase']} & {row['n_runs']} & "
                         + ' & '.join(fmt(row,m,2) for m in METRICS) + r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    if hybrid:
        lines += [r'\begin{table*}[t]',r'\centering\small',
            r'\caption{DOF2 Instrument Digital Twin errors in radians, mean $\pm$ sample SD across held-out trials. The hybrid correction is refitted per fold; the fixed physics prediction provides the reference baseline.}',
            r'\label{tab:hybrid_dt_loocv}',r'\begin{tabular}{lccccc}',r'\toprule',
            r'Model & $n$ & MAE [rad] & RMSE [rad] & Bias [rad] & Max. abs. error [rad] \\',r'\midrule']
        for row in hybrid:
            lines.append(f"{row['model']} & {row['n_runs']} & " + ' & '.join(fmt(row,m,3) for m in METRICS) + r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/home/leanne/ros2_ws/test_data/automated_trials/trainings data V7')
    parser.add_argument('--training-config', help='Existing training_config.yaml; default from the imported trainer.')
    parser.add_argument('--gearbox-config', help='Existing gearbox_params.yaml; default from the motor trainer.')
    parser.add_argument('--motor-training-script','--training-script', dest='motor_training_script')
    parser.add_argument('--hybrid-training-script')
    parser.add_argument('--dof2-dataset', help='Optional existing synchronized CSV; must have all seven authoritative inputs.')
    parser.add_argument('--dof2-data-dir', help='Optional raw ROS/video directory instead of YAML input_dir.')
    parser.add_argument('--skip-motor', action='store_true')
    parser.add_argument('--skip-dof2', action='store_true')
    parser.add_argument('--selection', choices=['nested','fixed'], default='nested')
    parser.add_argument('--model-family', choices=FAMILIES, help='Motor family for fixed mode only; default random_forest.')
    parser.add_argument('--hybrid-model-family', choices=FAMILIES, help='Hybrid family for fixed mode only; default random_forest.')
    parser.add_argument('--models', nargs='+', choices=FAMILIES, help='Candidate families for nested selection; default all trainer families.')
    parser.add_argument('--jobs', nargs='+', help='Motor dataset names to evaluate; see --preflight plan.')
    parser.add_argument('--preflight', action='store_true', help='Check imports, job discovery and hybrid schema; no fits or output writes.')
    parser.add_argument('--output-dir', default=None, help='New or empty output directory.')
    args = parser.parse_args()
    if args.skip_motor and args.skip_dof2:
        parser.error('No evaluations enabled.')
    if args.selection=='fixed' and args.models:
        parser.error('--models applies only to nested selection.')
    if args.selection=='nested' and (args.model_family or args.hybrid_model_family):
        parser.error('Fixed model-family options require --selection fixed.')
    args.model_family = args.model_family or 'random_forest'
    args.hybrid_model_family = args.hybrid_model_family or 'random_forest'
    root = path_of(args.data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    motor_module = hybrid_module = None
    sources = {}
    if not args.skip_motor:
        sources['motor'] = find_training_script(args.motor_training_script,'motor')
        motor_module = import_script(sources['motor'],'loocv_motor_trainer',[
            'load_segments_from_files','build_dataset','stack_samples','make_encoder_models','make_current_models',
            'get_split_group','stack_evaluation_masks','compute_metrics_by_protocol','find_motor_jsonl_files',
            'load_available_gearbox_variants','compute_current_zone_metrics','encoder_model_selection_score',
            'current_model_selection_score','select_common_model_family'])
    if not args.skip_dof2:
        sources['hybrid'] = find_training_script(args.hybrid_training_script,'hybrid')
        hybrid_module = import_script(sources['hybrid'],'loocv_hybrid_trainer',[
            'select_feature_columns','build_dataset','make_models'])
    config_module = motor_module if motor_module is not None else hybrid_module
    native_path = path_of(args.training_config or config_module.DEFAULT_TRAINING_CONFIG_PATH)
    with native_path.open(encoding='utf-8') as handle:
        native = yaml.safe_load(handle)
    jobs, skipped = [], []
    if motor_module is not None:
        gearbox_path = path_of(args.gearbox_config or motor_module.DEFAULT_GEARBOX_CONFIG_PATH)
        jobs, skipped = resolve_jobs(motor_module,native,root,gearbox_path,args.jobs)
    motor_candidates = []
    if motor_module is not None:
        available = list(motor_module.make_encoder_models())
        motor_candidates = args.models or available
        common = set(available) & set(motor_module.make_current_models())
        requested = motor_candidates if args.selection=='nested' else [args.model_family]
        if set(requested)-common:
            raise ValueError(f'Motor families unavailable: {sorted(set(requested)-common)}')
    hybrid_df = features = hybrid_origin = None
    hybrid_candidates = []
    if hybrid_module is not None:
        hybrid_candidates = args.models or list(hybrid_module.make_models())
        requested = hybrid_candidates if args.selection=='nested' else [args.hybrid_model_family]
        if set(requested)-set(hybrid_module.make_models()):
            raise ValueError('Requested hybrid families unavailable.')
        hybrid_df,features,hybrid_origin = load_hybrid_data(hybrid_module,native,root,args.dof2_dataset,args.dof2_data_dir)
        if args.selection=='nested' and hybrid_df['trial_id'].nunique()<3:
            raise ValueError('Nested hybrid evaluation requires at least three trials.')
    print('\nV7 LOOCV PLAN',flush=True)
    print(f'Data root: {root}\nSelection: {args.selection}')
    for job in jobs:
        print(f"  {job['name']}: {len(job['files'])} outer folds; source={job['input_source']}")
    for skipped_job in skipped:
        print(f"  SKIPPED {skipped_job['dataset']}: {skipped_job['reason']}")
    if hybrid_df is not None:
        print(f"  Hybrid Bend: {hybrid_df['trial_id'].nunique()} outer trial folds; inputs={features}")
    if args.preflight:
        print('Preflight complete. No model fits; raw motor feature building is exercised during evaluation.')
        return
    output = path_of(args.output_dir) if args.output_dir else root.parent/f'dt_loocv_v7_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    plan = [dict(name=j['name'],coupling_mode=j['coupling_mode'],input_source=j['input_source'],
                 variant=j['variant'],instrument=j['instrument'],files=[str(p) for p in j['files']],
                 folds=[dict(test_file=str(test),training_files=[str(p) for p in j['files'] if p!=test]) for test in j['files']]) for j in jobs]
    dump(output/'validation_plan.json',dict(motor_datasets=plan,skipped_jobs=skipped,hybrid_source=hybrid_origin,hybrid_features=features))
    rows = []
    for job_index,job in enumerate(jobs,1):
        for fold,test in enumerate(job['files'],1):
            print(f"{job['name']} fold {fold}/{len(job['files'])}: {test.name}",flush=True)
            fold_dir=output/'motor_folds'/f'job_{job_index:02d}'/f'fold_{fold:03d}'
            fold_dir.mkdir(parents=True)
            fold_rows=motor_fold(motor_module,job,test,args.selection,args.model_family,motor_candidates,fold_dir)
            write_csv(fold_dir/'metrics.csv',fold_rows)
            rows.extend(fold_rows)
    run_rows=pool_motor_runs(rows)
    motor_summary=summarize(run_rows,['dataset','configuration','input_source','variant','instrument_config','output','unit','phase'])
    per_motor_summary=summarize(rows,['dataset','motor','output','unit','phase'])
    # Save motor results before starting the separate hybrid evaluation.
    write_csv(output/'motor_per_motor_metrics.csv',rows)
    write_csv(output/'motor_per_run_metrics.csv',run_rows)
    write_csv(output/'motor_summary.csv',motor_summary)
    write_csv(output/'motor_per_motor_summary.csv',per_motor_summary)
    hybrid_rows=[]
    if hybrid_df is not None:
        hybrid_rows,normalized=evaluate_hybrid(hybrid_module,hybrid_df,features,args.selection,args.hybrid_model_family,hybrid_candidates,output)
        write_csv(output/'instrument_per_run_metrics.csv',hybrid_rows)
        write_csv(output/'instrument_target_normalized_metrics.csv',normalized)
    hybrid_summary=summarize(hybrid_rows,['dof','model','unit'])
    write_csv(output/'instrument_summary.csv',hybrid_summary)
    (output/'dt_performance_tables.tex').write_text(latex_tables(motor_summary,hybrid_summary),encoding='utf-8')
    dump(output/'evaluation_metadata.json',dict(version=VERSION,error_definition='measurement - prediction',
        cv_unit='one complete motor source file or one complete hybrid trial_id',selection=args.selection,
        motor_inner_selection='one run per protocol; dynamic encoder RMSE/current MAE; original common-family vote',
        hybrid_inner_selection='latest outer-training trial ID, ranked by hybrid video-angle MAE',
        hybrid_angle_reference='exact training convention; no extra zeroing',
        training_data_root=str(root),training_config=str(native_path),arguments=vars(args),
        training_script_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()},
        training_scripts={k:str(p) for k,p in sources.items()},
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_dir=str(output)))
    print(f'\nLOOCV complete. Results: {output}',flush=True)


if __name__=='__main__':
    main()
