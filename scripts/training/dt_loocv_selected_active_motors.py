#!/usr/bin/env python3
"""LOOCV of selected, saved Digital Twin model architectures on V7 data.

Run beside your training scripts:
  python3 dt_loocv_selected_active_motors.py --preflight
  python3 dt_loocv_selected_active_motors.py

Only estimators in the selected saved model packages are cloned and refitted.
There is no candidate-model comparison. All fitted preprocessing is reset by
sklearn.clone and learned again inside each leave-one-complete-run-out fold.
Read-only imports use your training scripts for deterministic per-run features.
Saved packages are read from training_config.yaml output_dir entries, using
that config's training_data_root (override with --model-root if needed).

A family originally selected using these same runs gives a conditional,
post-selection CV estimate; this does not independently validate model choice.

Primary motor summaries include only motors whose commanded target range exceeds
1 pulse within at least one evaluated motion pattern. Uncommanded motors remain
available in separate all-motor comparison exports. Activity is defined from
commands, never from measured motion, current or prediction accuracy.

Existing per-motor results can be reaggregated without model fitting:
  python3 dt_loocv_selected_active_motors.py --reaggregate RESULT_DIRECTORY

Completed folds and partial summaries are saved. Resume the NEW selected-model
run with --resume RESULT_DIRECTORY. Earlier all-motor or nested evaluations
cannot be resumed in this changed evaluation mode; use --reaggregate for saved metrics.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
import yaml

VERSION='1.1.0-selected-active-motors'
DEFAULT_ACTIVE_TARGET_RANGE_PULSES = 1.0
METRICS=['mae','rmse','bias','max_abs_error']

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
    folders = list(dict.fromkeys([here, here.parent, here / 'training', here.parent / 'training']))
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

def motor_activity(samples, threshold):
    """Largest within-pattern target range; no dependence on measurements/errors."""
    ranges=[]
    for sample in samples:
        target=np.asarray(sample[4]['target'],dtype=float).reshape(-1)
        if not len(target) or not np.isfinite(target).all():
            raise ValueError('Cannot classify motor activity from empty/nonfinite targets.')
        ranges.append(float(np.ptp(target)))
    value=max(ranges,default=0.0)
    return {'motor_target_range_pulses':value,
            'motor_activity_threshold_pulses':float(threshold),
            'active_motor_in_run':bool(value>threshold)}


def motor_activity_from_segments(module,segments,source,threshold):
    samples={motor:[] for motor in range(4)}
    for segment in segments:
        rows=segment['rows']
        if source=='targets':
            targets=np.asarray([r['motor_targets'] for r in rows],dtype=float).T
        else:
            commands=np.asarray([r['command_values'] for r in rows],dtype=float).T
            targets=module.relative_commands_to_target(commands)
        if targets.ndim!=2 or targets.shape[0]!=4:
            raise ValueError('Expected four target channels when checking motor activity.')
        for motor in range(4):
            samples[motor].append((None,None,None,None,{'target':targets[motor]}))
    return {motor:motor_activity(items,threshold) for motor,items in samples.items()}


def pool_motor_runs(rows, active_only=True):
    groups=defaultdict(list)
    keys=['dataset','configuration','input_source','variant','instrument_config','run_id','held_out_file','protocol','output','unit','phase']
    for row in rows:
        if active_only:
            if 'active_motor_in_run' not in row:
                raise ValueError('Activity flags missing; use --reaggregate on the original results.')
            if not row['active_motor_in_run']:
                continue
        groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,values in sorted(groups.items()):
        weights=np.asarray([r['n_samples'] for r in values],dtype=float)
        result.append(dict(zip(keys,key),model_family='|'.join(sorted({r['model_family'] for r in values})),
            n_motors=len(values),motor_indices='|'.join(str(v['motor']) for v in sorted(values,key=lambda r:r['motor'])),
            motor_selection='commanded_motors' if active_only else 'all_motors',n_samples=int(weights.sum()),
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
            r'\caption{Motor Digital Twin run-level cross-validation. Separate datasets and phases are reported. Errors are pooled only across motors whose commanded targets vary above the configured activity threshold within a motion pattern of the held-out run; values are mean $\pm$ sample SD of run-level errors. The selected saved model architectures and hyperparameters are held fixed; each estimator is refitted within each fold.}',
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
            r'\caption{DOF2 Instrument Digital Twin errors in radians, mean $\pm$ sample SD across held-out trials. The selected saved hybrid architecture is refitted per fold; the physics prediction provides the reference baseline.}',
            r'\label{tab:hybrid_dt_loocv}',r'\begin{tabular}{lccccc}',r'\toprule',
            r'Model & $n$ & MAE [rad] & RMSE [rad] & Bias [rad] & Max. abs. error [rad] \\',r'\midrule']
        for row in hybrid:
            lines.append(f"{row['model']} & {row['n_runs']} & " + ' & '.join(fmt(row,m,3) for m in METRICS) + r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    return '\n'.join(lines)

def atomic_text(path, content):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(content,encoding='utf-8')
    os.replace(temporary,path)


def dump(path,value):
    atomic_text(path,json.dumps(value,indent=2,allow_nan=False))


def write_csv(path,rows):
    if not rows:return
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def configured_model_root(native,override):
    original=path_of(native.get('training_data_root','.'),path_of(native.get('test_data_root',Path.cwd())))
    return original,path_of(override) if override else original


def model_output_path(value,original_root,model_root,override):
    p=Path(value).expanduser()
    if not p.is_absolute():return path_of(p,model_root)
    if override and p.is_relative_to(original_root):return path_of(model_root/p.relative_to(original_root))
    return p.resolve()


def expected_motor_names(module,motor,source):
    if source=='commands':
        return ['commanded_target_pulses','abs_commanded_target_pulses','target_delta_pulses',
                'abs_target_delta_pulses','target_direction','time_since_target_change_s','is_moving']
    if not hasattr(module,'coupled_motor_target_feature_names'):
        raise ValueError('Motor trainer lacks coupled_motor_target_feature_names for schema validation.')
    return list(module.coupled_motor_target_feature_names(motor))+['motor_phase_903_sin','motor_phase_903_cos']


def estimator_info(estimator,family):
    return {'family':str(family),'class':f'{type(estimator).__module__}.{type(estimator).__qualname__}',
            'parameters':repr(estimator)}


def load_motor_packages(path,job,module):
    path=path_of(path)
    if path.is_dir():
        combined=path/'motor_dt_all_motors.pkl'
        if not combined.exists():combined=path/'models'/'motor_dt_all_motors.pkl'
        if combined.is_file():path=combined
    sources=[]
    if path.is_file():
        package=joblib.load(path)
        if not isinstance(package,dict):raise ValueError(f'Expected a motor model package dict: {path}')
        if 'motor_models' in package:raw=package['motor_models']
        elif 'motor_index' in package:raw={int(package['motor_index']):package}
        else:raise ValueError(f'No motor_models or motor_index in {path}')
        if not isinstance(raw,dict):raise ValueError(f'Invalid motor_models in {path}')
        sources=[path]
    elif path.is_dir():
        model_dir=path/'models' if (path/'models').is_dir() else path
        files=sorted(model_dir.glob('motor_dt_m[0-3].pkl'))
        if not files:raise FileNotFoundError(f'No selected motor model packages under {path}')
        raw={}
        for file in files:
            item=joblib.load(file)
            index=int(item['motor_index'])
            if index in raw:raise ValueError(f'Duplicate M{index} package under {path}')
            raw[index]=item;sources.append(file)
    else:raise FileNotFoundError(f'Saved motor models not found: {path}')
    if not raw:raise ValueError(f'No saved motors in {path}')
    result={}
    for key,item in raw.items():
        motor=int(key)
        if motor not in range(4) or int(item.get('motor_index',motor))!=motor:
            raise ValueError(f'Invalid motor index in {path}: {key}')
        for field,expected in [('coupling_mode',job['coupling_mode']),('input_source',job['input_source'])]:
            if item.get(field)!=expected:
                raise ValueError(f'{job["name"]} model {field}={item.get(field)!r}, expected {expected!r}: {path}')
        for field,expected in [('gearbox_variant',job['variant']),('instrument_config',job['instrument'])]:
            if expected and item.get(field) not in (None,'',expected):
                raise ValueError(f'{job["name"]}: saved {field}={item[field]!r}, expected {expected!r}')
        expected_names=expected_motor_names(module,motor,job['input_source'])
        targets={}
        for kind in ('encoder','current'):
            estimator=item.get(f'{kind}_model');family=item.get(f'selected_{kind}_model')
            if estimator is None or not family:
                raise ValueError(f'Missing selected {kind} model/name for M{motor}: {path}')
            names=item.get(f'{kind}_feature_names')
            if list(names or [])!=expected_names:
                raise ValueError(f'M{motor} {kind}: saved feature order differs from the current trainer.\nSaved: {names}\nExpected: {expected_names}')
            if hasattr(estimator,'n_features_in_') and int(estimator.n_features_in_)!=len(expected_names):
                raise ValueError(f'M{motor} {kind}: saved estimator feature count mismatch.')
            # Validate cloning now; discard fitted state before any CV fit.
            template=clone(estimator)
            targets[kind]={'estimator':template,'family':str(family),'features':expected_names,
                           'info':estimator_info(template,family)}
        result[motor]=targets
    return result,[{'path':str(p),'sha256':file_sha(p)} for p in sources]


def load_hybrid_package(path,features):
    path=path_of(path)
    if path.is_dir():
        metrics_path=path/'hybrid_bend_metrics.json'
        if not metrics_path.is_file():
            raise FileNotFoundError(f'No {metrics_path}; specify --hybrid-model with the exact selected .joblib file.')
        selection=json.loads(metrics_path.read_text())
        selected=selection.get('selected_model')
        if not selected:raise ValueError(f'No selected_model in {metrics_path}')
        rel=selection.get('selected_model_file') or f'{selected}/{selected}_bend_hybrid_model.joblib'
        model_file=path_of(rel,path)
        source_metrics={'path':str(metrics_path),'sha256':file_sha(metrics_path)}
    else:
        model_file=path;selected=None;source_metrics=None
    if not model_file.is_file():raise FileNotFoundError(f'Selected Hybrid Bend package not found: {model_file}')
    package=joblib.load(model_file)
    if not isinstance(package,dict) or 'model' not in package or not package.get('model_name'):
        raise ValueError(f'Expected a Hybrid Bend model package with model/model_name: {model_file}')
    family=str(package['model_name'])
    if selected and family!=selected:raise ValueError('Selected hybrid metadata disagrees with saved model_name.')
    if list(package.get('feature_columns',[]))!=list(features):
        raise ValueError(f'Hybrid saved inputs differ from current seven-input schema: {package.get("feature_columns")} != {features}')
    if package.get('residual_unit')!='degrees':raise ValueError('Saved hybrid residual_unit must be degrees.')
    estimator=package['model']
    if hasattr(estimator,'n_features_in_') and int(estimator.n_features_in_)!=len(features):
        raise ValueError('Saved hybrid estimator feature count mismatch.')
    template=clone(estimator)
    sources=[{'path':str(model_file),'sha256':file_sha(model_file)}]
    if source_metrics:sources.append(source_metrics)
    return {'estimator':template,'family':family,'features':list(features),'info':estimator_info(template,family)},sources


def split_cached(samples,held):
    train,test={},{}
    held=str(held)
    for motor,items in samples.items():
        train[motor]=[];test[motor]=[]
        for item in items:
            is_test=source_of(item)==held
            marked=(*item[:4],{**item[4],'dataset_split':'test' if is_test else 'train'})
            (test if is_test else train)[motor].append(marked)
    return train,test


def fit_predict(template,X,y,X_test,label):
    start=time.monotonic()
    print(f'    FIT {label} ({len(y)} training samples)',flush=True)
    estimator=clone(template)
    estimator.fit(X,y)
    predicted=estimator.predict(X_test)
    print(f'    DONE {label}: {time.monotonic()-start:.1f} s',flush=True)
    return predicted


def evaluate_selected_motor(module,job,samples,models,held,fold_dir,activity_threshold=DEFAULT_ACTIVE_TARGET_RANGE_PULSES):
    train,test=split_cached(samples,held)
    rows=[];details={}
    for motor,targets in sorted(models.items()):
        items=test.get(motor,[])
        if not items:continue
        activity=motor_activity(items,activity_threshold)
        training=train.get(motor,[])
        if not training:raise ValueError(f'No training samples for selected M{motor}, fold {held}')
        sources=sorted({source_of(s) for s in training})
        if str(held) in sources:raise RuntimeError('Held-out source appears in training.')
        Xe,Xc,ye,yc=module.stack_samples(training)
        Xet,Xct,yet,yct=module.stack_samples(items)
        dynamic,stationary=module.stack_evaluation_masks(items)
        for kind,X,y,Xt,yt,output,unit in [
            ('encoder',Xe,ye,Xet,yet,'encoder_position','pulses'),
            ('current',Xc,yc,Xct,yct,'motor_current','mA')]:
            spec=targets[kind]
            if X.shape[1]!=len(spec['features']) or Xt.shape[1]!=len(spec['features']):
                raise ValueError(f'{job["name"]} M{motor}/{kind}: feature shape changed.')
            predicted=fit_predict(spec['estimator'],X,y,Xt,f'M{motor} {kind}: {spec["family"]}')
            masks={'overall':np.ones(len(yt),dtype=bool),'dynamic':np.asarray(dynamic,dtype=bool),'stationary':np.asarray(stationary,dtype=bool)}
            for phase,mask in masks.items():
                if not mask.any():continue
                rows.append(dict(dataset=job['name'],configuration=job['coupling_mode'],input_source=job['input_source'],
                    variant=job['variant'],instrument_config=job['instrument'],run_id=held.stem,held_out_file=str(held),
                    protocol='|'.join(sorted({s[4]['protocol'] for s in items})),motor=motor,model_family=spec['family'],
                    output=output,unit=unit,phase=phase,n_training_runs=len(sources),training_files_json=json.dumps(sources),
                    **activity,**metrics(yt[mask],predicted[mask])))
            details[f'motor_{motor}/{output}']=module.compute_metrics_by_protocol(items,predicted,response_kind=kind)
    if not rows:raise RuntimeError(f'No test samples for any selected saved motor: {held}')
    dump(fold_dir/'per_protocol_metrics.json',details)
    return rows


def evaluate_selected_hybrid(df,features,spec,trial,fold_dir):
    train=df[df['trial_id']!=trial];test=df[df['trial_id']==trial]
    training_trials=sorted(train['trial_id'].unique())
    correction=fit_predict(spec['estimator'],train[features],train['prediction_error'],test[features],f'Hybrid Bend: {spec["family"]}')
    video=test['video_angle'].to_numpy(dtype=float);physics=test['dt_prediction'].to_numpy(dtype=float)
    prediction=physics+correction
    records=test[['trial_id',*(['time'] if 'time' in test else [])]].copy()
    records['video_angle_deg']=video;records['physics_deg']=physics;records['hybrid_deg']=prediction
    temporary=fold_dir/'held_out_predictions.csv.tmp';records.to_csv(temporary,index=False)
    os.replace(temporary,fold_dir/'held_out_predictions.csv')
    angle_range=float(np.ptp(video));rows=[];normalized=[]
    for name,pred in [('physics',physics),('hybrid',prediction)]:
        family=spec['family'] if name=='hybrid' else 'physics'
        degree_metrics=metrics(video,pred)
        rows.append(dict(dof=2,run_id=trial,model=name,model_family=family,unit='rad',
            n_training_runs=len(training_trials) if name=='hybrid' else 0,
            training_trials_json=json.dumps(training_trials if name=='hybrid' else []),
            **metrics(np.deg2rad(video),np.deg2rad(pred))))
        normalized.append(dict(run_id=trial,model=name,model_family=family,video_angle_range_deg=angle_range,
            **{f'{m}_deg':degree_metrics[m] for m in METRICS},
            nmae_percent=100*degree_metrics['mae']/angle_range if angle_range>0 else None,
            nrmse_percent=100*degree_metrics['rmse']/angle_range if angle_range>0 else None))
    return rows,normalized


def checkpoint(path,token,key,rows=None,normalized=None):
    if rows is not None:
        dump(path,{'status':'complete','fingerprint':token,'fold':key,'rows':rows,'normalized':normalized or []})
        return None
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value.get('fingerprint')!=token or value.get('fold')!=key or value.get('status')!='complete':
        raise ValueError(f'Checkpoint does not match this evaluation: {path}')
    if not value.get('rows'):raise ValueError(f'Empty completed checkpoint: {path}')
    for row in value['rows']:
        if int(row['n_samples'])<=0 or not all(np.isfinite(float(row[m])) for m in METRICS):
            raise ValueError(f'Invalid checkpoint metrics: {path}')
    return value


def export_partial(output,motor_rows,hybrid_rows,normalized):
    run_rows=pool_motor_runs(motor_rows)
    all_run_rows=pool_motor_runs(motor_rows,active_only=False)
    keys=['dataset','configuration','input_source','variant','instrument_config','output','unit','phase']
    ms=summarize(run_rows,keys)
    hs=summarize(hybrid_rows,['dof','model','unit'])
    active_rows=[row for row in motor_rows if row['active_motor_in_run']]
    write_csv(output/'motor_per_motor_metrics.csv',motor_rows)
    write_csv(output/'motor_per_run_metrics.csv',run_rows)
    write_csv(output/'motor_summary.csv',ms)
    write_csv(output/'motor_per_motor_summary.csv',summarize(active_rows,['dataset','motor','output','unit','phase']))
    write_csv(output/'motor_summary_by_protocol.csv',summarize(run_rows,keys+['protocol']))
    write_csv(output/'motor_per_run_metrics_all_motors.csv',all_run_rows)
    write_csv(output/'motor_summary_all_motors.csv',summarize(all_run_rows,keys))
    write_csv(output/'motor_per_motor_summary_all_motors.csv',summarize(motor_rows,['dataset','motor','output','unit','phase']))
    audit={}
    for row in motor_rows:
        key=(row['dataset'],row['held_out_file'],row['motor'])
        audit[key]={k:row[k] for k in ['dataset','held_out_file','run_id','protocol','motor',
            'motor_target_range_pulses','motor_activity_threshold_pulses','active_motor_in_run']}
    write_csv(output/'motor_activity.csv',[audit[k] for k in sorted(audit)])
    write_csv(output/'instrument_per_run_metrics.csv',hybrid_rows)
    write_csv(output/'instrument_summary.csv',hs)
    write_csv(output/'instrument_target_normalized_metrics.csv',normalized)
    atomic_text(output/'dt_performance_tables.tex',latex_tables(ms,hs))


def read_metric_csv(path):
    if not path.is_file():return []
    integer_columns={'motor','n_samples','n_training_runs','dof'}
    float_columns=set(METRICS)|{'angle_range_deg','video_angle_range_deg','nmae_percent','nrmse_percent'}|{f'{m}_deg' for m in METRICS}
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        for key in integer_columns & row.keys():
            if row[key]!='':row[key]=int(row[key])
        for key in float_columns & row.keys():
            row[key]=float(row[key]) if row[key]!='' else None
    return rows


def reaggregate_existing(args):
    original=path_of(args.reaggregate)
    if not original.is_dir():raise FileNotFoundError(original)
    motor_rows=read_metric_csv(original/'motor_per_motor_metrics.csv')
    hybrid_rows=read_metric_csv(original/'instrument_per_run_metrics.csv')
    normalized=read_metric_csv(original/'instrument_target_normalized_metrics.csv')
    if not motor_rows and not hybrid_rows:
        raise ValueError('No per-motor or instrument metrics found in this results directory.')
    training_script=None
    if motor_rows:
        if args.motor_training_script:
            training_script=path_of(args.motor_training_script)
        else:
            manifest_path=original/'run_manifest.json'
            if manifest_path.is_file():
                manifest=json.loads(manifest_path.read_text())
                saved_path=manifest.get('identity',{}).get('training_scripts',{}).get('motor',{}).get('path')
                if saved_path and path_of(saved_path).is_file():training_script=path_of(saved_path)
            if training_script is None:training_script=find_training_script(None,'motor')
        module=import_script(training_script,'active_motor_reaggregation_trainer',['load_segments_from_files','get_split_group'])
        grouped=defaultdict(list)
        for row in motor_rows:
            grouped[(row['held_out_file'],row['configuration'],row['input_source'])].append(row)
        for (file,mode,source),rows in sorted(grouped.items()):
            held=path_of(file)
            if not held.is_file():raise FileNotFoundError(f'Original held-out log required for command activity: {held}')
            print(f'Checking commanded motors: {held.name}',flush=True)
            segments=module.load_segments_from_files(file_paths=[held],coupling_mode=mode,input_source=source)
            if not segments:raise ValueError(f'No evaluated motion patterns in {held}')
            activity=motor_activity_from_segments(module,segments,source,args.active_target_range_pulses)
            protocols=sorted({module.get_split_group(s,mode,source) for s in segments})
            for row in rows:
                row.update(activity[int(row['motor'])])
                row['protocol']='|'.join(protocols)
    output=path_of(args.output_dir) if args.output_dir else original.with_name(original.name+f'_active_motors_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if output.exists() and any(output.iterdir()):raise FileExistsError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True,exist_ok=True)
    export_partial(output,motor_rows,hybrid_rows,normalized)
    dump(output/'aggregation_manifest.json',{'version':VERSION,'source_results':str(original),
        'source_metrics_sha256':{p.name:file_sha(p) for p in [original/'motor_per_motor_metrics.csv',original/'instrument_per_run_metrics.csv'] if p.is_file()},
        'activity_training_script':str(training_script) if training_script else None,
        'activity_threshold_pulses':args.active_target_range_pulses,
        'motor_selection':'target range above threshold within at least one evaluated pattern',
        'model_fits_performed':0,'predictions_recomputed':False,
        'scope':'Only runs already present in the source metrics; newly added data require fresh LOOCV.'})
    print(f'Reaggregation complete; no models fitted. Results: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root',default='/home/leanne/ros2_ws/test_data/automated_trials/trainings data V7')
    parser.add_argument('--training-config');parser.add_argument('--gearbox-config')
    parser.add_argument('--motor-training-script','--training-script',dest='motor_training_script')
    parser.add_argument('--hybrid-training-script')
    parser.add_argument('--model-root',help='Override root for configured saved-model output directories.')
    parser.add_argument('--motor-model',action='append',default=[],metavar='DATASET=PATH',help='Exact model file/directory override; repeat for different motor datasets.')
    parser.add_argument('--hybrid-model',help='Exact selected .joblib file, or hybrid output directory with selection metadata.')
    parser.add_argument('--dof2-dataset');parser.add_argument('--dof2-data-dir')
    parser.add_argument('--skip-motor',action='store_true');parser.add_argument('--skip-dof2',action='store_true')
    parser.add_argument('--jobs',nargs='+');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output-dir');parser.add_argument('--resume',metavar='RESULT_DIRECTORY')
    parser.add_argument('--active-target-range-pulses',type=float,default=DEFAULT_ACTIVE_TARGET_RANGE_PULSES,
        help='Include a motor in primary metrics only when its target varies by more than this many pulses within an evaluated pattern (default: 1).')
    parser.add_argument('--reaggregate',metavar='RESULT_DIRECTORY',
        help='Rebuild summaries from existing per-motor metrics and raw logs without model fitting; save to a new directory.')
    args=parser.parse_args()
    if not np.isfinite(args.active_target_range_pulses) or args.active_target_range_pulses<0:
        parser.error('--active-target-range-pulses must be finite and nonnegative.')
    if args.reaggregate:
        if args.resume or args.preflight or args.jobs or args.skip_motor or args.skip_dof2:
            parser.error('--reaggregate uses the saved result set; do not combine it with --resume, --preflight, --jobs or --skip flags.')
        reaggregate_existing(args)
        return
    if args.skip_motor and args.skip_dof2:parser.error('No evaluations selected.')
    if args.output_dir and args.resume:parser.error('Use --output-dir or --resume, not both.')
    root=path_of(args.data_root)
    if not root.is_dir():raise FileNotFoundError(root)
    modules={};sources={}
    for kind,skip,explicit,required in [
        ('motor',args.skip_motor,args.motor_training_script,['load_segments_from_files','build_dataset','get_split_group','stack_samples',
            'stack_evaluation_masks','compute_metrics_by_protocol','find_motor_jsonl_files','load_available_gearbox_variants']),
        ('hybrid',args.skip_dof2,args.hybrid_training_script,['build_dataset','select_feature_columns'])]:
        if not skip:
            sources[kind]=find_training_script(explicit,kind)
            modules[kind]=import_script(sources[kind],f'selected_loocv_{kind}_trainer',required)
    base_module=modules.get('motor') or modules['hybrid']
    config_path=path_of(args.training_config or base_module.DEFAULT_TRAINING_CONFIG_PATH)
    native=yaml.safe_load(config_path.read_text())
    original_root,model_root=configured_model_root(native,args.model_root)
    overrides={}
    for item in args.motor_model:
        if '=' not in item:parser.error('--motor-model requires DATASET=PATH')
        name,path=item.split('=',1)
        if name in overrides:parser.error(f'Duplicate --motor-model for {name}')
        overrides[name]=path_of(path)
    jobs=[];skipped=[];motor_models={};model_sources={};gearbox_path=None
    if 'motor' in modules:
        gearbox_path=path_of(args.gearbox_config or modules['motor'].DEFAULT_GEARBOX_CONFIG_PATH)
        jobs,skipped=resolve_jobs(modules['motor'],native,root,gearbox_path,args.jobs)
        unknown=set(overrides)-{job['name'] for job in jobs}
        if unknown:raise ValueError(f'--motor-model does not match an enabled dataset: {sorted(unknown)}')
        entries=native['motor_dt']['configurations']
        for job in jobs:
            if job['name'] in overrides:model_path=overrides[job['name']]
            else:
                key=job['name'] if job['coupling_mode']=='motor_only' else job['coupling_mode']
                base=model_output_path(entries[key]['output_dir'],original_root,model_root,args.model_root)
                if job['coupling_mode']=='gearbox_only':base=base/job['variant']
                elif job['coupling_mode']=='full_setup':base=base/job['instrument']
                model_path=base
            motor_models[job['name']],model_sources[job['name']]=load_motor_packages(model_path,job,modules['motor'])
    df=features=origin=hybrid_model=None
    if 'hybrid' in modules:
        df,features,origin=load_hybrid_data(modules['hybrid'],native,root,args.dof2_dataset,args.dof2_data_dir)
        hp=path_of(args.hybrid_model) if args.hybrid_model else model_output_path(native['hybrid_bend_dt']['output_dir'],original_root,model_root,args.model_root)
        hybrid_model,model_sources['hybrid_bend']=load_hybrid_package(hp,features)
    selected_plan={}
    for job in jobs:
        selected_plan[job['name']]={str(motor):{kind:spec['info'] for kind,spec in targets.items()} for motor,targets in motor_models[job['name']].items()}
    if hybrid_model:selected_plan['hybrid_bend']=hybrid_model['info']
    print('\nSELECTED-MODEL LOOCV PLAN',flush=True)
    for name,targets in selected_plan.items():
        print(name,flush=True)
        if name=='hybrid_bend':print(f'  {targets["family"]}: {targets["parameters"]}',flush=True)
        else:
            for motor,outputs in targets.items():
                print(f'  M{motor}: encoder={outputs["encoder"]["family"]}; current={outputs["current"]["family"]}',flush=True)
        for source in model_sources[name]:print(f'  Package: {source["path"]}',flush=True)
    for item in skipped:print(f'SKIPPED {item["dataset"]}: {item["reason"]}',flush=True)
    n_motor=sum(len(j['files']) for j in jobs);n_hybrid=int(df['trial_id'].nunique()) if df is not None else 0
    planned_fits=sum(len(job['files'])*len(motor_models[job['name']])*2 for job in jobs)+n_hybrid
    print(f'Outer folds: {n_motor} motor + {n_hybrid} hybrid; at most {planned_fits} selected-estimator fits; no candidate comparison.',flush=True)
    print(f'Primary motor metrics: commanded target range > {args.active_target_range_pulses:g} pulses within an evaluated pattern; all-motor comparison also saved.',flush=True)
    print('This is fixed-selected-model CV. Selection previously performed on these same runs is not independently validated.',flush=True)
    if args.preflight:
        print('Preflight complete: saved models, cloning and feature metadata checked; no fits.',flush=True)
        return
    # Exact inputs needed for safe checkpoint reuse; fitted state is never used.
    data_files=[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for j in jobs for p in j['files']]
    hybrid_hash=None
    if df is not None:
        cols=['trial_id','video_angle','dt_prediction',*features,*(['time'] if 'time' in df else [])]
        hybrid_hash=hashlib.sha256(pd.util.hash_pandas_object(df[cols],index=False).values.tobytes()).hexdigest()
    identity={'motor_selection':{'policy':'commanded_target_range','threshold_pulses':args.active_target_range_pulses},'version':VERSION,'evaluator_sha256':file_sha(__file__),'training_scripts':{k:{'path':str(p),'sha256':file_sha(p)} for k,p in sources.items()},
        'config':{'path':str(config_path),'sha256':file_sha(config_path)},'models':model_sources,'model_info':selected_plan,'motor_data':data_files,
        'job_files':{j['name']:[str(p) for p in j['files']] for j in jobs},'hybrid_features':features,'hybrid_data_sha256':hybrid_hash,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}}
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output=path_of(args.resume or args.output_dir) if (args.resume or args.output_dir) else root.parent/f'dt_loocv_selected_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.resume:
        manifest_path=output/'run_manifest.json'
        if not manifest_path.is_file():raise ValueError('No selected-model run manifest here; cannot resume the earlier nested run.')
        previous=json.loads(manifest_path.read_text())
        if previous.get('fingerprint')!=token:raise ValueError('Inputs/models/settings changed. Choose a new output directory instead of resuming.')
    else:
        if output.exists() and any(output.iterdir()):raise FileExistsError(f'Output directory must be empty: {output}')
        output.mkdir(parents=True,exist_ok=True)
        dump(output/'run_manifest.json',{'fingerprint':token,'identity':identity,'error_definition':'measurement - prediction',
            'evaluation':'leave-one-run-out with saved selected architecture/hyperparameters fixed','selection_independently_validated':False})
    print(f'Results: {output}',flush=True)
    rows=[];hybrid_rows=[];normalized=[];completed=0;total=n_motor+n_hybrid
    for job_index,job in enumerate(jobs,1):
        cache=None
        for fold,held in enumerate(job['files'],1):
            key=f'{job["name"]}:{held}';folder=output/'motor_folds'/f'job_{job_index:02d}'/f'fold_{fold:03d}'
            folder.mkdir(parents=True,exist_ok=True)
            saved=checkpoint(folder/'result.json',token,key)
            if saved:
                print(f'REUSE {job["name"]} fold {fold}/{len(job["files"])}',flush=True)
                fold_rows=saved['rows']
            else:
                if cache is None:
                    print(f'Building deterministic per-run features once: {job["name"]}',flush=True)
                    cache=prepare_motor(modules['motor'],job['files'],job['coupling_mode'],job['input_source'],'train')
                print(f'[{completed+1}/{total}] {job["name"]} fold {fold}/{len(job["files"])}: {held.name}',flush=True)
                fold_rows=evaluate_selected_motor(modules['motor'],job,cache,motor_models[job['name']],held,folder,args.active_target_range_pulses)
                checkpoint(folder/'result.json',token,key,fold_rows)
            write_csv(folder/'metrics.csv',fold_rows);rows.extend(fold_rows);completed+=1
            export_partial(output,rows,hybrid_rows,normalized)
        cache=None
    if df is not None:
        for fold,trial in enumerate(sorted(df['trial_id'].unique()),1):
            key=f'hybrid_bend:{trial}';folder=output/'hybrid_folds'/f'fold_{fold:03d}';folder.mkdir(parents=True,exist_ok=True)
            saved=checkpoint(folder/'result.json',token,key)
            if saved:
                print(f'REUSE hybrid trial {trial}',flush=True);fold_rows=saved['rows'];norm=saved['normalized']
            else:
                print(f'[{completed+1}/{total}] Hybrid Bend fold {fold}/{n_hybrid}: {trial}',flush=True)
                fold_rows,norm=evaluate_selected_hybrid(df,features,hybrid_model,trial,folder)
                checkpoint(folder/'result.json',token,key,fold_rows,norm)
            hybrid_rows.extend(fold_rows);normalized.extend(norm);completed+=1
            export_partial(output,rows,hybrid_rows,normalized)
    dump(output/'completed.json',{'fingerprint':token,'completed_folds':completed,'total_folds':total,'completed_at':datetime.now().isoformat()})
    print(f'\nSelected-model LOOCV complete: {completed}/{total} folds. Results: {output}',flush=True)


if __name__=='__main__':main()
