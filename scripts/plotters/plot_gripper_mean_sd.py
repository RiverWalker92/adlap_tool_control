#!/usr/bin/env python3
"""Bereken de gripper-CSV opnieuw uit de runs en plot gemiddelde +/-1 SD.

Vanaf de ruwe runs (standaard de oorspronkelijke 15 runs):
    python3 plot_gripper_mean_sd.py --runs-dir /pad/naar/dof4

Alleen een bestaande CSV opnieuw plotten blijft ook mogelijk:
    python3 plot_gripper_mean_sd.py --input /pad/naar/mean_curves.csv

WERKWIJZE BIJ --runs-dir
1. Lees uitsluitend *_ros_log.jsonl en *_webcam_angles.jsonl van de 15
   expliciet genoemde runs. Motor- en instrument-replays worden niet gebruikt.
2. Neem predicted_instrument_angles[3] uit de ROS-log (radiaal -> graden).
   Dit zijn de GELOGDE physics-based voorspellingen; het DT wordt niet opnieuw
   uitgevoerd of gefit. De video gebruikt measured_angle_between_jaws_filtered
   in graden, met de reeds gesynchroniseerde absolute tijd ros_time_s.
3. Bewegingsaanvang = eerste absolute verandering > 0.0001 rad in
   commanded_instrument_angles[3] t.o.v. het eerste geldige commando.
   Trek alleen deze tijd af: geen amplitude-offset, tijdschaling of extra filter.
4. Interpoleer de ROS-voorspelling eerst naar de geldige videoframetijden.
   Geen extrapolatie; over ROS-gaten >0.10 s wordt niet geinterpoleerd.
5. Interpoleer de gematchte video- en voorspellingscurves naar een 30-Hz-rooster
   met t=0 als anker. Over videogaten >0.15 s wordt niet geinterpoleerd.
   Gebruik alleen roosterpunten met video EN voorspelling van ALLE runs.
6. Bereken het gelijk gewogen gemiddelde en de sample-SD (ddof=1) over de
   runs. De SD is geen betrouwbaarheidsinterval. Sla de nieuwe CSV op en plot.

Er worden geen ruwe bestanden overschreven. Standaard wordt een nieuwe map
figures_rebuilt_<tijdstempel> gemaakt. Bronbestanden krijgen een SHA-256-
controle voor en na uitvoering. settings.json legt de werkwijze en bronnen vast.
Een bestaande figures-map of mean_curves.csv is niet nodig bij --runs-dir.

Optioneel: --run-ids NAAM1 NAAM2 ... selecteert andere runs (minimaal twee).
Een enkele echte run kan geen SD over runs leveren.
Benodigd: numpy, pandas, matplotlib. Uitvoer: CSV, JSON, PNG, PDF en SVG.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_IDS = [
    'auto_dof4_20260929_' + time for time in [
        '085054', '085515', '085946', '090414', '091001',
        '091526', '092232', '092702', '093053', '093534',
        '094321', '094845', '095439', '095854', '100616']]


def read_log(path):
    """Lees JSONL en bewaar de oorspronkelijke hash; verander het bestand niet."""
    payload = path.read_bytes()
    rows = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f'{path}, regel {line_number}: ongeldige JSON') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path}, regel {line_number}: verwacht een JSON-object')
            rows.append(row)
    return rows, {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                  'logged_rows': len(rows)}


def series(rows, time_key, value_key, index=None):
    """Gebruik eindige waarden; rapporteer ontbrekende waarden in de metadata."""
    samples = []
    for row in rows:
        t, value = row.get(time_key), row.get(value_key)
        if index is not None:
            value = value[index] if isinstance(value, list) and len(value) > index else None
        try:
            t, value = float(t), float(value)
        except (ValueError, TypeError):
            continue
        if np.isfinite(t) and np.isfinite(value):
            samples.append((t, value))
    if len(samples) < 2:
        raise ValueError(f'Te weinig geldige waarden voor {time_key}, {value_key}, index={index}.')
    t, values = np.asarray(samples, dtype=float).T
    if not np.all(np.diff(t) > 0):
        raise ValueError(f'{time_key}/{value_key}: dubbele of niet-oplopende tijden; geen stilzwijgende correctie.')
    return t, values


def interpolate(t, values, query, maximum_gap):
    """Lineaire interpolatie, zonder extrapolatie of overbruggen van grote gaten."""
    t, values, query = np.asarray(t), np.asarray(values), np.asarray(query)
    result = np.interp(query, t, values)
    upper = np.searchsorted(t, query, side='left')
    clipped = np.clip(upper, 0, len(t)-1)
    exact = np.abs(t[clipped]-query) <= 1e-9
    lower = np.clip(upper-1, 0, len(t)-1)
    gap_ok = (t[clipped]-t[lower]) <= maximum_gap
    valid = (query >= t[0]) & (query <= t[-1]) & (exact | gap_ok)
    result[~valid] = np.nan
    return result


def load_run(ros_path, video_path, onset_threshold=1e-4, ros_gap=.10):
    """Reconstrueer een run op de originele videoframetijden."""
    ros_rows, ros_source = read_log(ros_path)
    video_rows, video_source = read_log(video_path)
    rt, prediction_rad = series(ros_rows, 'ros_timestamp', 'predicted_instrument_angles', 3)
    ct, command_rad = series(ros_rows, 'ros_timestamp', 'commanded_instrument_angles', 3)
    vt, video = series(video_rows, 'ros_time_s', 'measured_angle_between_jaws_filtered')
    changed = np.flatnonzero(np.abs(command_rad-command_rad[0]) > onset_threshold)
    if not len(changed):
        raise ValueError(f'{ros_path}: geen DOF4-commandobeweging boven {onset_threshold} rad.')
    onset = float(ct[changed[0]])
    prediction = np.rad2deg(prediction_rad)
    # Deze volgorde is essentieel: eerst ROS -> video, daarna video -> rooster.
    matched_prediction = interpolate(rt, prediction, vt, ros_gap)
    valid = np.isfinite(matched_prediction)
    if valid.sum() < 2:
        raise ValueError(f'{ros_path}: geen voldoende overlappende video- en ROS-data.')
    t = vt[valid]-onset
    video, matched_prediction = video[valid], matched_prediction[valid]
    residual = video-matched_prediction
    name = ros_path.name.removesuffix('_ros_log.jsonl')
    metrics = {
        'run': name, 'n_samples': int(valid.sum()),
        'mae_deg': float(np.mean(np.abs(residual))),
        'rmse_deg': float(np.sqrt(np.mean(residual**2))),
        'bias_deg': float(np.mean(residual)),
        'max_abs_error_deg': float(np.max(np.abs(residual))),
        'onset_ros_time_s': onset, 'alignment_ros_time_s': onset,
        'alignment_source': 'first commanded DOF4 motion onset',
        'initial_prediction_deg': float(prediction[0]),
        'excluded_video_samples': int((~valid).sum()),
        'video_rows_without_valid_time_or_angle': len(video_rows)-len(vt),
        'ros_rows_without_valid_prediction': len(ros_rows)-len(rt),
        'ros_rows_without_valid_command': len(ros_rows)-len(ct)}
    return {'name': name, 't': t, 'video': video, 'prediction': matched_prediction,
            'command_t': ct-onset, 'command_deg': np.rad2deg(command_rad),
            'metrics': metrics, 'sources': [ros_source, video_source]}


def find_run_files(root, run_ids):
    """Selecteer de gevraagde runs; geen onbedoelde kopie of replay meenemen."""
    pairs = []
    for name in run_ids:
        ros_matches = sorted(root.rglob(name+'_ros_log.jsonl'))
        if len(ros_matches) != 1:
            raise ValueError(f'{name}: verwacht exact een ROS-log onder {root}, gevonden: {len(ros_matches)}.')
        ros = ros_matches[0]
        video = ros.with_name(name+'_webcam_angles.jsonl')
        if not video.is_file():
            raise FileNotFoundError(f'Bijbehorende videolog ontbreekt: {video}')
        pairs.append((ros, video))
    return pairs


def common_grid(runs, hz=30., video_gap=.15, ros_gap=.10, command_limit=3.):
    """Gelijke weging van runs op uitsluitend volledig ondersteunde tijdpunten."""
    if len(runs) < 2:
        raise ValueError('Minimaal twee verschillende runs nodig voor SD met ddof=1.')
    start = max(run['t'][0] for run in runs)
    end = min(run['t'][-1] for run in runs)
    first, last = int(np.ceil(start*hz)), int(np.floor(end*hz))
    if last <= first:
        raise ValueError('Geen gemeenschappelijk tijdsinterval tussen de runs.')
    grid = np.arange(first, last+1, dtype=float)/hz
    videos = np.array([interpolate(run['t'], run['video'], grid, video_gap) for run in runs])
    predictions = np.array([interpolate(run['t'], run['prediction'], grid, video_gap) for run in runs])
    valid = np.isfinite(videos).all(axis=0) & np.isfinite(predictions).all(axis=0)
    excluded = int((~valid).sum())
    grid, videos, predictions = grid[valid], videos[:, valid], predictions[:, valid]
    if len(grid) < 2:
        raise ValueError('Te weinig roosterpunten met geldige data van alle runs.')
    commands = np.array([interpolate(run['command_t'], run['command_deg'], grid, ros_gap) for run in runs])
    if not np.isfinite(commands).all():
        raise ValueError('Commandolog heeft onvoldoende ondersteuning op het gezamenlijke rooster.')
    maximum_difference = float(np.ptp(commands, axis=0).max())
    if maximum_difference > command_limit:
        raise ValueError(f'Commandoverlopen verschillen maximaal {maximum_difference:.6g} graden; '
                         f'limiet is {command_limit} graden. Controleer runs en tijdsuitlijning.')
    residuals = videos-predictions
    frame = pd.DataFrame({
        'time_s': grid, 'n_runs_used': len(runs),
        'mean_video_deg': videos.mean(axis=0), 'sd_video_deg': videos.std(axis=0, ddof=1),
        'mean_prediction_deg': predictions.mean(axis=0),
        'sd_prediction_deg': predictions.std(axis=0, ddof=1),
        'mean_residual_deg': residuals.mean(axis=0), 'sd_residual_deg': residuals.std(axis=0, ddof=1)})
    residual_frame = pd.DataFrame({'time_s': grid})
    for index, run in enumerate(runs, start=1):
        residual_frame[f'run_{index:02d}_residual_deg'] = residuals[index-1]
    checks = [{'run': run['name'],
               'max_command_difference_deg': float(np.max(np.abs(commands[i]-commands[0]))),
               'rms_command_difference_deg': float(np.sqrt(np.mean((commands[i]-commands[0])**2)))}
              for i, run in enumerate(runs)]
    return frame, residual_frame, pd.DataFrame(checks), {
        'n_common_grid_points': len(grid), 'excluded_grid_points_due_to_gaps': excluded,
        'plot_time_range_s': [float(grid[0]), float(grid[-1])],
        'command_pairwise_max_difference_deg': maximum_difference}


def verify_sources(sources):
    for source in sources:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError('Bronbestand is tijdens de uitvoering veranderd: '+source['path'])


def rebuild(args, output):
    root = args.runs_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    run_ids = args.run_ids or DEFAULT_RUN_IDS
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError('Selecteer minimaal twee verschillende runnamen.')
    pairs = find_run_files(root, run_ids)
    runs = []
    for index, (ros, video) in enumerate(pairs, start=1):
        print(f'Run {index}/{len(pairs)}: {ros.name}', flush=True)
        runs.append(load_run(ros, video, args.onset_threshold_rad, args.max_ros_gap))
    data, residuals, command_checks, summary = common_grid(
        runs, args.grid_hz, args.max_video_gap, args.max_ros_gap, args.command_limit_deg)
    sources = [source for run in runs for source in run['sources']]
    verify_sources(sources)
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output/'mean_curves.csv', index=False)
    residuals.to_csv(output/'residuals_per_run.csv', index=False)
    pd.DataFrame([run['metrics'] for run in runs]).to_csv(output/'per_run_metrics.csv', index=False)
    command_checks.to_csv(output/'command_alignment_check.csv', index=False)
    (output/'run_index.json').write_text(json.dumps(
        {f'run_{i:02d}': str(pair[0].relative_to(root)) for i, pair in enumerate(pairs, start=1)}, indent=2))
    settings = {
        'source_data': str(root), 'n_runs': len(runs), 'prediction_source': 'recorded',
        'ros_time_key': 'ros_timestamp', 'prediction_key': 'predicted_instrument_angles',
        'prediction_index': 3, 'prediction_input_units': 'radians', 'output_units': 'degrees',
        'video_time_key': 'ros_time_s', 'video_key': 'measured_angle_between_jaws_filtered',
        'reference': 'absolute', 'additional_temporal_filter': False, 'time_rescaling': False,
        'alignment': 'motion-onset', 'onset_threshold_rad': args.onset_threshold_rad,
        'onset_definition': 'First absolute change from initial commanded_instrument_angles[3] exceeding threshold.',
        'interpolation_order': 'ROS prediction to original video frames, then matched curves to common grid.',
        'grid_hz': args.grid_hz, 'max_ros_interpolation_gap_s': args.max_ros_gap,
        'max_video_interpolation_gap_s': args.max_video_gap,
        'command_difference_limit_deg': args.command_limit_deg,
        'mean_definition': 'equal weight per run; only grid points with data from all runs',
        'sd_definition': 'sample standard deviation across runs, ddof=1; not a confidence interval',
        'video_alpha': args.video_alpha, 'prediction_alpha': args.prediction_alpha,
        'source_files': [{'ros': run['sources'][0], 'video': run['sources'][1]} for run in runs]}
    (output/'settings.json').write_text(json.dumps(settings, indent=2))
    (output/'summary.json').write_text(json.dumps({'n_runs': len(runs), **summary}, indent=2))
    print(f'Nieuwe CSV: {output / "mean_curves.csv"} ({len(data)} roosterpunten)', flush=True)
    return data, sources


def opacity(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('alpha moet tussen 0 en 1 liggen')
    return number


def plot_curves(data, output, video_alpha=.40, prediction_alpha=.35):
    columns = ['time_s', 'n_runs_used', 'mean_video_deg', 'sd_video_deg',
               'mean_prediction_deg', 'sd_prediction_deg']
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError('Ontbrekende kolommen: ' + ', '.join(missing))
    if len(data) < 2 or not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError('De CSV moet minimaal twee rijen met eindige waarden bevatten.')
    if not np.all(np.diff(data.time_s.to_numpy()) > 0):
        raise ValueError('time_s moet strikt oplopen; het script sorteert of verwijdert geen rijen.')
    if (data[['sd_video_deg', 'sd_prediction_deg']] < 0).any().any():
        raise ValueError('Standaarddeviaties mogen niet negatief zijn.')
    counts = data.n_runs_used.unique()
    if len(counts) != 1 or counts[0] < 1 or counts[0] != int(counts[0]):
        raise ValueError('Dit overzicht verwacht hetzelfde gehele aantal runs op ieder tijdstip.')
    n_runs = int(counts[0])

    # De vijf arrays komen rechtstreeks uit de CSV: geen verdere bewerking.
    t = data.time_s.to_numpy()
    video = data.mean_video_deg.to_numpy()
    video_sd = data.sd_video_deg.to_numpy()
    prediction = data.mean_prediction_deg.to_numpy()
    prediction_sd = data.sd_prediction_deg.to_numpy()

    output.mkdir(parents=True, exist_ok=True)
    style = {'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
             'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 9,
             'ytick.labelsize': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        fig.subplots_adjust(left=.105, right=.99, bottom=.18, top=.72)
        fig.text(.5, .99, f'DOF4 gripper opening across {n_runs} runs',
                 ha='center', va='top', fontsize=12)
        fig.text(.5, .93, 'Mean trajectory with ±1 SD',
                 ha='center', va='top', fontsize=10)

        # SD-band = gemiddelde min SD tot gemiddelde plus SD.
        # alpha verandert uitsluitend de dekking van de kleur.
        video_band = ax.fill_between(t, video-video_sd, video+video_sd,
                                    color='#0072B2', alpha=video_alpha,
                                    linewidth=0, label='Video ±1 SD', zorder=1)
        prediction_band = ax.fill_between(t, prediction-prediction_sd, prediction+prediction_sd,
                                         color='#D55E00', alpha=prediction_alpha,
                                         linewidth=0, label='Prediction ±1 SD', zorder=1)
        video_line, = ax.plot(t, video, color='#0072B2', linewidth=1.6,
                              label=f'Video measurement mean (n={n_runs})', zorder=3)
        prediction_line, = ax.plot(t, prediction, color='#D55E00', linestyle='--',
                                   linewidth=1.6, label='Physics-based prediction mean', zorder=3)
        fig.legend(handles=[video_line, video_band, prediction_line, prediction_band],
                   loc='upper center', bbox_to_anchor=(.52, .865), ncol=2,
                   frameon=False, fontsize=9, columnspacing=2, handlelength=2.3)
        ax.set(xlabel='Time from motion onset [s]', ylabel='Jaw opening [deg]',
               xlim=(float(t[0]), float(t[-1])))
        ax.grid(axis='y', color='#D5D7DB', alpha=.65, linewidth=.65)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        for extension in ['png', 'pdf', 'svg']:
            destination = output/f'01_mean_video_prediction.{extension}'
            fig.savefig(destination, dpi=600, bbox_inches='tight', facecolor='white')
            print(f'Opgeslagen: {destination}')
        plt.close(fig)

    print(f'Gebruikt: alle {len(data)} CSV-rijen; {n_runs} runs per tijdstip.')
    print(f'Alpha: video={video_alpha}, prediction={prediction_alpha}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--runs-dir', type=Path, help='Zoek de originele runs onder deze map, ook in submappen.')
    inputs.add_argument('--input', type=Path, help='Alleen een bestaande mean_curves.csv opnieuw plotten.')
    parser.add_argument('--output-dir', type=Path, help='Bij --runs-dir moet dit een nieuwe, nog niet bestaande map zijn.')
    parser.add_argument('--run-ids', nargs='+', help='Anders de 15 oorspronkelijke runnamen; minimaal twee runs.')
    parser.add_argument('--video-alpha', type=opacity, default=.40)
    parser.add_argument('--prediction-alpha', type=opacity, default=.35)
    parser.add_argument('--grid-hz', type=float, default=30.)
    parser.add_argument('--max-ros-gap', type=float, default=.10)
    parser.add_argument('--max-video-gap', type=float, default=.15)
    parser.add_argument('--onset-threshold-rad', type=float, default=1e-4)
    parser.add_argument('--command-limit-deg', type=float, default=3.)
    args = parser.parse_args()
    for name in ['grid_hz', 'max_ros_gap', 'max_video_gap', 'onset_threshold_rad', 'command_limit_deg']:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name.replace('_', '-')+' moet positief en eindig zijn')
    if args.runs_dir:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = (args.output_dir or args.runs_dir/f'figures_rebuilt_{stamp}').resolve()
        if output.exists():
            parser.error('Kies voor --runs-dir een nieuwe outputmap; bestaande resultaten worden niet overschreven.')
        data, sources = rebuild(args, output)
    else:
        if args.run_ids:
            parser.error('--run-ids vereist --runs-dir')
        source = (args.input or Path('mean_curves.csv')).resolve()
        sources = [{'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]
        data = pd.read_csv(source)
        output = (args.output_dir or source.parent/'figures_darker_sd').resolve()
    plot_curves(data, output, args.video_alpha, args.prediction_alpha)
    verify_sources(sources)
    print(f'Broncontrole geslaagd: alle {len(sources)} bronbestanden zijn ongewijzigd.')


if __name__ == '__main__':
    main()
