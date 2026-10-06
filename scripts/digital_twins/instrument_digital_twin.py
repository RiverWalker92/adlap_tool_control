#!/usr/bin/env python3
import math

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Sequence, Dict

# Training and runtime share this feature schema. Gearbox units are unchanged;
# commanded_instrument_angles[1] is the requested DOF2 bend, in radians.
HYBRID_BEND_FEATURE_COLUMNS = [f"gearbox_{i}" for i in range(6)] + ["cmd_instr_1"]
HYBRID_BEND_FEATURE_SCHEMA_VERSION = 2


def build_hybrid_bend_features(gearbox_output, commanded_bend_rad=None):
    """Build instantaneous inputs; do not replace a missing command by zero."""
    if len(gearbox_output) != 6:
        raise ValueError(f"Expected 6 gearbox states, got {len(gearbox_output)}")
    values = {f"gearbox_{i}": float(value) for i, value in enumerate(gearbox_output)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Gearbox states must all be finite")
    if commanded_bend_rad is not None:
        bend = float(commanded_bend_rad)
        if not math.isfinite(bend):
            raise ValueError("Commanded DOF2 bend must be finite, in radians")
        values["cmd_instr_1"] = bend
    return values


class HybridBendResidualModel:
    """Load the residual model; support legacy gearbox-only and command models.

    Output is the DOF2 bend correction in degrees, added to physics in radians.
    Video measurements remain the training target, never an online input.
    """

    def __init__(self, model_path):
        # Keep the physics-only DT usable without ML packages installed.
        import joblib
        saved = joblib.load(Path(model_path).expanduser())
        self.model = saved["model"]
        self.feature_columns = list(saved["feature_columns"])
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Model feature_columns must be nonempty and unique")
        unsupported = [col for col in self.feature_columns if col not in HYBRID_BEND_FEATURE_COLUMNS]
        if unsupported:
            raise ValueError(f"Unsupported Hybrid Bend inputs: {unsupported}")
        if saved.get("command_unit", "radians") != "radians":
            raise ValueError("Hybrid Bend command must use radians")
        if saved.get("residual_unit", "degrees") != "degrees":
            raise ValueError("Hybrid Bend residual must use degrees")
        self.requires_command = "cmd_instr_1" in self.feature_columns

    def predict_residual_deg(self, gearbox_output, commanded_bend_rad=None):
        import pandas as pd
        values = build_hybrid_bend_features(gearbox_output, commanded_bend_rad)
        missing = [col for col in self.feature_columns if col not in values]
        if missing:
            raise ValueError(f"Hybrid Bend inputs not yet available: {missing}")
        x = pd.DataFrame([[values[col] for col in self.feature_columns]], columns=self.feature_columns)
        residual = float(self.model.predict(x)[0])
        if not math.isfinite(residual):
            raise ValueError("Hybrid Bend model returned a nonfinite residual")
        return residual


class InstrumentDigitalTwin:
    """
    Converts gearbox output states to predicted instrument output angles.
    Input order:
    [
        inner_shaft_rotation_deg,
        inner_shaft_relative_rotation_deg,
        inner_shaft_translation_mm,
        middle_shaft_rotation_deg,
        outer_shaft_rotation_deg,
        middle_outer_relative_rotation_deg,
    ]
    Output order:
    [   dof1 = shaft_roll,
        dof2 = bend,
        dof3 = tip_rotation,
        dof4 = articulation
    ]
    all in radians
    """
# Load instrument calibration parameters from the YAML
# configuration file.
    def __init__(self, config_path="config/dof_pattern_params.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["instrument"]
# self.articulation_factor = config["articulation_factor"]
        self.jaw_angle_deg_per_mm = config["jaw_angle_deg_per_mm"]
        self.between_jaws_deg_per_mm = config["between_jaws_deg_per_mm"]
        self.jaw_opening_threshold_mm = config["jaw_opening_threshold_mm"]
        self.jaw_closing_threshold_mm = config["jaw_closing_threshold_mm"]
        self.jaw_opening_backlash_mm = config["jaw_opening_backlash_mm"]
        self.jaw_closing_backlash_mm = config["jaw_closing_backlash_mm"]
        self.bend_factor = config["bend_factor"]
        self.instrument_bend_offset_deg = config["instrument_bend_offset_deg"]
        self.instrument_bend_backlash_positive_deg = config["instrument_bend_backlash_positive_deg"]
        self.instrument_bend_backlash_negative_deg = config["instrument_bend_backlash_negative_deg"]
        self.jaw_translation_min_mm = config.get("jaw_translation_min_mm", None)
        self.jaw_translation_max_mm = config.get("jaw_translation_max_mm", None)
        self.usable_jaw_translation_min_mm = config.get(
            "usable_jaw_translation_min_mm",
            0.0,
        )
        self.usable_jaw_translation_max_mm = config.get(
            "usable_jaw_translation_max_mm",
            self.jaw_translation_max_mm,
        )
        self.shaft_roll_sign = config["shaft_roll_sign"]
        self.bend_sign = config["bend_sign"]
        self.tip_rotation_sign = config["tip_rotation_sign"]
        self.articulation_sign = config["articulation_sign"]
        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None
        self._last_usable_translation_mm = None
        self._last_jaw_direction = 0
# Reset backlash memory before starting a new
# trial or independent simulation.
    def reset(self):
        self._jaw_effective_translation_mm = None
        self._effective_bend_deg = None
        self._last_usable_translation_mm = None
        self._last_jaw_direction = 0
    def apply_jaw_backlash(self, translation_mm: float) -> float:
        opening_backlash = self.jaw_opening_backlash_mm
        closing_backlash = self.jaw_closing_backlash_mm
        if self._jaw_effective_translation_mm is None:
            self._jaw_effective_translation_mm = translation_mm
            return translation_mm
        previous_effective = self._jaw_effective_translation_mm
# If the jaw is opening enough to overcome backlash,
# the effective translation starts moving.
        if translation_mm > previous_effective + opening_backlash:
            effective_translation = translation_mm - opening_backlash
#If the jaw is closing enough to overcome backlash, the
# effective translation starts moving in the other direction.
        elif translation_mm < previous_effective - closing_backlash:
            effective_translation = translation_mm + closing_backlash
# Small movements are absorbed by backlash and do not change the output.
        else:
            effective_translation = previous_effective
        self._jaw_effective_translation_mm = effective_translation
        return effective_translation
# Apply direction-dependent bend backlash.
# The effective bend only changes once the raw bend
# exceeds the backlash threshold.
    def apply_instrument_bend_backlash(self, raw_bend_deg: float) -> float:
        if self._effective_bend_deg is None:
            self._effective_bend_deg = raw_bend_deg
            return raw_bend_deg
        previous = self._effective_bend_deg
        if raw_bend_deg > previous:
            backlash = self.instrument_bend_backlash_positive_deg
        else:
            backlash = self.instrument_bend_backlash_negative_deg
        if raw_bend_deg > previous + backlash:
            effective = raw_bend_deg - backlash
        elif raw_bend_deg < previous - backlash:
            effective = raw_bend_deg + backlash
        else:
            effective = previous
        self._effective_bend_deg = effective
        return effective
# Main conversion from gearbox output vector
# to instrument output angles.
    def euler_angles_from_gearbox_output(
        self,
        gearbox_output: Sequence[float],
    ) -> Dict[str, float]:
# Unpack gearbox states. The order must match the output
# order of the gearbox digital twin.
        inner_shaft_rotation_deg = gearbox_output[0]
        inner_shaft_relative_rotation_deg = gearbox_output[1]
        inner_shaft_translation_mm = gearbox_output[2]
        middle_shaft_rotation_deg = gearbox_output[3]
        outer_shaft_rotation_deg = gearbox_output[4]
        middle_outer_relative_rotation_deg = gearbox_output[5]
        shaft_roll = math.radians(
            self.shaft_roll_sign * middle_shaft_rotation_deg
        )
# bend = math.radians(
#     self.params.bend_sign
#     * middle_outer_relative_rotation_deg
#     / self.params.bend_factor
# )
        raw_bend_deg = (
            self.bend_sign
            * middle_outer_relative_rotation_deg
            / self.bend_factor
        )
# A fixed bend offset is added > is now 0 so delete?
        raw_bend_deg = raw_bend_deg + self.instrument_bend_offset_deg
# Backlash is applied
        bend_deg = self.apply_instrument_bend_backlash(raw_bend_deg)
        bend = math.radians(bend_deg)
        tip_rotation = math.radians(
            self.tip_rotation_sign * inner_shaft_rotation_deg
        )
        signed_translation_mm = (
            self.articulation_sign
            * inner_shaft_translation_mm
        )
# Clamp instrument jaw translation to the
# usable/physical instrument range
        jaw_translation_mm = signed_translation_mm
        if self.jaw_translation_min_mm is not None:
            jaw_translation_mm = max(
                self.jaw_translation_min_mm,
                jaw_translation_mm,
            )
        if self.jaw_translation_max_mm is not None:
            jaw_translation_mm = min(
                self.jaw_translation_max_mm,
                jaw_translation_mm,
            )
# Usable opening range: below usable_jaw_translation_min_mm,
# the shaft may translate, but the jaws do not visibly open yet.
        usable_translation_mm = jaw_translation_mm - self.usable_jaw_translation_min_mm
        if self.usable_jaw_translation_max_mm is not None:
            usable_range_mm = (
                self.usable_jaw_translation_max_mm
                - self.usable_jaw_translation_min_mm
            )
            usable_translation_mm = min(
                usable_range_mm,
                usable_translation_mm,
            )
        usable_translation_mm = max(0.0, usable_translation_mm)
# Determine whether the jaw mechanism is opening or closing
# based on the raw usable translation before backlash.
        if self._last_usable_translation_mm is None:
            jaw_direction = 0
        else:
            delta_translation = usable_translation_mm - self._last_usable_translation_mm
            if delta_translation > 1e-6:
                jaw_direction = 1      # opening
            elif delta_translation < -1e-6:
                jaw_direction = -1     # closing
            else:
                jaw_direction = self._last_jaw_direction
        self._last_usable_translation_mm = usable_translation_mm
        self._last_jaw_direction = jaw_direction
# Apply direction-dependent jaw backlash.
        backlash_translation_mm = self.apply_jaw_backlash(usable_translation_mm)
# Apply direction-dependent threshold, but keep one shared gain.
        if jaw_direction < 0:
            jaw_threshold_mm = self.jaw_closing_threshold_mm
        else:
            jaw_threshold_mm = self.jaw_opening_threshold_mm
        opening_translation_mm = max(
            0.0,
            backlash_translation_mm - jaw_threshold_mm
        )
        articulation_one_jaw_deg = (
            opening_translation_mm
            * self.jaw_angle_deg_per_mm
        )
        articulation_deg = (
            opening_translation_mm
            * self.between_jaws_deg_per_mm
        )
        articulation_one_jaw = math.radians(articulation_one_jaw_deg)
        articulation = math.radians(articulation_deg)
        l1 = math.tan(bend)
        h = -l1 * math.cos(shaft_roll)
        w = -l1 * math.sin(shaft_roll)
# Resolve the bend direction into pitch and yaw based
# on the current shaft roll angle.
        pitch = -math.atan(h)
        yaw = -math.atan(w)
# Resolve the bend direction into pitch and yaw based
# on the current shaft roll angle.
        return {
            "tip_rotation": tip_rotation,
            "pitch": pitch,
            "yaw": yaw,
            "articulation": articulation, #between_jaws
            "articulation_one_jaw": articulation_one_jaw,
            "shaft_roll": shaft_roll,
            "raw_bend": math.radians(raw_bend_deg),
            "bend": bend,
        }


# Offline replay uses the saved physics output, matching the residual training target.
def _replay_model_from_config(config_path):
    config_path = config_path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    data_root = Path(config['test_data_root']).expanduser()
    training_root = Path(config['training_data_root']).expanduser()
    if not training_root.is_absolute():
        training_root = data_root / training_root
    output = Path(config['hybrid_bend_dt']['output_dir']).expanduser()
    if not output.is_absolute():
        output = training_root / output
    metrics_file = output / 'hybrid_bend_metrics.json'
    metrics = json.loads(metrics_file.read_text())
    return (output / metrics['selected_model_file']).resolve(), metrics


def _replay_find_plotter(scripts_dir):
    for name in ['plot_gearbox_instrument.py', 'plot_dof.py']:
        candidates = sorted(scripts_dir.rglob(name))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f'Multiple {name} files found; select one with --plotter')
    raise FileNotFoundError('No instrument plotter found; select its path with --plotter')


def _replay_atomic_jsonl(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replay_log_file(source, model, provenance, in_place=False):
    import numpy as np
    import pandas as pd

    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    indices, feature_rows = [], []
    for i, row in enumerate(rows):
        # Clear stale hybrid predictions, including invalid startup rows.
        row['hybrid_predicted_instrument_angles'] = None
        row['hybrid_predicted_instrument_angles_timestamp'] = None
        row['hybrid_bend_residual_deg'] = None
        try:
            physics = row.get('predicted_instrument_angles')
            if not isinstance(physics, list) or len(physics) < 4:
                continue
            if not all(math.isfinite(float(v)) for v in physics):
                continue
            gearbox = row.get('gearbox_state')
            if not isinstance(gearbox, list):
                continue
            command = row.get('commanded_instrument_angles')
            bend_command = (float(command[1]) if isinstance(command, list) and len(command) >= 2
                            else None)
            values = build_hybrid_bend_features(gearbox, bend_command)
            feature_rows.append([values[col] for col in model.feature_columns])
            indices.append(i)
        except (TypeError, ValueError, KeyError):
            continue
    if not indices:
        raise RuntimeError('No samples with finite physics, gearbox states and required commands')
    X = pd.DataFrame(feature_rows, columns=model.feature_columns)
    residuals = np.asarray(model.model.predict(X), dtype=float).reshape(-1)
    if len(residuals) != len(indices) or not np.isfinite(residuals).all():
        raise RuntimeError('Model returned invalid residual predictions')
    for i, residual_deg in zip(indices, residuals):
        row = rows[i]
        hybrid = [float(v) for v in row['predicted_instrument_angles']]
        correction = math.radians(float(residual_deg))
        hybrid[1] += correction
        if len(hybrid) > 4:
            hybrid[4] += correction  # same bend/pitch composition as the state node
        row['hybrid_predicted_instrument_angles'] = hybrid
        row['hybrid_predicted_instrument_angles_timestamp'] = row.get('ros_timestamp')
        row['hybrid_bend_residual_deg'] = float(residual_deg)
    run = source.name.removesuffix('_ros_log.jsonl')
    replay = source if in_place else source.with_name(run+'_hybrid_bend_replay.jsonl')
    suffix = '_hybrid_bend_update_metadata.json' if in_place else '_hybrid_bend_replay_metadata.json'
    metadata_path = source.with_name(run+suffix)
    metadata = dict(provenance, source_ros_log=str(source), replay_file=str(replay),
                    run=run, samples_total=len(rows), samples_predicted=len(indices),
                    samples_without_prediction=len(rows)-len(indices),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    physics_source='saved predicted_instrument_angles; physics not recomputed',
                    composition='saved physics radians + predicted residual degrees converted to radians',
                    bias_correction_applied=False,
                    updated_in_place=in_place,
                    training_member=run in provenance.get('training_trials', []),
                    designated_held_out_run=run == provenance.get('test_trial'))
    if in_place:
        # Back up the exact original bytes before atomically replacing the log.
        # Backup filenames do not match the input *_ros_log.jsonl glob.
        import shutil
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = source.with_name(source.name+'.bak_'+stamp)
        with backup.open('xb') as destination, source.open('rb') as original:
            shutil.copyfileobj(original, destination)
        metadata['backup_file'] = str(backup)
    _replay_atomic_jsonl(replay, rows)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return replay, metadata


def main():
    parser = argparse.ArgumentParser(description='Instrument Digital Twin and offline Hybrid Bend replay')
    commands = parser.add_subparsers(dest='command', required=True)
    replay_parser = commands.add_parser('replay', aliases=['update-logs'], help='Apply Hybrid Bend; update-logs updates original logs with backups')
    inputs = replay_parser.add_mutually_exclusive_group(required=True)
    package = Path.home() / 'ros2_ws/src/adlap_tool_control'
    inputs.add_argument('--base-dir', type=Path, help='Recursively process auto_dof2_*_ros_log.jsonl')
    inputs.add_argument('--file', type=Path, help='Process one original *_ros_log.jsonl')
    replay_parser.add_argument('--training-config', type=Path, default=package/'config/training_config.yaml')
    replay_parser.add_argument('--model-file', type=Path, help='Optional override for the selected model')
    replay_parser.add_argument('--scripts-dir', type=Path, default=package/'scripts')
    replay_parser.add_argument('--params-file', type=Path, default=package/'config/dof_pattern_params.yaml')
    replay_parser.add_argument('--in-place', action='store_true', help='Update the original ROS logs with backups; automatic for update-logs')
    replay_parser.add_argument('--plot', action='store_true', help='Plot the updated predictions together with the matching webcam angles')
    replay_parser.add_argument('--plotter', type=Path)
    replay_parser.add_argument('--plot-dir-name', default='instrument_outputs', help='Output subfolder under each run/plots; existing plots there can be replaced')
    args = parser.parse_args()
    in_place = args.in_place or args.command == 'update-logs'
    base = (args.base_dir.expanduser().resolve() if args.base_dir
            else args.file.expanduser().resolve().parent)
    scripts = args.scripts_dir.expanduser().resolve()
    if args.model_file:
        model_path = args.model_file.expanduser().resolve()
        metrics = {}
    else:
        model_path, metrics = _replay_model_from_config(args.training_config)
    model = HybridBendResidualModel(model_path)
    provenance = dict(model_file=str(model_path),
                      model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                      feature_columns=model.feature_columns, command_unit='radians', residual_unit='degrees',
                      training_trials=metrics.get('training_trials', []), test_trial=metrics.get('test_trial'))
    plotter = (args.plotter.expanduser().resolve() if args.plotter else _replay_find_plotter(scripts)) if args.plot else None
    if plotter is not None and not plotter.is_file():
        raise FileNotFoundError(plotter)
    files = ([args.file.expanduser().resolve()] if args.file
             else sorted(base.rglob('auto_dof2_*_ros_log.jsonl')))
    for source in files:
        if not source.is_file() or not source.name.endswith('_ros_log.jsonl'):
            raise ValueError(f'Expected an original *_ros_log.jsonl file: {source}')
    if not files:
        raise FileNotFoundError(f'No original DOF2 ROS logs found in {base}')
    print(f'Model: {model_path}\nInputs: {model.feature_columns}\nRuns: {len(files)}', flush=True)
    summaries = []
    for source in files:
        try:
            replay, summary = _replay_log_file(source, model, provenance, in_place=in_place)
            print(f"{summary['run']}: {summary['samples_predicted']}/{summary['samples_total']} predictions -> {replay.name}", flush=True)
            summary['plot_status'] = 'not_requested'
            if args.plot:
                video = source.with_name(source.name.replace('_ros_log.jsonl', '_webcam_angles.jsonl'))
                if video.is_file() and video.stat().st_size:
                    plot_dir = source.parent/'plots'/args.plot_dir_name
                    command = [sys.executable, str(plotter), '--file', str(replay),
                               '--video-angles', str(video), '--output-dir', str(plot_dir),
                               '--params-file', str(args.params_file.expanduser()),
                               '--coupling-mode', 'full_setup', '--dof', '2']
                    subprocess.run(command, check=True, env={**os.environ, 'MPLBACKEND': 'Agg'})
                    summary['plot_status'] = 'succeeded'
                    summary['plot_dir'] = str(plot_dir)
                else:
                    summary['plot_status'] = 'skipped_missing_video'
                    print(f'Video missing for {source.name}; predictions saved, plots skipped', flush=True)
            summaries.append(summary)
        except Exception as exc:
            print(f'FAILED {source}: {exc}', flush=True)
            summaries.append(dict(source_ros_log=str(source), error=str(exc)))
    report = base/('hybrid_bend_update_summary.json' if in_place else 'hybrid_bend_replay_summary.json')
    report.write_text(json.dumps(dict(provenance, runs=summaries), indent=2)+'\n')
    failed = sum('error' in row for row in summaries)
    print(f'Completed: {len(summaries)-failed}; failed: {failed}; summary: {report}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
