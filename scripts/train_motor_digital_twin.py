#!/usr/bin/env python3

import argparse
import json
import csv
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_TRAINING_DATA_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "setup_01_motors"
    / "training motor data NEW"
)

MOTOR_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}

def make_current_models():
    models = {
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),

        "polynomial_ridge": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0),
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    return models

def causal_moving_average(signal, window_size=5):
    signal = np.asarray(signal, dtype=float)

    filtered = np.full_like(signal, np.nan)

    for index in range(len(signal)):
        start_index = max(0, index - window_size + 1)
        window = signal[start_index:index + 1]

        finite = np.isfinite(window)

        if np.any(finite):
            filtered[index] = np.mean(window[finite])

    return filtered

def safe_float(value):
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def flatten_current_model_metrics(
    motor_index,
    model_name,
    overall_metrics,
    zone_metrics,
):
    row = {
        "motor_index": int(motor_index),
        "model": model_name,

        "overall_mae_mA": safe_float(overall_metrics.get("mae")),
        "overall_rmse_mA": safe_float(overall_metrics.get("rmse")),
        "overall_max_abs_error_mA": safe_float(overall_metrics.get("max_abs_error")),

        "peak_threshold_mA": safe_float(zone_metrics.get("peak_threshold_mA")),
    }

    zones = {
        "baseline": "baseline_current_mA",
        "moving": "moving_current_mA",
        "peak": "peak_current_mA",
    }

    for prefix, zone_name in zones.items():
        zone = zone_metrics.get(zone_name, {})

        row[f"{prefix}_mae_mA"] = safe_float(zone.get("mae"))
        row[f"{prefix}_rmse_mA"] = safe_float(zone.get("rmse"))
        row[f"{prefix}_max_abs_error_mA"] = safe_float(zone.get("max_abs_error"))

    return row


def current_model_selection_score(model_result):
    """
    Select the current model mainly on moving-current MAE.
    The goal is no longer to fit short transient peaks, but to estimate
    the expected current during commanded motor motion.
    """
    zone_metrics = model_result["zone_metrics"]

    if "moving_current_mA" in zone_metrics:
        return zone_metrics["moving_current_mA"]["mae"]

    return model_result["overall_metrics"]["mae"]


def write_current_model_comparison(current_model_rows, output_dir):
    if not current_model_rows:
        return

    output_dir = Path(output_dir)

    json_path = output_dir / "current_model_comparison_metrics.json"
    csv_path = output_dir / "current_model_comparison_metrics.csv"
    txt_path = output_dir / "current_model_comparison_metrics.txt"

    with open(json_path, "w") as f:
        json.dump(current_model_rows, f, indent=2)

    fieldnames = [
        "motor_index",
        "model",

        "overall_mae_mA",
        "overall_rmse_mA",
        "overall_max_abs_error_mA",

        "baseline_mae_mA",
        "baseline_rmse_mA",
        "baseline_max_abs_error_mA",

        "moving_mae_mA",
        "moving_rmse_mA",
        "moving_max_abs_error_mA",

        "peak_mae_mA",
        "peak_rmse_mA",
        "peak_max_abs_error_mA",
        "peak_threshold_mA",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in current_model_rows:
            writer.writerow(row)

    with open(txt_path, "w") as f:
        f.write("Motor current model comparison\n")
        f.write("==============================\n\n")
        f.write("Models are ranked per motor by moving-current MAE.\n\n")
        
        motor_indices = sorted(set(row["motor_index"] for row in current_model_rows))

        for motor_index in motor_indices:
            f.write(f"Motor {motor_index}\n")
            f.write("-" * 20 + "\n")

            motor_rows = [
                row for row in current_model_rows
                if row["motor_index"] == motor_index
            ]

            motor_rows = sorted(
                motor_rows,
                key=lambda row: (
                    row["moving_mae_mA"] is None,
                    row["moving_mae_mA"] if row["moving_mae_mA"] is not None else float("inf"),
                ),
            )

            for i, row in enumerate(motor_rows):
                f.write(f"{i + 1}. {row['model']}\n")
                f.write(f"   Overall MAE:       {row['overall_mae_mA']:.3f} mA\n")
                f.write(f"   Overall RMSE:      {row['overall_rmse_mA']:.3f} mA\n")
                f.write(f"   Overall max error: {row['overall_max_abs_error_mA']:.3f} mA\n")
                f.write(f"   Moving MAE:        {row['moving_mae_mA']:.3f} mA\n")
                f.write(f"   Peak MAE:          {row['peak_mae_mA']:.3f} mA\n")
                f.write(f"   Peak RMSE:         {row['peak_rmse_mA']:.3f} mA\n")
                f.write(f"   Peak max error:    {row['peak_max_abs_error_mA']:.3f} mA\n\n")

            f.write("\n")

    print(f"Saved current model comparison: {json_path}")
    print(f"Saved current model comparison: {csv_path}")
    print(f"Saved current model comparison: {txt_path}")

def make_encoder_models():
    # Zelfde modelopties als voor current, maar nu voor encoder response.
    return make_current_models()


def flatten_encoder_model_metrics(
    motor_index,
    model_name,
    encoder_metrics,
):
    return {
        "motor_index": int(motor_index),
        "model": model_name,
        "encoder_mae_pulses": safe_float(encoder_metrics.get("mae")),
        "encoder_rmse_pulses": safe_float(encoder_metrics.get("rmse")),
        "encoder_max_abs_error_pulses": safe_float(
            encoder_metrics.get("max_abs_error")
        ),
    }


def encoder_model_selection_score(model_result):
    # RMSE straft grotere position errors sterker dan MAE.
    return model_result["encoder_metrics"]["rmse"]


def write_encoder_model_comparison(encoder_model_rows, output_dir):
    if not encoder_model_rows:
        return

    output_dir = Path(output_dir)

    json_path = output_dir / "encoder_model_comparison_metrics.json"
    csv_path = output_dir / "encoder_model_comparison_metrics.csv"
    txt_path = output_dir / "encoder_model_comparison_metrics.txt"

    with open(json_path, "w") as f:
        json.dump(encoder_model_rows, f, indent=2)

    fieldnames = [
        "motor_index",
        "model",
        "encoder_mae_pulses",
        "encoder_rmse_pulses",
        "encoder_max_abs_error_pulses",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in encoder_model_rows:
            writer.writerow(row)

    with open(txt_path, "w") as f:
        f.write("Motor encoder response model comparison\n")
        f.write("=======================================\n\n")
        f.write("Models are ranked per motor by encoder RMSE.\n\n")

        motor_indices = sorted(set(row["motor_index"] for row in encoder_model_rows))

        for motor_index in motor_indices:
            f.write(f"Motor {motor_index}\n")
            f.write("-" * 20 + "\n")

            motor_rows = [
                row for row in encoder_model_rows
                if row["motor_index"] == motor_index
            ]

            motor_rows = sorted(
                motor_rows,
                key=lambda row: row["encoder_rmse_pulses"],
            )

            for i, row in enumerate(motor_rows):
                f.write(f"{i + 1}. {row['model']}\n")
                f.write(f"   Encoder MAE:       {row['encoder_mae_pulses']:.3f} pulses\n")
                f.write(f"   Encoder RMSE:      {row['encoder_rmse_pulses']:.3f} pulses\n")
                f.write(f"   Encoder max error: {row['encoder_max_abs_error_pulses']:.3f} pulses\n\n")

            f.write("\n")

    print(f"Saved encoder model comparison: {json_path}")
    print(f"Saved encoder model comparison: {csv_path}")
    print(f"Saved encoder model comparison: {txt_path}")

def parse_task_label(label):
    if not label:
        return {
            "setup": "unknown_setup",
            "test_type": "unknown_test",
            "motor_name": "unknown_motor",
            "trial": "unknown_trial",
            "timestamp": "unknown_time",
        }

    parts = label.split("|")

    return {
        "setup": parts[0] if len(parts) > 0 else "unknown_setup",
        "test_type": parts[1] if len(parts) > 1 else "unknown_test",
        "motor_name": parts[2] if len(parts) > 2 else "unknown_motor",
        "trial": parts[3] if len(parts) > 3 else "unknown_trial",
        "timestamp": parts[4] if len(parts) > 4 else "unknown_time",
    }


def safe_name(value):
    return str(value).replace("/", "_").replace("|", "_").replace(" ", "_")


def get_time_value(data):
    if "time" in data:
        return data["time"]

    if "timestamp" in data:
        return data["timestamp"]

    if "ros_timestamp" in data:
        return data["ros_timestamp"]

    return None


def load_motor_log(file_path):
    rows = []
    current_task_label = None

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            if data.get("task_label") is not None:
                current_task_label = data.get("task_label")
            elif data.get("task") is not None:
                current_task_label = data.get("task")

            t_value = get_time_value(data)
            positions = data.get("measured_motor_positions")
            currents = data.get("measured_currents")
            commands = data.get("commanded_motor_positions")

            if t_value is None or positions is None or currents is None:
                continue

            if len(positions) < 4 or len(currents) < 4:
                continue

            if commands is None or len(commands) < 4:
                commands = [0, 0, 0, 0]

            rows.append({
                "time": float(t_value),
                "task_label": current_task_label,
                "positions": [float(x) for x in positions[:4]],
                "currents": [float(x) for x in currents[:4]],
                "commands": [float(x) for x in commands[:4]],
            })

    if not rows:
        raise RuntimeError(f"No usable motor samples found in {file_path}")

    t0 = rows[0]["time"]

    for row in rows:
        row["time"] -= t0

    return rows


def split_into_segments(rows):
    segments = []
    current_segment = []
    current_label = None

    for row in rows:
        label = row["task_label"]

        if label != current_label:
            if current_segment:
                segments.append({
                    "task_label": current_label,
                    "rows": current_segment,
                    "info": parse_task_label(current_label),
                })

            current_label = label
            current_segment = []

        current_segment.append(row)

    if current_segment:
        segments.append({
            "task_label": current_label,
            "rows": current_segment,
            "info": parse_task_label(current_label),
        })

    clean_segments = []

    for segment in segments:
        label = segment["task_label"]
        test_type = segment["info"]["test_type"]

        if label is None:
            continue

        if label in ["unlabeled", "motor_tests_finished", "tests_finished"]:
            continue

        if test_type in ["unknown_test", "motor_tests_finished", "tests_finished"]:
            continue

        clean_segments.append(segment)

    return clean_segments


def get_active_motor_index(segment):
    motor_name = segment["info"]["motor_name"]

    if motor_name.startswith("m") and motor_name[1:].isdigit():
        motor_index = int(motor_name[1:])
        if 0 <= motor_index <= 3:
            return motor_index

    return None


def relative_commands_to_target(commands):
    """
    Converts repeated relative motor commands into a cumulative relative target.

    Example:
    command: 0, 300, 300, 0, -300, -300, 0
    target:  0, 300, 300, 300, 0,    0,    0
    """
    commands = np.asarray(commands, dtype=float)
    target = np.zeros_like(commands)

    for motor_index in range(commands.shape[0]):
        cumulative_target = 0.0
        previous_command = 0.0

        for sample_index in range(commands.shape[1]):
            command = commands[motor_index, sample_index]

            if command != previous_command:
                if command != 0.0:
                    cumulative_target += command

                previous_command = command

            target[motor_index, sample_index] = cumulative_target

    return target


def segment_to_arrays(segment):
    rows = segment["rows"]

    t = np.array([row["time"] for row in rows], dtype=float)

    positions = np.array(
        [row["positions"] for row in rows],
        dtype=float,
    ).T

    currents = np.array(
        [row["currents"] for row in rows],
        dtype=float,
    ).T

    commands = np.array(
        [row["commands"] for row in rows],
        dtype=float,
    ).T

    if len(t) > 0:
        t = t - t[0]

    commanded_target = relative_commands_to_target(commands)

    for motor_index in range(4):
        if positions.shape[1] > 0:
            positions[motor_index] = positions[motor_index] - positions[motor_index][0]

    return t, commands, commanded_target, positions, currents


def time_since_change(t, target):
    result = np.zeros_like(t, dtype=float)

    last_change_time = t[0] if len(t) > 0 else 0.0
    previous_target = target[0] if len(target) > 0 else 0.0

    for i in range(len(t)):
        if target[i] != previous_target:
            last_change_time = t[i]
            previous_target = target[i]

        result[i] = t[i] - last_change_time

    return result

def time_since_signal_change(t, signal):
    result = np.zeros_like(t, dtype=float)

    last_change_time = t[0] if len(t) > 0 else 0.0
    previous_signal = signal[0] if len(signal) > 0 else 0.0

    for i in range(len(t)):
        if signal[i] != previous_signal:
            last_change_time = t[i]
            previous_signal = signal[i]

        result[i] = t[i] - last_change_time

    return result

def safe_gradient(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return np.zeros_like(y)

    # Voorkomt problemen bij dubbele of bijna gelijke timestamps
    dt = np.diff(t)
    median_dt = np.nanmedian(dt[dt > 0.0]) if np.any(dt > 0.0) else 0.01

    t_safe = t.copy()
    for i in range(1, len(t_safe)):
        if t_safe[i] <= t_safe[i - 1]:
            t_safe[i] = t_safe[i - 1] + median_dt

    return np.gradient(y, t_safe)

def build_samples_from_segment(segment, motor_index):
    t, raw_commands, commanded_target, positions, currents = segment_to_arrays(segment)

    if len(t) < 3:
        return None

    target = commanded_target[motor_index]
    raw_command = raw_commands[motor_index]
    encoder = positions[motor_index]
    raw_current = currents[motor_index]
    filtered_current = causal_moving_average(
        raw_current,
        window_size=5,
    )

    current = filtered_current
    target_velocity = safe_gradient(target, t)
    time_after_change = time_since_change(t, target)

    # Instantaneous target step:
    # example target: 0, 0, 100, 100, 100, 0, 0
    # instant_target_delta: 0, 0, 100, 0, 0, -100, 0
    instant_target_delta = np.zeros_like(target)
    instant_target_delta[1:] = target[1:] - target[:-1]

    # Hold the last non-zero target step over the whole response phase.
    # This tells the model which movement caused the current response.
    target_delta = np.zeros_like(target)
    last_delta = 0.0

    for sample_index in range(len(target)):
        if instant_target_delta[sample_index] != 0.0:
            last_delta = instant_target_delta[sample_index]

        target_delta[sample_index] = last_delta

    abs_target_delta = np.abs(target_delta)
    target_direction = np.sign(target_delta)

    # Target-based moving indicator.
    # This replaces command_based_is_moving.
    movement_memory_s = 1.2
    target_based_is_moving = (
        (np.abs(target_delta) > 0.0)
        & (time_after_change < movement_memory_s)
    ).astype(float)
    # target_velocity = safe_gradient(target, t)
    # target_change = np.gradient(target)
    # target_direction = np.sign(target_velocity)
    # time_after_change = time_since_change(t, target)

    encoder_velocity = safe_gradient(encoder, t)
    encoder_acceleration = safe_gradient(encoder_velocity, t)

    encoder_error = target - encoder
    abs_encoder_error = np.abs(encoder_error)

    abs_target = np.abs(target)
    abs_target_velocity = np.abs(target_velocity)
    abs_raw_command = np.abs(raw_command)

    is_command_active = (raw_command != 0.0).astype(float)
    is_moving = ((np.abs(encoder_velocity) > 5.0) | (raw_command != 0.0)).astype(float)

    time_after_raw_command_change = time_since_signal_change(t, raw_command)
    command_direction = np.sign(raw_command)
    command_based_is_moving = is_command_active.copy()

    # Model 1: command -> encoder response
    # X_encoder = np.column_stack([
    #     target,
    #     # raw_command,
    #     abs_raw_command,
    #     command_direction,
    #     time_after_change,
    #     time_after_raw_command_change,
    #     abs_target,
    #     is_command_active,
    # ])
    X_encoder = np.column_stack([
        target,
        abs_target,
        target_delta,
        abs_target_delta,
        target_direction,
        time_after_change,
        target_based_is_moving,
    ])

    # Model 2: command-only -> current
    # X_current = np.column_stack([
    #     target,
    #     raw_command,
    #     abs_raw_command,
    #     command_direction,
    #     time_after_raw_command_change,
    #     abs_target,
    #     is_command_active,
    #     command_based_is_moving,
    # ])
    X_current = np.column_stack([
        target,
        abs_target,
        target_delta,
        abs_target_delta,
        target_direction,
        time_after_change,
        target_based_is_moving,
    ])

    y_encoder = encoder
    y_current = current

    metadata = {
        "test_type": segment["info"]["test_type"],
        "motor_name": segment["info"]["motor_name"],
        "trial": segment["info"]["trial"],
        "motor_index": motor_index,
        "time": t,
        "target": target,
        "raw_command": raw_command,
        "encoder": encoder,
        "current": current,
        "raw_current": raw_current,
        "filtered_current": filtered_current,
        "encoder_velocity": encoder_velocity,
        "encoder_error": encoder_error,
        "is_moving": is_moving,
    }

    return X_encoder, X_current, y_encoder, y_current, metadata

def build_dataset(segments):
    samples_by_motor = {0: [], 1: [], 2: [], 3: []}

    for segment in segments:
        info = segment["info"]
        active_motor = get_active_motor_index(segment)

        if info["test_type"] == "idle_baseline":
            motor_indices = range(4)
        elif active_motor is not None:
            motor_indices = [active_motor]
        else:
            continue

        for motor_index in motor_indices:
            sample = build_samples_from_segment(segment, motor_index)

            if sample is None:
                continue

            samples_by_motor[motor_index].append(sample)

    return samples_by_motor


def split_train_test(samples):
    train_samples = []
    test_samples = []

    for sample in samples:
        _, _, _, _, metadata = sample
        trial = metadata["trial"]

        if trial == "trial_03":
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    if not test_samples and len(samples) > 1:
        test_samples = [samples[-1]]
        train_samples = samples[:-1]

    if not train_samples:
        raise RuntimeError("No training samples available.")

    if not test_samples:
        raise RuntimeError("No test samples available.")

    return train_samples, test_samples


def stack_samples(samples):
    X_encoder_list = []
    X_current_list = []
    y_encoder_list = []
    y_current_list = []

    for X_encoder, X_current, y_encoder, y_current, _ in samples:
        X_encoder_list.append(X_encoder)
        X_current_list.append(X_current)
        y_encoder_list.append(y_encoder)
        y_current_list.append(y_current)

    X_encoder_all = np.vstack(X_encoder_list)
    X_current_all = np.vstack(X_current_list)
    y_encoder_all = np.concatenate(y_encoder_list)
    y_current_all = np.concatenate(y_current_list)

    return X_encoder_all, X_current_all, y_encoder_all, y_current_all


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "max_abs_error": float(max_error),
    }

def compute_current_zone_metrics(y_true, y_pred, X_current, current_feature_names):
    metrics = {}

    is_moving_index = current_feature_names.index("is_moving")
    moving_mask = X_current[:, is_moving_index] > 0.5
    baseline_mask = ~moving_mask

    # Peak-zone: hoogste 5% van gemeten current
    peak_threshold = np.nanpercentile(y_true, 95)
    peak_mask = y_true >= peak_threshold

    if np.any(baseline_mask):
        metrics["baseline_current_mA"] = compute_metrics(
            y_true[baseline_mask],
            y_pred[baseline_mask],
        )

    if np.any(moving_mask):
        metrics["moving_current_mA"] = compute_metrics(
            y_true[moving_mask],
            y_pred[moving_mask],
        )

    if np.any(peak_mask):
        metrics["peak_current_mA"] = compute_metrics(
            y_true[peak_mask],
            y_pred[peak_mask],
        )

    metrics["peak_threshold_mA"] = float(peak_threshold)

    return metrics

def find_motor_jsonl_files(input_dir):
    input_dir = Path(input_dir).expanduser()

    if not input_dir.exists():
        raise RuntimeError(f"Input folder does not exist: {input_dir}")

    candidates = []

    for path in input_dir.rglob("*.jsonl"):
        name = path.name.lower()

        if "angle" in name or "debug" in name:
            continue

        if path.stat().st_size == 0:
            continue

        candidates.append(path)

    if not candidates:
        raise RuntimeError(
            f"No usable .jsonl motor logs found in:\n{input_dir}\n\n"
            "Put one or more motor-only ROS logs in this folder first."
        )

    candidates = sorted(candidates)

    print("\nAutomatically selected training files:")
    for path in candidates:
        print(f"  - {path}")

    return candidates

def load_segments_from_files(file_paths):
    all_segments = []

    for file_path in file_paths:
        file_path = Path(file_path).expanduser()

        print(f"\nLoading motor log: {file_path}")
        rows = load_motor_log(file_path)

        print("Splitting into task segments...")
        segments = split_into_segments(rows)

        print(f"Found {len(segments)} valid segments in {file_path.name}.")

        for segment in segments:
            segment["source_file"] = str(file_path)

        all_segments.extend(segments)

    if not all_segments:
        raise RuntimeError("No valid task segments found in the selected JSONL files.")

    return all_segments

def train_motor_models(samples_by_motor, output_dir):
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    trained_models = {}
    current_model_comparison_rows = []
    encoder_model_comparison_rows = []


    # encoder_feature_names = [
    #     "commanded_target_pulses",
    #     "raw_relative_command_pulses",
    #     "abs_raw_relative_command_pulses",
    #     "command_direction",
    #     "time_since_target_change_s",
    #     "time_since_raw_command_change_s",
    #     "abs_commanded_target_pulses",
    #     "is_command_active",
    # ]
    # current_feature_names = [
    #     "commanded_target_pulses",
    #     "raw_relative_command_pulses",
    #     "abs_raw_relative_command_pulses",
    #     "command_direction",
    #     "time_since_raw_command_change_s",
    #     "abs_commanded_target_pulses",
    #     "is_command_active",
    #     "is_moving",
    # ]
    encoder_feature_names = [
        "commanded_target_pulses",
        "abs_commanded_target_pulses",
        "target_delta_pulses",
        "abs_target_delta_pulses",
        "target_direction",
        "time_since_target_change_s",
        "is_moving",
    ]
    current_feature_names = [
        "commanded_target_pulses",
        "abs_commanded_target_pulses",
        "target_delta_pulses",
        "abs_target_delta_pulses",
        "target_direction",
        "time_since_target_change_s",
        "is_moving",
    ]

    for motor_index, samples in samples_by_motor.items():
        if len(samples) < 2:
            print(f"Skipping motor {motor_index}: not enough samples.")
            continue

        train_samples, test_samples = split_train_test(samples)

        X_encoder_train, X_current_train, y_encoder_train, y_current_train = stack_samples(train_samples)
        X_encoder_test, X_current_test, y_encoder_test, y_current_test = stack_samples(test_samples)

        encoder_models = make_encoder_models()
        encoder_model_results = {}

        for encoder_model_name, encoder_model_candidate in encoder_models.items():
            encoder_model_candidate.fit(X_encoder_train, y_encoder_train)

            candidate_encoder_pred = encoder_model_candidate.predict(X_encoder_test)

            candidate_encoder_metrics = compute_metrics(
                y_encoder_test,
                candidate_encoder_pred,
            )

            encoder_model_results[encoder_model_name] = {
                "model": encoder_model_candidate,
                "prediction": candidate_encoder_pred,
                "encoder_metrics": candidate_encoder_metrics,
            }

            encoder_model_comparison_rows.append(
                flatten_encoder_model_metrics(
                    motor_index=motor_index,
                    model_name=encoder_model_name,
                    encoder_metrics=candidate_encoder_metrics,
                )
            )

        best_encoder_model_name = min(
            encoder_model_results,
            key=lambda name: encoder_model_selection_score(encoder_model_results[name]),
        )

        encoder_model = encoder_model_results[best_encoder_model_name]["model"]
        encoder_pred = encoder_model_results[best_encoder_model_name]["prediction"]
        encoder_metrics = encoder_model_results[best_encoder_model_name]["encoder_metrics"]
        
        current_models = make_current_models()
        current_model_results = {}

        for current_model_name, current_model_candidate in current_models.items():
            current_model_candidate.fit(X_current_train, y_current_train)

            candidate_current_pred = current_model_candidate.predict(X_current_test)

            candidate_overall_metrics = compute_metrics(
                y_current_test,
                candidate_current_pred,
            )

            candidate_zone_metrics = compute_current_zone_metrics(
                y_current_test,
                candidate_current_pred,
                X_current_test,
                current_feature_names,
            )

            current_model_results[current_model_name] = {
                "model": current_model_candidate,
                "prediction": candidate_current_pred,
                "overall_metrics": candidate_overall_metrics,
                "zone_metrics": candidate_zone_metrics,
            }

            current_model_comparison_rows.append(
                flatten_current_model_metrics(
                    motor_index=motor_index,
                    model_name=current_model_name,
                    overall_metrics=candidate_overall_metrics,
                    zone_metrics=candidate_zone_metrics,
                )
            )

        best_current_model_name = min(
            current_model_results,
            key=lambda name: current_model_selection_score(current_model_results[name]),
        )

        current_model = current_model_results[best_current_model_name]["model"]
        current_pred = current_model_results[best_current_model_name]["prediction"]
        current_overall_metrics = current_model_results[best_current_model_name]["overall_metrics"]
        current_zone_metrics = current_model_results[best_current_model_name]["zone_metrics"]
        motor_metrics = {
            "encoder_response_pulses": encoder_metrics,
            "selected_encoder_model": best_encoder_model_name,
            "encoder_model_comparison": {
                model_name: model_result["encoder_metrics"]
                for model_name, model_result in encoder_model_results.items()
            },            
            "motor_current_mA": current_overall_metrics,
            "motor_current_zones_mA": current_zone_metrics,
            "selected_current_model": best_current_model_name,
            "current_model_comparison": {
                model_name: {
                    "motor_current_mA": model_result["overall_metrics"],
                    "motor_current_zones_mA": model_result["zone_metrics"],
                }
                for model_name, model_result in current_model_results.items()
            },
            "n_train_samples": int(len(X_encoder_train)),
            "n_test_samples": int(len(X_encoder_test)),
            "train_segments": len(train_samples),
            "test_segments": len(test_samples),
            "encoder_feature_names": encoder_feature_names,
            "current_feature_names": current_feature_names,
        }

        metrics[f"motor_{motor_index}"] = motor_metrics

        model_package = {
            "motor_index": motor_index,
            "encoder_model": encoder_model,
            "current_model": current_model,
            "selected_encoder_model": best_encoder_model_name,
            "selected_current_model": best_current_model_name,
            "encoder_feature_names": encoder_feature_names,
            "current_feature_names": current_feature_names,
        }

        model_path = models_dir / f"motor_dt_m{motor_index}.pkl"
        joblib.dump(model_package, model_path)

        trained_models[motor_index] = model_package

        print(f"\nMotor {motor_index}")
        print(f"  Encoder MAE: {motor_metrics['encoder_response_pulses']['mae']:.2f} pulses")
        print(f"  Current MAE: {motor_metrics['motor_current_mA']['mae']:.2f} mA")
        print(f"  Saved model: {model_path}")

    metrics_path = output_dir / "motor_dt_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")
    write_encoder_model_comparison(
        encoder_model_rows=encoder_model_comparison_rows,
        output_dir=output_dir,
    )
    
    write_current_model_comparison(
        current_model_rows=current_model_comparison_rows,
        output_dir=output_dir,
    )

    return trained_models, metrics


def plot_predictions(samples_by_motor, trained_models, output_dir):
    plot_dir = output_dir / "prediction_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for motor_index, samples in samples_by_motor.items():
        if motor_index not in trained_models:
            continue

        _, test_samples = split_train_test(samples)

        encoder_model = trained_models[motor_index]["encoder_model"]
        current_model = trained_models[motor_index]["current_model"]

        for X_encoder, X_current, y_encoder, y_current, metadata in test_samples:
            t = metadata["time"]
            target = metadata["target"]

            encoder_pred = encoder_model.predict(X_encoder)
            current_pred = current_model.predict(X_current)

            test_type = metadata["test_type"]
            trial = metadata["trial"]

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            fig.suptitle(
                f"Motor DT prediction - motor {motor_index} - {test_type} - {trial}",
                fontsize=14,
            )

            axes[0].plot(
                t,
                target,
                linestyle="--",
                linewidth=1.3,
                color="black",
                label="commanded target",
            )

            axes[0].plot(
                t,
                y_encoder,
                linewidth=1.5,
                color=MOTOR_COLORS[motor_index],
                label="measured encoder response",
            )

            axes[0].plot(
                t,
                encoder_pred,
                linewidth=1.5,
                color="tab:purple",
                label="predicted encoder response",
            )

            axes[0].set_ylabel("Relative pulse count [pulses]")
            axes[0].grid(True)
            axes[0].legend(fontsize=8)

            axes[1].plot(
                t,
                y_current,
                linewidth=1.4,
                color=MOTOR_COLORS[motor_index],
                label="filtered measured current",
            )

            axes[1].plot(
                t,
                current_pred,
                linewidth=1.4,
                color="tab:purple",
                label="predicted current",
            )

            axes[1].set_ylabel("Current [mA]")
            axes[1].set_xlabel("Time [s]")
            axes[1].grid(True)
            axes[1].legend(fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.94])

            filename = (
                f"motor_{motor_index}_"
                f"{safe_name(test_type)}_"
                f"{safe_name(trial)}.png"
            )

            plot_path = plot_dir / filename
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

            print(f"Saved prediction plot: {plot_path}")


def train_motor_digital_twin(file_paths, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    segments = load_segments_from_files(file_paths)

    print(f"\nTotal valid segments: {len(segments)}")

    samples_by_motor = build_dataset(segments)

    for motor_index, samples in samples_by_motor.items():
        print(f"Motor {motor_index}: {len(samples)} segments")

    trained_models, _ = train_motor_models(samples_by_motor, output_dir)
    plot_predictions(samples_by_motor, trained_models, output_dir)

    print(f"\nMotor Digital Twin training completed.")
    print(f"Output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a Motor Digital Twin from motor-only JSONL data."
    )

    parser.add_argument(
        "--file",
        default=None,
        help="Optional path to a specific motor-only .jsonl log file.",
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_TRAINING_DATA_DIR),
        help="Folder where the training .jsonl file is searched automatically.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder where models, metrics and plots will be saved. "
             "If omitted, a motor_dt_results folder is created inside the input folder.",
    )

    args = parser.parse_args()
    if args.file is not None:
        file_paths = [Path(args.file).expanduser()]
    else:
        file_paths = find_motor_jsonl_files(args.input_dir)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = Path(args.input_dir).expanduser() / "motor_dt_results"

    train_motor_digital_twin(
        file_paths=file_paths,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()