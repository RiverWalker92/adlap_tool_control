#!/usr/bin/env python3

import argparse
import json
import csv
import copy
import math
import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

VALID_CONFIGURATIONS = (
    "motor_only",
    "gearbox_only",
    "full_setup",
)

RUN_SPLIT_SEED = 42
EVENT_TAIL_S = 1.0
STATIONARY_ENCODER_VELOCITY_THRESHOLD = 5.0
MOTOR_COMMAND_TOLERANCE_PULSES = 0.5

DEFAULT_GEARBOX_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "gearbox_params.yaml"
)

MOTOR_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}

DEFAULT_TRAINING_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "training_config.yaml"
)

DYNAMIC_COMPLETION_HOLD_S = 0.1
DYNAMIC_MIN_POSITION_TOLERANCE_PULSES = 5.0
DYNAMIC_REL_POSITION_TOLERANCE = 0.01


def load_training_config(config_path: Path):
    config_path = Path(config_path).expanduser()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_data_root = Path(
        config["test_data_root"]
    ).expanduser()

    training_data_root = Path(
        config["training_data_root"]
    ).expanduser()

    if not training_data_root.is_absolute():
        training_data_root = (
            test_data_root / training_data_root
        )

    return config, training_data_root


def resolve_training_path(training_data_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return training_data_root / path


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
    protocol_metrics = model_result.get("by_protocol", {})
    protocol_scores = []

    for result in protocol_metrics.values():
        dynamic = result.get("by_phase", {}).get("dynamic")
        if dynamic is not None:
            protocol_scores.append(dynamic["mae"])

    if protocol_scores:
        # Give every held-out DOF equal weight, irrespective of run length.
        return float(np.mean(protocol_scores))

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
    protocol_metrics = model_result.get("by_protocol", {})
    protocol_scores = []

    for result in protocol_metrics.values():
        dynamic = result.get("by_phase", {}).get("dynamic")
        if dynamic is not None:
            protocol_scores.append(dynamic["rmse"])

    if protocol_scores:
        # Give every held-out DOF equal weight, irrespective of run length.
        return float(np.mean(protocol_scores))

    # RMSE penalizes larger position errors more strongly than MAE.
    return model_result["encoder_metrics"]["rmse"]


def select_common_model_family(task_results):
    """
    Choose one model family for every motor and both prediction targets.

    Each motor/output task first selects its own winner using the existing
    DOF-balanced dynamic-error score. The family with the most task wins is
    then selected globally. Ties are resolved by the lowest mean rank across
    all tasks, followed by the model name for complete reproducibility.
    """
    if not task_results:
        raise RuntimeError("No model-comparison results available.")

    common_names = set(task_results[0]["results"])
    for task in task_results[1:]:
        common_names &= set(task["results"])

    if not common_names:
        raise RuntimeError(
            "Encoder and current comparisons have no common model families."
        )

    vote_counts = {name: 0 for name in sorted(common_names)}
    rank_sums = {name: 0.0 for name in sorted(common_names)}
    local_selections = []

    for task in task_results:
        score_function = task["score_function"]
        scores = {
            name: float(score_function(task["results"][name]))
            for name in common_names
        }
        ranking = sorted(common_names, key=lambda name: (scores[name], name))
        local_winner = ranking[0]
        vote_counts[local_winner] += 1

        for rank, name in enumerate(ranking, start=1):
            rank_sums[name] += rank

        local_selections.append({
            "motor_index": int(task["motor_index"]),
            "output": task["output"],
            "winner": local_winner,
            "ranking": ranking,
            "selection_scores": scores,
        })

    task_count = float(len(task_results))
    mean_ranks = {
        name: rank_sums[name] / task_count
        for name in sorted(common_names)
    }
    selected_name = min(
        common_names,
        key=lambda name: (-vote_counts[name], mean_ranks[name], name),
    )

    return selected_name, {
        "policy": "majority_vote_over_motor_output_task_winners",
        "number_of_tasks": len(task_results),
        "selected_common_model_family": selected_name,
        "vote_counts": vote_counts,
        "mean_ranks": mean_ranks,
        "tie_break": "lowest_mean_rank_then_alphabetical_model_name",
        "local_selections": local_selections,
    }


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


def cpp_round(value):
    """Match C++ std::round (halfway values round away from zero)."""
    value = float(value)
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def load_gearbox_conversion_parameters(config_path, gearbox_variant=None):
    config_path = Path(config_path).expanduser().resolve()

    with open(config_path, "r") as f:
        document = yaml.safe_load(f)

    gearbox = document.get("gearbox", {})
    variants = gearbox.get("variants", {})
    selected_variant = gearbox_variant or gearbox.get("active_variant")

    if selected_variant not in variants:
        available = ", ".join(sorted(variants)) or "none"
        raise RuntimeError(
            f"Unknown gearbox variant '{selected_variant}' in {config_path}. "
            f"Available variants: {available}."
        )

    variant = variants[selected_variant]
    base_pulses = int(variant["encoder"]["pulses_per_motor_rotation"])
    upper_factor = float(variant["ratios"]["upper_motor_factor"])
    lower_factor = float(variant["ratios"]["lower_motor_factor"])
    upper_motors = {int(index) for index in variant["motor_groups"]["upper_motors"]}

    # C++ Gearbox::get_pulses_per_rotation casts the product to int.
    pulses_per_rotation = np.array([
        int(base_pulses * (upper_factor if index in upper_motors else lower_factor))
        for index in range(4)
    ], dtype=float)

    lower_play_degrees = float(
        variant.get("backlash", {}).get("lower_motors_play_deg", 0.0)
    )

    return {
        "variant": selected_variant,
        "pulses_per_rotation": pulses_per_rotation,
        "lower_motors_play_pulses": cpp_round(
            lower_play_degrees * pulses_per_rotation[1] / 360.0
        ),
        # These values are hard-coded in instrument_controller.hpp and are
        # deliberately kept identical here.
        "bend_factor": 1.5,
        "articulation_factor": 16.0,
        "max_bend_angle_degrees": 45.0,
    }


class InstrumentCommandConverter:
    """Stateful Python equivalent of InstrumentController's joint conversion."""

    def __init__(self, parameters, starting_positions):
        self.pulses_per_rotation = np.asarray(
            parameters["pulses_per_rotation"], dtype=float
        )
        self.lower_play = int(parameters["lower_motors_play_pulses"])
        self.bend_factor = float(parameters["bend_factor"])
        self.articulation_factor = float(parameters["articulation_factor"])
        self.max_bend_degrees = float(parameters["max_bend_angle_degrees"])
        self.starting_positions = np.asarray(starting_positions, dtype=float)

        self.absolute_shaft_roll = 0.0
        self.bend_play_compensation = 0
        self.m1_m2_offset = 0

    def pulses_per_degree(self, motor_index):
        return self.pulses_per_rotation[motor_index] / 360.0

    def calculate_real_play(self, current_positions):
        current_positions = np.asarray(current_positions, dtype=float)
        starting_difference = (
            self.starting_positions[2] - self.starting_positions[1]
        )
        difference = int(
            current_positions[2]
            - current_positions[1]
            - starting_difference
        )

        if difference > 0:
            if difference > self.m1_m2_offset + 2 * self.lower_play:
                self.m1_m2_offset = difference - 2 * self.lower_play
            elif difference < self.m1_m2_offset - 2 * self.lower_play:
                self.m1_m2_offset = difference + 2 * self.lower_play
        else:
            if difference < self.m1_m2_offset - 2 * self.lower_play:
                self.m1_m2_offset = difference + 2 * self.lower_play
            elif difference > self.m1_m2_offset + 2 * self.lower_play:
                self.m1_m2_offset = difference - 2 * self.lower_play

        # C++ integer division truncates toward zero.
        return int((difference - self.m1_m2_offset) / 2)

    def get_motor2_value_for_angle(self, bend_radians, current_positions):
        degrees = np.clip(
            math.degrees(float(bend_radians)),
            -self.max_bend_degrees,
            self.max_bend_degrees,
        )
        starting_difference = (
            self.starting_positions[2] - self.starting_positions[1]
        )
        real_play = self.calculate_real_play(current_positions)
        current_difference = int(
            current_positions[2]
            - current_positions[1]
            - starting_difference
            - 2 * real_play
        )
        wanted_difference = cpp_round(
            degrees * self.pulses_per_degree(1) * self.bend_factor
        )
        relative_difference = wanted_difference - current_difference
        deadband = cpp_round(self.pulses_per_degree(1) * 2.5)

        if self.bend_play_compensation > 0:
            if relative_difference < -deadband:
                self.bend_play_compensation = -self.lower_play
            elif relative_difference <= 0:
                return wanted_difference
        else:
            if relative_difference > deadband:
                self.bend_play_compensation = self.lower_play
            elif relative_difference >= 0:
                return wanted_difference

        return wanted_difference

    def convert(self, instrument_angles, current_positions):
        shaft_roll, bend, tip_rotation, articulation = (
            float(value) for value in instrument_angles[:4]
        )
        bend = float(np.clip(bend, -math.pi / 4.0, math.pi / 4.0))
        articulation = float(np.clip(articulation, 0.0, math.pi / 6.0))

        delta_shaft_roll = shaft_roll - self.absolute_shaft_roll
        difference = math.fmod(delta_shaft_roll, 2.0 * math.pi)
        if difference < -math.pi:
            difference += 2.0 * math.pi
        elif difference >= math.pi:
            difference -= 2.0 * math.pi
        self.absolute_shaft_roll += difference

        shaft_pulses = int(
            self.absolute_shaft_roll
            * self.pulses_per_rotation[1]
            / (2.0 * math.pi)
        )
        bend_pulses = self.get_motor2_value_for_angle(
            bend, current_positions
        )
        tip_pulses = -cpp_round(
            tip_rotation
            * self.pulses_per_rotation[0]
            / (2.0 * math.pi)
        )
        articulation_pulses = cpp_round(
            self.articulation_factor
            * articulation
            * self.pulses_per_rotation[3]
            / (2.0 * math.pi)
        )

        return np.array([
            self.starting_positions[0] + tip_pulses,
            self.starting_positions[1] + shaft_pulses - self.bend_play_compensation,
            self.starting_positions[2] + shaft_pulses + bend_pulses + self.bend_play_compensation,
            self.starting_positions[3] + tip_pulses + articulation_pulses,
        ], dtype=float)


def reconstruct_instrument_targets(rows, converter):
    if not rows:
        return
    reconstructed_count = 0

    for row in rows:
        if row["command_source"] == "motor_absolute":
            row["command_target"] = np.asarray(
                row["command_values"], dtype=float
            ).tolist()
        elif row["command_source"] == "instrument_angles":
            row["command_target"] = converter.convert(
                instrument_angles=row["command_values"],
                current_positions=row["positions"],
            ).tolist()
            reconstructed_count += 1
        else:
            raise RuntimeError(
                f"Cannot reconstruct command source: {row['command_source']}"
            )

    if reconstructed_count:
        print(
            f"Reconstructed {reconstructed_count} absolute motor targets "
            "from commanded_instrument_angles."
        )


def load_motor_log(file_path, coupling_mode, command_converter=None):
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
            motor_commands = data.get("commanded_motor_positions")
            motor_targets = data.get("motor_target_positions")
            instrument_commands = data.get("commanded_instrument_angles")

            if t_value is None or positions is None or currents is None:
                continue

            if len(positions) < 4 or len(currents) < 4:
                continue

            motor_commands_valid = (
                motor_commands is not None and len(motor_commands) >= 4
            )
            motor_targets_valid = (
                motor_targets is not None and len(motor_targets) >= 4
            )
            instrument_commands_valid = (
                instrument_commands is not None and len(instrument_commands) >= 4
            )

            if coupling_mode == "motor_only":
                if not motor_commands_valid:
                    # A true idle-baseline segment intentionally has no motor
                    # step. Preserve it as a zero command, but never invent
                    # zero commands for a moving trial.
                    test_type = parse_task_label(current_task_label)["test_type"]
                    if test_type != "idle_baseline":
                        continue
                    motor_commands = [0.0, 0.0, 0.0, 0.0]
                command_source = "motor_relative"
                command_values = motor_commands[:4]
            # elif instrument_commands_valid:
            #     # For both coupled configurations, use the command that was
            #     # actually logged. Reconstructing motor targets would require
            #     # controller/backlash states that are absent from the logs.
            #     command_source = "instrument_angles"
            #     command_values = instrument_commands[:4]
            # else:
            #     continue
            elif motor_targets_valid and instrument_commands_valid:
                # For coupled configurations, the Motor DT uses the absolute
                # targets that were actually sent to the motor controller.
                # Instrument commands are retained for DOF-pattern segmentation
                # and visualization only.
                command_source = "motor_absolute"
                command_values = motor_targets[:4]
            else:
                continue

            rows.append({
                "time": float(t_value),
                "task_label": current_task_label,
                "positions": [float(x) for x in positions[:4]],
                "currents": [float(x) for x in currents[:4]],
                "command_source": command_source,
                "command_values": [float(x) for x in command_values],
                "motor_targets": (
                    [float(x) for x in motor_targets[:4]]
                    if motor_targets_valid
                    else None
                ),
                # Kept separately for explicit coupled-configuration features.
                "instrument_commands": (
                    [float(x) for x in instrument_commands[:4]]
                    if instrument_commands_valid
                    else None
                ),
            })

    if not rows:
        raise RuntimeError(f"No usable motor samples found in {file_path}")

    t0 = rows[0]["time"]

    for row in rows:
        row["time"] -= t0

    return rows

def split_motor_pattern_segments(rows):
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

def parse_number_token(value):
    """
    Convert label notation such as:
        0p5 -> 0.5
        1p0 -> 1.0
        m0p5 -> -0.5
    """
    value = str(value)

    negative = value.startswith("m")
    if negative:
        value = value[1:]

    number = float(value.replace("p", "."))

    return -number if negative else number


def parse_dof_sequence_label(task_label):
    """
    Extract DOF sequence information from labels such as:

    continuous_dof2_sequence_sinusoid_triangle_
    freqx0p5_1p0_2p0_rangex0p5
    """
    if not task_label:
        return None

    label = str(task_label).lower()

    dof_match = re.search(r"dof[\s_-]*([1-4])", label)

    freq_match = re.search(
        r"freqx([0-9mp_]+?)(?=_rangex|$)",
        label,
    )

    range_match = re.search(
        r"rangex([0-9mp_]+?)(?:\||$)",
        label,
    )

    if dof_match is None or freq_match is None or range_match is None:
        return None

    dof_number = int(dof_match.group(1))

    frequency_factors = [
        parse_number_token(token)
        for token in freq_match.group(1).split("_")
        if token
    ]

    range_factors = [
        parse_number_token(token)
        for token in range_match.group(1).split("_")
        if token
    ]

    sequence_part_match = re.search(
        r"sequence_(.+?)_freqx",
        label,
    )

    if sequence_part_match is None:
        return None

    sequence_part = sequence_part_match.group(1)

    modes = []

    mode_pattern = re.compile(
        r"(sinusoid_mid|triangle_mid|sinusoid|triangle)"
    )

    for match in mode_pattern.finditer(sequence_part):
        mode_name = match.group(1)

        if mode_name.startswith("sinusoid"):
            modes.append("sinusoid")

        elif mode_name.startswith("triangle"):
            modes.append("triangle")

    if not modes:
        return None

    return {
        "dof_number": dof_number,
        "frequency_factors": frequency_factors,
        "range_factors": range_factors,
        "modes": modes,
    }

def split_dof_pattern_segments(
    rows,
    command_tolerance=1e-4,
    minimum_pause_s=1.0,
):
    """
    Split one continuous DOF sequence into individual waveform segments.

    Output segment definition:
        one DOF
        + one range factor
        + one frequency factor
        + one waveform type

    Example:
        DOF2 / range 0.5 / freq 0.5 / sinusoid
        DOF2 / range 0.5 / freq 0.5 / triangle
        DOF2 / range 0.5 / freq 1.0 / sinusoid
        DOF2 / range 0.5 / freq 1.0 / triangle

    Sinusoid and triangle remain separate segments here.
    The plotter can later combine segments with equal
    DOF + range + frequency into one comparison figure.
    """

    if not rows:
        return []

    # ----------------------------------------------------------
    # 1. Remove irrelevant/unlabelled rows
    # ----------------------------------------------------------

    valid_rows = [
        row
        for row in rows
        if row.get("task_label") not in [
            None,
            "unlabeled",
            "between_trials",
            "motor_tests_finished",
            "tests_finished",
        ]
    ]

    if not valid_rows:
        return []

    # A DOF log should contain one continuous sequence label.
    task_label = valid_rows[0]["task_label"]
    info = parse_task_label(task_label)

    sequence_info = parse_dof_sequence_label(task_label)

    if sequence_info is None:
        raise RuntimeError(
            "Could not parse DOF sequence information from task label:\n"
            f"{task_label}"
        )

    dof_number = sequence_info["dof_number"]
    dof_index = dof_number - 1

    frequency_factors = sequence_info["frequency_factors"]
    range_factors = sequence_info["range_factors"]
    modes = sequence_info["modes"]

    # ----------------------------------------------------------
    # 2. Get active DOF command
    # ----------------------------------------------------------

    t = np.asarray(
        [row["time"] for row in valid_rows],
        dtype=float,
    )

    command = np.asarray(
        [
            row["instrument_commands"][dof_index]
            for row in valid_rows
        ],
        dtype=float,
    )

    if len(t) < 3:
        return []

    t = t - t[0]

    finite = (
        np.isfinite(t)
        & np.isfinite(command)
    )

    if np.sum(finite) < 3:
        return []

    # ----------------------------------------------------------
    # 3. Determine neutral/baseline command
    # ----------------------------------------------------------

    # The sequence starts with a stationary hold.
    # Median of the first second is more robust than using one sample.
    initial_mask = (
        finite
        & (t <= min(1.0, float(t[-1])))
    )

    if np.any(initial_mask):
        baseline = float(
            np.median(command[initial_mask])
        )
    else:
        baseline = float(command[np.flatnonzero(finite)[0]])

    moving = (
        finite
        & (np.abs(command - baseline) > command_tolerance)
    )

    # ----------------------------------------------------------
    # 4. Find sustained neutral pauses
    # ----------------------------------------------------------

    # A sinusoid/triangle naturally crosses zero.
    # Therefore a single zero sample must NOT split a segment.
    #
    # Only a neutral interval lasting >= minimum_pause_s
    # counts as a true stage pause.

    pause_mask = finite & ~moving

    pause_intervals = []

    pause_start = None

    for index, is_pause in enumerate(pause_mask):
        if is_pause and pause_start is None:
            pause_start = index

        is_last = index == len(pause_mask) - 1

        if pause_start is not None and (
            (not is_pause) or is_last
        ):
            pause_end = (
                index
                if is_pause and is_last
                else index - 1
            )

            duration = (
                t[pause_end]
                - t[pause_start]
            )

            if duration >= minimum_pause_s:
                pause_intervals.append(
                    (pause_start, pause_end)
                )

            pause_start = None

    # ----------------------------------------------------------
    # 5. Convert pauses into motion blocks
    # ----------------------------------------------------------

    motion_blocks = []

    previous_pause_end = -1

    for pause_start, pause_end in pause_intervals:
        block_start = previous_pause_end + 1
        block_end = pause_start - 1

        block_moving = moving[
            block_start:block_end + 1
        ]

        if (
            block_end >= block_start
            and np.any(block_moving)
        ):
            first_motion = (
                block_start
                + np.flatnonzero(block_moving)[0]
            )

            # Include the following neutral hold in the same pattern segment.
            segment_end = pause_end

            motion_blocks.append(
                (first_motion, segment_end)
            )

        previous_pause_end = pause_end

    # Possible final block after final detected pause.
    block_start = previous_pause_end + 1
    block_end = len(valid_rows) - 1

    if block_end >= block_start:
        block_moving = moving[
            block_start:block_end + 1
        ]

        if np.any(block_moving):
            first_motion = (
                block_start
                + np.flatnonzero(block_moving)[0]
            )

            last_motion = (
                block_start
                + np.flatnonzero(block_moving)[-1]
            )

            motion_blocks.append(
                (first_motion, last_motion)
            )

    # ----------------------------------------------------------
    # 6. Build expected sequence order
    # ----------------------------------------------------------

    expected_patterns = []

    for range_factor in range_factors:
        for frequency_factor in frequency_factors:
            for mode in modes:
                expected_patterns.append({
                    "range_factor": range_factor,
                    "frequency_factor": frequency_factor,
                    "pattern_type": mode,
                })

    if len(motion_blocks) != len(expected_patterns):
        raise RuntimeError(
            "DOF pattern split does not match the sequence label.\n"
            f"Detected motion blocks: {len(motion_blocks)}\n"
            f"Expected blocks:        {len(expected_patterns)}\n"
            f"DOF:                    {dof_number}\n"
            f"Frequencies:            {frequency_factors}\n"
            f"Ranges:                 {range_factors}\n"
            f"Modes:                  {modes}\n"
            f"Task label:             {task_label}"
        )

    # ----------------------------------------------------------
    # 7. Create final DOF pattern segments
    # ----------------------------------------------------------

    segments = []

    for block_index, (
        motion_block,
        pattern_info,
    ) in enumerate(
        zip(motion_blocks, expected_patterns)
    ):
        start_index, end_index = motion_block

        segment_rows = valid_rows[
            start_index:end_index + 1
        ]

        if len(segment_rows) < 3:
            continue

        segment = {
            "task_label": task_label,
            "rows": segment_rows,
            "run_rows": valid_rows,
            "run_start_index": start_index,
            "run_end_index": end_index,

            "info": dict(info),

            "dof_number": dof_number,
            "dof_index": dof_index,

            "frequency_factor": (
                pattern_info["frequency_factor"]
            ),

            "range_factor": (
                pattern_info["range_factor"]
            ),

            "pattern_type": (
                pattern_info["pattern_type"]
            ),

            "pattern_index": block_index,
        }

        segments.append(segment)

    return segments

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

    if len(t) > 0:
        t = t - t[0]

    command_sources = {row["command_source"] for row in rows}

    if command_sources == {"motor_relative"}:
        commands = np.array(
            [row["command_values"] for row in rows], dtype=float
        ).T
        commanded_target = relative_commands_to_target(commands)
    elif command_sources.issubset({"motor_absolute", "instrument_angles"}):
        commanded_target = np.array(
            [row["command_target"] for row in rows], dtype=float
        ).T
        commands = np.zeros_like(commanded_target)
        commands[:, 1:] = np.diff(commanded_target, axis=1)
    else:
        raise RuntimeError(
            f"Mixed or unsupported command sources in segment: {command_sources}"
        )

    for motor_index in range(4):
        if positions.shape[1] > 0:
            positions[motor_index] = positions[motor_index] - positions[motor_index][0]
            commanded_target[motor_index] = (
                commanded_target[motor_index] - commanded_target[motor_index][0]
            )

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


def held_command_delta(signal):
    """Hold the last non-zero command change, resetting at segment start."""
    signal = np.asarray(signal, dtype=float)
    instantaneous_delta = np.zeros_like(signal)
    if len(signal) > 1:
        instantaneous_delta[1:] = np.diff(signal)

    held_delta = np.zeros_like(signal)
    last_delta = 0.0
    for sample_index, delta in enumerate(instantaneous_delta):
        if abs(delta) > 1e-12:
            last_delta = delta
        held_delta[sample_index] = last_delta

    return instantaneous_delta, held_delta

def instrument_commands_from_segment(segment):
    """Return all four logged instrument command channels as shape (4, N)."""
    commands = []
    for row in segment["rows"]:
        values = row.get("instrument_commands")
        if values is None or len(values) < 4:
            raise RuntimeError(
                "Coupled Motor-DT training requires all four "
                "commanded_instrument_angles in "
                f"{segment.get('source_file', 'unknown file')}."
            )
        commands.append([float(value) for value in values[:4]])
    return np.asarray(commands, dtype=float).T


def coupled_motor_target_feature_names():
    names = []

    for motor_index in range(4):
        prefix = f"m{motor_index}"

        names.extend([
            f"{prefix}_target_relative_pulses",
            f"{prefix}_previous_target_relative_pulses",
            f"{prefix}_target_delta_pulses",
            f"abs_{prefix}_target_delta_pulses",
            f"{prefix}_target_direction",
            f"{prefix}_target_velocity_pulses_per_s",
            f"time_since_{prefix}_target_change_s",
            f"{prefix}_target_is_moving",
        ])

    return names

def build_coupled_samples_from_segment(segment, motor_index):
    """
    Build gearbox/full-setup Motor-DT samples from logged motor targets.

    All four logged motor_target_positions channels are used as input
    features. Targets and encoder positions are expressed relative to
    the start of each individual motion pattern.

    Commanded instrument angles are retained only for DOF segmentation
    and plotting.
    """

    # ----------------------------------------------------------
    # 1. Full-run context
    # ----------------------------------------------------------

    run_rows = segment["run_rows"]
    start_index = int(segment["run_start_index"])
    end_index = int(segment["run_end_index"])

    if len(run_rows) < 3:
        return None

    if start_index < 0 or end_index >= len(run_rows):
        raise RuntimeError(
            f"Invalid pattern indices: {start_index}:{end_index} "
            f"for run with {len(run_rows)} rows."
        )

    if end_index < start_index:
        return None

    # ----------------------------------------------------------
    # 2. Full-run arrays
    # ----------------------------------------------------------

    t_run = np.asarray(
        [row["time"] for row in run_rows],
        dtype=float,
    )
    t_run = t_run - t_run[0]

    positions_run = np.asarray(
        [row["positions"] for row in run_rows],
        dtype=float,
    ).T

    currents_run = np.asarray(
        [row["currents"] for row in run_rows],
        dtype=float,
    ).T

    # ----------------------------------------------------------
    # 3. Full-run motor targets and instrument commands
    # ----------------------------------------------------------

    motor_target_rows = []
    instrument_command_rows = []

    for row in run_rows:
        motor_values = row.get("motor_targets")

        if motor_values is None or len(motor_values) < 4:
            raise RuntimeError(
                "Coupled Motor-DT training requires all four "
                "motor_target_positions in "
                f"{segment.get('source_file', 'unknown file')}."
            )

        instrument_values = row.get("instrument_commands")

        if instrument_values is None or len(instrument_values) < 4:
            raise RuntimeError(
                "DOF segmentation and plotting require all four "
                "commanded_instrument_angles in "
                f"{segment.get('source_file', 'unknown file')}."
            )

        motor_target_rows.append(
            [float(value) for value in motor_values[:4]]
        )
        instrument_command_rows.append(
            [float(value) for value in instrument_values[:4]]
        )

    motor_targets_absolute_run = np.asarray(
        motor_target_rows,
        dtype=float,
    ).T

    instrument_commands_run = np.asarray(
        instrument_command_rows,
        dtype=float,
    ).T

    # Pattern-start reference used later to express the absolute
    # target channels relative to the start of this motion pattern.
    motor_target_reference = motor_targets_absolute_run[
        :,
        start_index,
    ].copy()
    # ----------------------------------------------------------
    # 4. Current preprocessing over complete run
    # ----------------------------------------------------------

    raw_current_run = currents_run[motor_index]

    filtered_current_run = causal_moving_average(
        raw_current_run,
        window_size=5,
    )

    # ----------------------------------------------------------
    # 5. Command features
    # ----------------------------------------------------------

    feature_columns = []
    motor_moving_masks = []

    for target_motor_index in range(4):
        absolute_target = motor_targets_absolute_run[target_motor_index]

        relative_target = (
            absolute_target
            - motor_target_reference[target_motor_index]
        )

        previous_relative_target = relative_target.copy()

        if len(relative_target) > 1:
            previous_relative_target[1:] = relative_target[:-1]

        instantaneous_delta, held_delta = held_command_delta(
            absolute_target
        )

        direction = np.sign(
            held_delta
        )

        velocity = safe_gradient(
            absolute_target,
            t_run,
        )

        time_after_change = time_since_signal_change(
            t_run,
            absolute_target,
        )

        moving = (
            (np.abs(held_delta) > 1e-12)
            & (time_after_change < EVENT_TAIL_S)
        ).astype(float)

        motor_moving_masks.append(
            moving.astype(bool)
        )

        feature_columns.extend([
            relative_target,
            previous_relative_target,
            instantaneous_delta,
            np.abs(instantaneous_delta),
            direction,
            velocity,
            time_after_change,
            moving,
        ])

    X_command_run = np.column_stack(
        feature_columns
    )

    dynamic_mask_run = np.logical_or.reduce(
        motor_moving_masks
    )

    # ----------------------------------------------------------
    # 6. Extract current sinusoid/triangle pattern
    # ----------------------------------------------------------

    pattern_slice = slice(
        start_index,
        end_index + 1,
    )

    X_command = X_command_run[
        pattern_slice
    ]

    motor_targets = (
        motor_targets_absolute_run[
            :,
            pattern_slice,
        ]
        - motor_target_reference[:, np.newaxis]
    )
    instrument_commands = instrument_commands_run[
        :,
        pattern_slice,
    ]

    raw_current = raw_current_run[
        pattern_slice
    ]

    filtered_current = filtered_current_run[
        pattern_slice
    ]

    dynamic_mask = dynamic_mask_run[
        pattern_slice
    ]

    # Pattern-local time.
    t = t_run[
        pattern_slice
    ].copy()

    if len(t) > 0:
        t = t - t[0]

    # ----------------------------------------------------------
    # 7. PATTERN-RELATIVE ENCODER TARGET
    # ----------------------------------------------------------

    encoder_absolute = positions_run[
        motor_index,
        pattern_slice,
    ].copy()

    if len(encoder_absolute) < 3:
        return None

    # Use the measured encoder position at the start of the
    # actual motion pattern as the local zero reference.
    #
    # Example:
    #   run A:   0 -> 83   becomes 0 -> 83
    #   run B: -60 -> 23   becomes 0 -> 83
    #   run C: -82 ->  1   becomes 0 -> 83
    #
    # The Motor DT therefore learns the displacement caused by
    # the commanded motion, independent of the mechanical offset
    # at which the pattern happened to start.
    encoder_reference = float(
        encoder_absolute[0]
    )

    encoder = (
        encoder_absolute
        - encoder_reference
    )

    # Encoder velocity is calculated from the relative pattern response.
    encoder_velocity = safe_gradient(
        encoder,
        t,
    )

    stationary_mask = (
        (~dynamic_mask)
        & (
            np.abs(encoder_velocity)
            <= STATIONARY_ENCODER_VELOCITY_THRESHOLD
        )
    )

    # ----------------------------------------------------------
    # 8. Active DOF
    # ----------------------------------------------------------

    protocol = segment.get(
        "split_group",
        "unknown_protocol",
    )

    match = re.fullmatch(
        r"dof_(\d+)",
        protocol,
    )

    active_dof_index = (
        int(match.group(1)) - 1
        if match
        else None
    )

    active_command = (
        instrument_commands[active_dof_index]
        if (
            active_dof_index is not None
            and 0 <= active_dof_index < 4
        )
        else None
    )

    # ----------------------------------------------------------
    # 9. Metadata
    # ----------------------------------------------------------

    metadata = {
        "test_type": segment["info"]["test_type"],
        "motor_name": segment["info"]["motor_name"],
        "trial": segment["info"]["trial"],
        "motor_index": motor_index,

        "source_file": segment["source_file"],
        "protocol": protocol,
        "run_name": Path(
            segment["source_file"]
        ).stem,

        "dataset_split": segment["dataset_split"],

        "time": t,

        "target":
            motor_targets[motor_index],

        "model_command_source": (
            "motor_target_positions"
        ),

        "encoder_reference": (
            "relative_to_motion_pattern_start"
        ),

        "motor_target_reference": (
            "relative_to_motion_pattern_start"
        ),

        "motor_target_reference_pulses":
            motor_target_reference,

        "motor_target_positions_relative":
            motor_targets,

        "instrument_commands_rad":
            instrument_commands,

        "active_dof_index":
            active_dof_index,

        "active_instrument_command_rad":
            active_command,

        "dof_number":
            segment["dof_number"],

        "dof_index":
            segment["dof_index"],

        "frequency_factor":
            segment["frequency_factor"],

        "range_factor":
            segment["range_factor"],

        "pattern_type":
            segment["pattern_type"],

        "pattern_index":
            segment["pattern_index"],

        "raw_command": None,

        # This is now displacement from the start of the pattern.
        "encoder":
            encoder,

        "current":
            filtered_current,

        "raw_current":
            raw_current,

        "filtered_current":
            filtered_current,

        "encoder_velocity":
            encoder_velocity,

        "encoder_error":
            motor_targets[motor_index] - encoder,

        "is_moving":
            dynamic_mask.astype(float),

        "dynamic_mask":
            dynamic_mask,

        "stationary_mask":
            stationary_mask,
    }

    # Encoder model and current model use the same command features,
    # but have different targets:
    #
    # encoder target = relative displacement [pulses]
    # current target = filtered current [mA]
    return (
        X_command,
        X_command.copy(),
        encoder,
        filtered_current,
        metadata,
    )

def build_measured_phase_masks(
    t,
    target,
    encoder,
    encoder_velocity,
):
    dynamic_mask = np.zeros(len(t), dtype=bool)

    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0.0]
    median_dt = (
        np.median(positive_dt)
        if len(positive_dt)
        else 0.01
    )

    hold_samples = max(
        1,
        int(np.ceil(
            DYNAMIC_COMPLETION_HOLD_S / median_dt
        )),
    )

    target_changes = np.flatnonzero(
        np.r_[
            False,
            np.abs(np.diff(target))
            > MOTOR_COMMAND_TOLERANCE_PULSES,
        ]
    )

    for event_index, start_index in enumerate(target_changes):
        next_command_index = (
            target_changes[event_index + 1]
            if event_index + 1 < len(target_changes)
            else len(t)
        )

        target_step = (
            target[start_index]
            - target[start_index - 1]
        )

        position_tolerance = max(
            DYNAMIC_MIN_POSITION_TOLERANCE_PULSES,
            DYNAMIC_REL_POSITION_TOLERANCE
            * abs(target_step),
        )

        settled = (
            (
                np.abs(
                    encoder[start_index:next_command_index]
                    - target[start_index]
                )
                <= position_tolerance
            )
            & (
                np.abs(
                    encoder_velocity[
                        start_index:next_command_index
                    ]
                )
                <= STATIONARY_ENCODER_VELOCITY_THRESHOLD
            )
        )

        completion_index = None

        for local_index in range(
            0,
            len(settled) - hold_samples + 1,
        ):
            if np.all(
                settled[
                    local_index:
                    local_index + hold_samples
                ]
            ):
                completion_index = (
                    start_index + local_index
                )
                break

        dynamic_end_index = (
            completion_index
            if completion_index is not None
            else next_command_index
        )

        dynamic_mask[
            start_index:dynamic_end_index
        ] = True

    stationary_mask = (
        (~dynamic_mask)
        & (
            np.abs(encoder_velocity)
            <= STATIONARY_ENCODER_VELOCITY_THRESHOLD
        )
    )

    return dynamic_mask, stationary_mask

def build_samples_from_segment(segment, motor_index, coupling_mode):
    if coupling_mode != "motor_only":
        return build_coupled_samples_from_segment(segment, motor_index)

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
    movement_memory_s = EVENT_TAIL_S
    target_based_is_moving = (
        (np.abs(target_delta) > 0.0)
        & (time_after_change < movement_memory_s)
    ).astype(float)

    encoder_velocity = safe_gradient(encoder, t)
    encoder_acceleration = safe_gradient(encoder_velocity, t)

    # Evaluation masks for held-out healthy data.
    # Dynamic starts at a target change and ends when the measured
    # encoder remains close to the target and stationary for 0.1 s.
    # These masks affect metrics and model selection only;
    # they do not filter the training samples or change model inputs.
    dynamic_mask, stationary_mask = (
        build_measured_phase_masks(
            t=t,
            target=target,
            encoder=encoder,
            encoder_velocity=encoder_velocity,
        )
    )

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
        np.abs(target),
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
        np.abs(target),
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
        "source_file": segment["source_file"],
        "protocol": segment.get("split_group", "unknown_protocol"),
        "run_name": Path(segment["source_file"]).stem,
        "dataset_split": segment["dataset_split"],
        "time": t,
        "target": target,
        "model_command_source": "reconstructed_motor_target",
        "raw_command": raw_command,
        "encoder": encoder,
        "current": current,
        "raw_current": raw_current,
        "filtered_current": filtered_current,
        "encoder_velocity": encoder_velocity,
        "encoder_error": encoder_error,
        "is_moving": is_moving,
        "dynamic_mask": dynamic_mask,
        "stationary_mask": stationary_mask,
    }

    return X_encoder, X_current, y_encoder, y_current, metadata

def build_dataset(segments, coupling_mode):
    samples_by_motor = {0: [], 1: [], 2: [], 3: []}

    for segment in segments:
        info = segment["info"]
        active_motor = get_active_motor_index(segment)

        if coupling_mode == "motor_only":
            if info["test_type"] == "idle_baseline":
                motor_indices = range(4)
            elif active_motor is not None:
                motor_indices = [active_motor]
            else:
                continue
        else:
            # Every coupled motor model receives every DOF trial. This lets it
            # learn both direct actuation and the absence/presence of coupling
            # from the same four commanded instrument-angle channels.
            motor_indices = range(4)

        for motor_index in motor_indices:
            sample = build_samples_from_segment(
                segment,
                motor_index,
                coupling_mode=coupling_mode,
            )

            if sample is None:
                continue

            samples_by_motor[motor_index].append(sample)

    return samples_by_motor


def split_train_test(samples):
    train_samples = []
    test_samples = []

    for sample in samples:
        metadata = sample[4]

        if metadata["dataset_split"] == "test":
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    if not train_samples:
        raise RuntimeError("No training samples available.")

    if not test_samples:
        raise RuntimeError("No test samples available.")

    return train_samples, test_samples


def get_split_group(segment, coupling_mode):
    """Return the experimental protocol whose repetitions form one split."""
    if coupling_mode == "motor_only":
        return "motor_only_all_runs"

    info = segment["info"]
    source_file = Path(segment["source_file"])
    description = "|".join([
        str(info.get("test_type", "")),
        str(info.get("motor_name", "")),
        source_file.stem,
        *source_file.parts,
    ]).lower()

    dof_match = re.search(r"dof[\s_-]*([1-4])", description)
    if dof_match:
        return f"dof_{dof_match.group(1)}"

    return (
        f"{info.get('test_type', 'unknown_test')}|"
        f"{info.get('motor_name', 'unknown_motor')}"
    )


def assign_complete_run_splits(segments, coupling_mode):
    """
    Hold out one complete log per experimental protocol.

    Gearbox/full-setup logs are DOF-specific, so one global test file can leave
    some motors without any test data. Selecting one repetition per DOF keeps
    the split independent at file level while covering every relevant motor.
    """
    files_by_group = {}

    for segment in segments:
        group = get_split_group(segment, coupling_mode)
        source_file = Path(segment["source_file"]).resolve()
        files_by_group.setdefault(group, set()).add(source_file)

    rng = np.random.default_rng(RUN_SPLIT_SEED)
    test_file_by_group = {}

    print("\nComplete-run train/test split per protocol:")

    for group in sorted(files_by_group):
        group_files = sorted(files_by_group[group])

        if len(group_files) < 2:
            raise RuntimeError(
                f"Protocol '{group}' has only {len(group_files)} complete run(s). "
                "At least two are required so one can be held out for testing."
            )

        test_index = int(rng.integers(0, len(group_files)))
        test_file = group_files[test_index]
        test_file_by_group[group] = test_file

        print(f"  {group}")
        print(f"    Test run: {test_file.name}")
        print(f"    Training runs: {len(group_files) - 1}")

    for segment in segments:
        group = get_split_group(segment, coupling_mode)
        source_file = Path(segment["source_file"]).resolve()
        segment["split_group"] = group
        segment["dataset_split"] = (
            "test" if source_file == test_file_by_group[group] else "train"
        )

    return test_file_by_group

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


def stack_evaluation_masks(samples):
    dynamic_mask = np.concatenate([
        np.asarray(sample[4]["dynamic_mask"], dtype=bool)
        for sample in samples
    ])
    stationary_mask = np.concatenate([
        np.asarray(sample[4]["stationary_mask"], dtype=bool)
        for sample in samples
    ])
    return dynamic_mask, stationary_mask


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "max_abs_error": float(max_error),
        "n_samples": int(len(y_true)),
    }

def compute_phase_metrics(y_true, y_pred, dynamic_mask, stationary_mask):
    phase_metrics = {
        "overall": compute_metrics(y_true, y_pred),
    }

    if np.any(dynamic_mask):
        phase_metrics["dynamic"] = compute_metrics(
            y_true[dynamic_mask],
            y_pred[dynamic_mask],
        )

    if np.any(stationary_mask):
        phase_metrics["stationary"] = compute_metrics(
            y_true[stationary_mask],
            y_pred[stationary_mask],
        )

    return phase_metrics


def add_target_range_normalization(metrics, target_range_pulses):
    """Add encoder errors as percentages of one full motor-target excursion."""
    normalized = dict(metrics)
    normalized["target_range_pulses"] = (
        float(target_range_pulses)
        if np.isfinite(target_range_pulses)
        else None
    )

    if not np.isfinite(target_range_pulses) or target_range_pulses <= 1.0:
        normalized["nmae_percent_of_target_range"] = None
        normalized["nrmse_percent_of_target_range"] = None
        normalized["max_abs_error_percent_of_target_range"] = None
        return normalized

    scale = 100.0 / float(target_range_pulses)
    normalized["nmae_percent_of_target_range"] = float(
        metrics["mae"] * scale
    )
    normalized["nrmse_percent_of_target_range"] = float(
        metrics["rmse"] * scale
    )
    normalized["max_abs_error_percent_of_target_range"] = float(
        metrics["max_abs_error"] * scale
    )
    return normalized

def add_protocol_target_range_normalization(
    metrics,
    target_range_pulses,
):
    """Normalize aggregated protocol errors using the full held-out target range."""
    normalized = dict(metrics)

    active_motor = (
        target_range_pulses is not None
        and np.isfinite(target_range_pulses)
        and target_range_pulses > 0.0
    )

    normalized["motor_target_range_pulses"] = (
        float(target_range_pulses)
        if target_range_pulses is not None
        and np.isfinite(target_range_pulses)
        else None
    )
    normalized["active_motor_in_protocol"] = bool(active_motor)

    if not active_motor:
        normalized["nmae_percent_of_target_range"] = None
        normalized["nrmse_percent_of_target_range"] = None
        normalized["max_abs_error_percent_of_target_range"] = None
        return normalized

    scale = 100.0 / float(target_range_pulses)

    normalized["nmae_percent_of_target_range"] = float(
        metrics["mae"] * scale
    )
    normalized["nrmse_percent_of_target_range"] = float(
        metrics["rmse"] * scale
    )
    normalized["max_abs_error_percent_of_target_range"] = float(
        metrics["max_abs_error"] * scale
    )

    return normalized

def compute_range_normalized_phase_metrics(
    y_true,
    y_pred,
    motor_target,
    dynamic_mask,
    stationary_mask,
):
    """Normalize every phase with the full range of its own test segment."""
    motor_target = np.asarray(motor_target, dtype=float)
    finite_target = motor_target[np.isfinite(motor_target)]

    target_range_pulses = (
        float(np.max(finite_target) - np.min(finite_target))
        if finite_target.size >= 2
        else np.nan
    )

    by_phase = {
        "overall": add_target_range_normalization(
            compute_metrics(y_true, y_pred),
            target_range_pulses,
        ),
    }

    if np.any(dynamic_mask):
        by_phase["dynamic"] = add_target_range_normalization(
            compute_metrics(y_true[dynamic_mask], y_pred[dynamic_mask]),
            target_range_pulses,
        )

    if np.any(stationary_mask):
        by_phase["stationary"] = add_target_range_normalization(
            compute_metrics(
                y_true[stationary_mask],
                y_pred[stationary_mask],
            ),
            target_range_pulses,
        )

    return by_phase


def summarize_range_normalized_segments(segment_results):
    """Summarize segment percentages without mixing their pulse ranges."""
    summary = {}
    percent_fields = (
        "nmae_percent_of_target_range",
        "nrmse_percent_of_target_range",
        "max_abs_error_percent_of_target_range",
    )

    for phase in ("overall", "dynamic", "stationary"):
        phase_summary = {}
        phase_rows = [
            segment["by_phase"][phase]
            for segment in segment_results
            if phase in segment["by_phase"]
        ]

        for field in percent_fields:
            values = np.asarray([
                row[field]
                for row in phase_rows
                if row.get(field) is not None
            ], dtype=float)

            if values.size == 0:
                continue

            phase_summary[field] = {
                "mean": float(np.mean(values)),
                "sd": float(
                    np.std(values, ddof=1)
                    if values.size > 1
                    else 0.0
                ),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_segments": int(values.size),
            }

        if phase_summary:
            summary[phase] = phase_summary

    return summary


def compute_current_zone_metrics(
    y_true,
    y_pred,
    dynamic_mask,
    stationary_mask,
):
    metrics = {}

    # Peak-zone: hoogste 5% van gemeten current
    peak_threshold = np.nanpercentile(y_true, 95)
    peak_mask = y_true >= peak_threshold

    if np.any(stationary_mask):
        metrics["baseline_current_mA"] = compute_metrics(
            y_true[stationary_mask],
            y_pred[stationary_mask],
        )

    if np.any(dynamic_mask):
        metrics["moving_current_mA"] = compute_metrics(
            y_true[dynamic_mask],
            y_pred[dynamic_mask],
        )

    if np.any(peak_mask):
        metrics["peak_current_mA"] = compute_metrics(
            y_true[peak_mask],
            y_pred[peak_mask],
        )

    metrics["peak_threshold_mA"] = float(peak_threshold)

    return metrics


def compute_metrics_by_protocol(samples, prediction, response_kind):
    """Evaluate complete held-out runs without merging different DOFs."""
    prediction = np.asarray(prediction, dtype=float)
    grouped = {}
    offset = 0

    for sample in samples:
        _, _, y_encoder, y_current, metadata = sample
        y_true = y_encoder if response_kind == "encoder" else y_current
        length = len(y_true)
        y_pred = prediction[offset:offset + length]
        offset += length

        if len(y_pred) != length:
            raise RuntimeError("Prediction length does not match held-out samples.")

        protocol = metadata["protocol"]
        entry = grouped.setdefault(protocol, {
            "y_true": [],
            "y_pred": [],
            "dynamic_mask": [],
            "stationary_mask": [],
            "source_runs": set(),
            "motor_target": [],
            "range_normalized_segments": [],
        })
        entry["y_true"].append(np.asarray(y_true, dtype=float))
        entry["y_pred"].append(np.asarray(y_pred, dtype=float))
        entry["dynamic_mask"].append(
            np.asarray(metadata["dynamic_mask"], dtype=bool)
        )
        entry["stationary_mask"].append(
            np.asarray(metadata["stationary_mask"], dtype=bool)
        )
        entry["source_runs"].add(metadata["run_name"])

        if response_kind == "encoder":
            motor_target = np.asarray(metadata["target"], dtype=float)
            
            if motor_target.size != length:
                raise RuntimeError(
                    "Motor target length does not match held-out encoder sample."
                )
            entry["motor_target"].append(motor_target)

            entry["range_normalized_segments"].append({
                "source_run": metadata["run_name"],
                "trial": metadata.get("trial"),
                "pattern_type": metadata.get("pattern_type"),
                "frequency_factor": metadata.get("frequency_factor"),
                "range_factor": metadata.get("range_factor"),
                "by_phase": compute_range_normalized_phase_metrics(
                    np.asarray(y_true, dtype=float),
                    np.asarray(y_pred, dtype=float),
                    motor_target,
                    np.asarray(metadata["dynamic_mask"], dtype=bool),
                    np.asarray(metadata["stationary_mask"], dtype=bool),
                ),
            })

    if offset != len(prediction):
        raise RuntimeError("Unused predictions remain after per-DOF evaluation.")

    results = {}
    for protocol, entry in grouped.items():
        y_true = np.concatenate(entry["y_true"])
        y_pred = np.concatenate(entry["y_pred"])
        dynamic_mask = np.concatenate(entry["dynamic_mask"])
        stationary_mask = np.concatenate(entry["stationary_mask"])

        result = {
            "source_runs": sorted(entry["source_runs"]),
            "by_phase": compute_phase_metrics(
                y_true,
                y_pred,
                dynamic_mask,
                stationary_mask,
            ),
        }
        if response_kind == "current":
            result["zones"] = compute_current_zone_metrics(
                y_true,
                y_pred,
                dynamic_mask,
                stationary_mask,
            )
        else:
            complete_motor_target = np.concatenate(
                entry["motor_target"]
            )

            finite_motor_target = complete_motor_target[
                np.isfinite(complete_motor_target)
            ]

            if finite_motor_target.size > 0:
                target_minimum = float(np.min(finite_motor_target))
                target_maximum = float(np.max(finite_motor_target))
                target_range = target_maximum - target_minimum
            else:
                target_minimum = None
                target_maximum = None
                target_range = None

            result["motor_target_range"] = {
                "minimum": target_minimum,
                "maximum": target_maximum,
                "range": target_range,
                "samples": int(finite_motor_target.size),
            }
            result["motor_target_range_source"] = "held_out_source_run"

            result["range_normalized_by_segment"] = (
                entry["range_normalized_segments"]
            )
            result["range_normalized_summary_percent"] = (
                summarize_range_normalized_segments(
                    entry["range_normalized_segments"]
                )
            )
        results[protocol] = result

    return results

def find_motor_jsonl_files(input_dir, coupling_mode):
    input_dir = Path(input_dir).expanduser()

    if not input_dir.exists():
        raise RuntimeError(f"Input folder does not exist: {input_dir}")

    candidates = []

    for path in input_dir.rglob("*.jsonl"):
        name = path.name.lower()
        path_parts = {part.lower() for part in path.parts}

        if any(
            excluded_part in path_parts
            for excluded_part in {
                "models",
                "plots",
                "prediction_plots",
                "motor_dt_results",
                "full_setup_motor_dt_results",
                "instrument_current_dt_results",
            }
        ):
            continue

        if coupling_mode == "full_setup" and "motor_only" in path_parts:
            continue

        if any(
            excluded_text in name
            for excluded_text in {
                "angle",
                "debug",
                "metadata",
                "motor_dt_replay",
                "instrument_current_dt_replay",
                "webcam",
            }
        ):
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

def load_segments_from_files(
    file_paths,
    coupling_mode,
    conversion_parameters=None,
    controller_starting_positions=None,
):
    all_segments = []

    def chronological_log_key(file_path):
        """Sort DOF logs by the YYYYMMDD_HHMMSS timestamp in their name."""
        path = Path(file_path)
        match = re.search(r"(\d{8})_(\d{6})", path.stem)
        if match:
            return (0, match.group(1), match.group(2), str(path))
        return (1, "", "", str(path))

    ordered_file_paths = sorted(file_paths, key=chronological_log_key)

    command_converter = None
    if coupling_mode != "motor_only":
        print(
            "Coupled Motor-DT input: all four logged "
            "motor_target_positions."
        )

    for file_path in ordered_file_paths:
        file_path = Path(file_path).expanduser()

        print(f"\nLoading motor log: {file_path}")
        rows = load_motor_log(
            file_path,
            coupling_mode=coupling_mode,
            command_converter=command_converter,
        )

        if coupling_mode == "motor_only":
            print("Splitting motor patterns into segments...")
            segments = split_motor_pattern_segments(rows)

        else:
            print("Splitting DOF patterns into segments...")
            segments = split_dof_pattern_segments(rows)

        if not segments and coupling_mode != "motor_only":
            fallback_label = (
                f"{coupling_mode}|tool_sequence|all_motors|"
                f"{file_path.stem}|unknown_time"
            )
            segments = [{
                "task_label": fallback_label,
                "rows": rows,
                "info": parse_task_label(fallback_label),
            }]

        print(f"Found {len(segments)} valid segments in {file_path.name}.")

        for segment in segments:
            segment["source_file"] = str(file_path)

        all_segments.extend(segments)

    if not all_segments:
        raise RuntimeError("No valid task segments found in the selected JSONL files.")

    return all_segments

def create_target_normalized_metrics(metrics):
    """Create a normalized copy without changing the original metrics."""
    normalized_metrics = copy.deepcopy(metrics)

    for motor_name, motor_data in normalized_metrics.items():
        if not motor_name.startswith("motor_"):
            continue

        # Normaliseer de protocolmetrics van alle kandidaatmodellen.
        for model_data in motor_data[
            "encoder_model_comparison"
        ].values():
            for protocol_data in model_data[
                "by_protocol"
            ].values():
                target_range = protocol_data[
                    "motor_target_range"
                ]["range"]

                for phase, phase_metrics in protocol_data[
                    "by_phase"
                ].items():
                    protocol_data["by_phase"][phase] = (
                        add_protocol_target_range_normalization(
                            phase_metrics,
                            target_range,
                        )
                    )

        # Normaliseer de metrics van het geselecteerde model.
        for protocol_data in motor_data[
            "evaluation_by_protocol"
        ].values():
            target_range = protocol_data[
                "motor_target_range"
            ]["range"]

            encoder_phases = protocol_data[
                "encoder_response_by_phase_pulses"
            ]

            for phase, phase_metrics in encoder_phases.items():
                encoder_phases[phase] = (
                    add_protocol_target_range_normalization(
                        phase_metrics,
                        target_range,
                    )
                )

    normalized_metrics["target_range_normalization"] = {
        "formula": (
            "100 * pulse_error / "
            "(max_motor_target - min_motor_target)"
        ),
        "denominator_scope": (
            "complete held-out source run per protocol and motor"
        ),
        "model_retrained": False,
        "predictions_recomputed": False,
    }

    return normalized_metrics

def train_motor_models(samples_by_motor, output_dir, coupling_mode):
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    trained_models = {}
    current_model_comparison_rows = []
    encoder_model_comparison_rows = []
    motor_training_cache = {}
    model_selection_tasks = []


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
    if coupling_mode == "motor_only":
        encoder_feature_names = [
            "commanded_target_pulses",
            "abs_commanded_target_pulses",
            "target_delta_pulses",
            "abs_target_delta_pulses",
            "target_direction",
            "time_since_target_change_s",
            "is_moving",
        ]
        feature_schema = "motor_relative_command_v1"
    else:
        encoder_feature_names = coupled_motor_target_feature_names()
        feature_schema = "all_motor_targets_v1"
    current_feature_names = list(encoder_feature_names)

    for motor_index, samples in samples_by_motor.items():
        if len(samples) < 2:
            print(f"Skipping motor {motor_index}: not enough samples.")
            continue

        train_samples, test_samples = split_train_test(samples)

        X_encoder_train, X_current_train, y_encoder_train, y_current_train = stack_samples(train_samples)
        X_encoder_test, X_current_test, y_encoder_test, y_current_test = stack_samples(test_samples)
        dynamic_test_mask, stationary_test_mask = stack_evaluation_masks(
            test_samples
        )

        encoder_models = make_encoder_models()
        encoder_model_results = {}

        for encoder_model_name, encoder_model_candidate in encoder_models.items():
            encoder_model_candidate.fit(X_encoder_train, y_encoder_train)

            candidate_encoder_pred = encoder_model_candidate.predict(X_encoder_test)

            candidate_encoder_metrics = compute_metrics(
                y_encoder_test,
                candidate_encoder_pred,
            )
            candidate_encoder_phase_metrics = compute_phase_metrics(
                y_encoder_test,
                candidate_encoder_pred,
                dynamic_test_mask,
                stationary_test_mask,
            )
            candidate_encoder_protocol_metrics = compute_metrics_by_protocol(
                test_samples,
                candidate_encoder_pred,
                response_kind="encoder",
            )

            encoder_model_results[encoder_model_name] = {
                "model": encoder_model_candidate,
                "prediction": candidate_encoder_pred,
                "encoder_metrics": candidate_encoder_metrics,
                "phase_metrics": candidate_encoder_phase_metrics,
                "by_protocol": candidate_encoder_protocol_metrics,
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
            key=lambda name: (
                encoder_model_selection_score(encoder_model_results[name]),
                name,
            ),
        )
        model_selection_tasks.append({
            "motor_index": motor_index,
            "output": "encoder_position",
            "results": encoder_model_results,
            "score_function": encoder_model_selection_score,
        })
        
        current_models = make_current_models()
        current_model_results = {}

        for current_model_name, current_model_candidate in current_models.items():
            current_model_candidate.fit(X_current_train, y_current_train)

            candidate_current_pred = current_model_candidate.predict(X_current_test)

            candidate_overall_metrics = compute_metrics(
                y_current_test,
                candidate_current_pred,
            )
            candidate_phase_metrics = compute_phase_metrics(
                y_current_test,
                candidate_current_pred,
                dynamic_test_mask,
                stationary_test_mask,
            )

            candidate_zone_metrics = compute_current_zone_metrics(
                y_current_test,
                candidate_current_pred,
                dynamic_test_mask,
                stationary_test_mask,
            )
            candidate_current_protocol_metrics = compute_metrics_by_protocol(
                test_samples,
                candidate_current_pred,
                response_kind="current",
            )

            current_model_results[current_model_name] = {
                "model": current_model_candidate,
                "prediction": candidate_current_pred,
                "overall_metrics": candidate_overall_metrics,
                "phase_metrics": candidate_phase_metrics,
                "zone_metrics": candidate_zone_metrics,
                "by_protocol": candidate_current_protocol_metrics,
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
            key=lambda name: (
                current_model_selection_score(current_model_results[name]),
                name,
            ),
        )
        model_selection_tasks.append({
            "motor_index": motor_index,
            "output": "motor_current",
            "results": current_model_results,
            "score_function": current_model_selection_score,
        })

        motor_training_cache[motor_index] = {
            "encoder_model_results": encoder_model_results,
            "current_model_results": current_model_results,
            "local_encoder_winner": best_encoder_model_name,
            "local_current_winner": best_current_model_name,
            "n_train_samples": int(len(X_encoder_train)),
            "n_test_samples": int(len(X_encoder_test)),
            "train_segments": len(train_samples),
            "test_segments": len(test_samples),
        }

        print(f"\nMotor {motor_index} candidate comparison completed")
        print(f"  Local encoder winner: {best_encoder_model_name}")
        print(f"  Local current winner: {best_current_model_name}")

    common_model_name, global_selection = select_common_model_family(
        model_selection_tasks
    )

    print("\nGlobal model-family selection")
    for model_name, vote_count in global_selection["vote_counts"].items():
        print(
            f"  {model_name}: {vote_count} task wins, "
            f"mean rank {global_selection['mean_ranks'][model_name]:.2f}"
        )
    print(f"  Selected common model family: {common_model_name}")

    metrics["global_model_selection"] = global_selection

    # Finalize every motor with the same globally selected model family.
    # Candidate models are already fitted, so this does not repeat training.
    for motor_index, cached in motor_training_cache.items():
        encoder_model_results = cached["encoder_model_results"]
        current_model_results = cached["current_model_results"]

        encoder_result = encoder_model_results[common_model_name]
        current_result = current_model_results[common_model_name]

        encoder_model = encoder_result["model"]
        encoder_metrics = encoder_result["encoder_metrics"]
        encoder_phase_metrics = encoder_result["phase_metrics"]
        current_model = current_result["model"]
        current_overall_metrics = current_result["overall_metrics"]
        current_phase_metrics = current_result["phase_metrics"]
        current_zone_metrics = current_result["zone_metrics"]
        encoder_protocol_metrics = encoder_result["by_protocol"]
        current_protocol_metrics = current_result["by_protocol"]
        evaluation_by_protocol = {}
        for protocol in sorted(
            set(encoder_protocol_metrics) | set(current_protocol_metrics)
        ):
            evaluation_by_protocol[protocol] = {
                "source_runs": sorted(set(
                    encoder_protocol_metrics.get(protocol, {}).get("source_runs", [])
                    + current_protocol_metrics.get(protocol, {}).get("source_runs", [])
                )),
                "motor_target_range": (
                    encoder_protocol_metrics.get(protocol, {}).get(
                        "motor_target_range"
                    )
                ),
                "motor_target_range_source": (
                    encoder_protocol_metrics.get(protocol, {}).get(
                        "motor_target_range_source"
                    )
                ),
                "encoder_response_by_phase_pulses": (
                    encoder_protocol_metrics.get(protocol, {}).get("by_phase", {})
                ),
                "encoder_response_normalized_by_target_range": (
                    encoder_protocol_metrics.get(protocol, {}).get(
                        "range_normalized_by_segment",
                        [],
                    )
                ),
                "encoder_response_normalized_summary_percent": (
                    encoder_protocol_metrics.get(protocol, {}).get(
                        "range_normalized_summary_percent",
                        {},
                    )
                ),
                "motor_current_by_phase_mA": (
                    current_protocol_metrics.get(protocol, {}).get("by_phase", {})
                ),
                "motor_current_zones_mA": (
                    current_protocol_metrics.get(protocol, {}).get("zones", {})
                ),
            }
        motor_metrics = {
            "coupling_mode": coupling_mode,
            "encoder_response_pulses": encoder_metrics,
            "encoder_response_by_phase_pulses": encoder_phase_metrics,
            "selected_encoder_model": common_model_name,
            "local_best_encoder_model": cached["local_encoder_winner"],
            "encoder_model_comparison": {
                model_name: {
                    "overall": model_result["encoder_metrics"],
                    "by_phase": model_result["phase_metrics"],
                    "by_protocol": model_result["by_protocol"],
                }
                for model_name, model_result in encoder_model_results.items()
            },            
            "motor_current_mA": current_overall_metrics,
            "motor_current_by_phase_mA": current_phase_metrics,
            "motor_current_zones_mA": current_zone_metrics,
            "evaluation_by_protocol": evaluation_by_protocol,
            "selected_current_model": common_model_name,
            "local_best_current_model": cached["local_current_winner"],
            "current_model_comparison": {
                model_name: {
                    "motor_current_mA": model_result["overall_metrics"],
                    "motor_current_by_phase_mA": model_result["phase_metrics"],
                    "motor_current_zones_mA": model_result["zone_metrics"],
                    "by_protocol": model_result["by_protocol"],
                }
                for model_name, model_result in current_model_results.items()
            },
            "n_train_samples": cached["n_train_samples"],
            "n_test_samples": cached["n_test_samples"],
            "train_segments": cached["train_segments"],
            "test_segments": cached["test_segments"],
            "encoder_feature_names": encoder_feature_names,
            "current_feature_names": current_feature_names,
            "evaluation_windows": {
                "dynamic_definition": (
                    "target_change_until_measured_target_settled"
                ),
                "completion_hold_s": DYNAMIC_COMPLETION_HOLD_S,
                "minimum_position_tolerance_pulses": (
                    DYNAMIC_MIN_POSITION_TOLERANCE_PULSES
                ),
                "relative_position_tolerance": (
                    DYNAMIC_REL_POSITION_TOLERANCE
                ),
                "stationary_encoder_velocity_threshold_pulses_per_s": (
                    STATIONARY_ENCODER_VELOCITY_THRESHOLD
                ),
                "model_feature_memory_s": EVENT_TAIL_S,
            },
        }

        metrics[f"motor_{motor_index}"] = motor_metrics

        model_package = {
            "coupling_mode": coupling_mode,
            "motor_index": motor_index,
            "feature_schema": feature_schema,
            "encoder_model": encoder_model,
            "current_model": current_model,
            "selected_encoder_model": common_model_name,
            "selected_current_model": common_model_name,
            "global_model_selection_policy": global_selection["policy"],
            "encoder_feature_names": encoder_feature_names,
            "current_feature_names": current_feature_names,
        }

        model_path = models_dir / f"motor_dt_m{motor_index}.pkl"
        joblib.dump(model_package, model_path)

        trained_models[motor_index] = model_package

        print(f"\nMotor {motor_index}")
        print(f"  Selected encoder model: {common_model_name}")
        print(f"  Selected current model: {common_model_name}")
        print(f"  Local encoder winner: {cached['local_encoder_winner']}")
        print(f"  Local current winner: {cached['local_current_winner']}")
        print(f"  Encoder MAE: {motor_metrics['encoder_response_pulses']['mae']:.2f} pulses")
        print(f"  Current MAE: {motor_metrics['motor_current_mA']['mae']:.2f} mA")
        for protocol, protocol_metrics in evaluation_by_protocol.items():
            encoder_dynamic = protocol_metrics[
                "encoder_response_by_phase_pulses"
            ].get("dynamic", {})
            current_dynamic = protocol_metrics[
                "motor_current_by_phase_mA"
            ].get("dynamic", {})
            print(
                f"  {protocol}: dynamic encoder RMSE "
                f"{encoder_dynamic.get('rmse', float('nan')):.2f} pulses, "
                f"dynamic current RMSE "
                f"{current_dynamic.get('rmse', float('nan')):.2f} mA"
            )
        print(f"  Saved model: {model_path}")

    metrics_path = output_dir / "motor_dt_metrics.json"

    normalized_metrics_path = (
        output_dir
        / "motor_dt_metrics_target_normalized.json"
    )

    combined_model_path = models_dir / "motor_dt_all_motors.pkl"
    joblib.dump({
        "coupling_mode": coupling_mode,
        "feature_schema": feature_schema,
        "selected_common_model_family": common_model_name,
        "global_model_selection": global_selection,
        "motor_models": trained_models,
    }, combined_model_path)
    print(f"\nSaved combined model package: {combined_model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics: {metrics_path}")

    # Create and save normalized metrics
    normalized_metrics = create_target_normalized_metrics(metrics)
    with open(normalized_metrics_path, "w") as f:
        json.dump(normalized_metrics, f, indent=2)
    print(f"\nSaved normalized metrics: {normalized_metrics_path}")

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
    coupling_modes = {
        trained_models[motor_index]["coupling_mode"]
        for motor_index in trained_models
    }

    if coupling_modes == {"motor_only"}:
        plot_motor_pattern_predictions(
            samples_by_motor,
            trained_models,
            output_dir,
        )
    else:
        plot_dof_pattern_predictions(
            samples_by_motor,
            trained_models,
            output_dir,
        )

# ============================================================
# MOTOR PATTERN PLOTTING
# ============================================================

def plot_motor_pattern_predictions(
    samples_by_motor,
    trained_models,
    output_dir,
):
    plot_dir = output_dir / "prediction_plots" / "motor_patterns"
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

            raw_current = metadata["raw_current"]
            filtered_current = metadata["filtered_current"]

            encoder_pred = encoder_model.predict(X_encoder)
            current_pred = current_model.predict(X_current)

            test_type = metadata["test_type"]
            trial = metadata["trial"]

            encoder_residual = y_encoder - encoder_pred
            current_residual = filtered_current - current_pred

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(13, 10),
                sharex=True,
            )

            fig.suptitle(
                f"Motor DT - M{motor_index} - {test_type} - {trial}",
                fontsize=14,
            )

            # --------------------------------------------------
            # 1. ENCODER POSITION
            # --------------------------------------------------

            axes[0].plot(
                t,
                target,
                linestyle=":",
                linewidth=1.2,
                color="black",
                label="Commanded motor target",
            )

            axes[0].plot(
                t,
                y_encoder,
                linestyle="-",
                linewidth=1.5,
                color="#6BAED6",
                label="Measured encoder position",
            )

            axes[0].plot(
                t,
                encoder_pred,
                linestyle="--",
                linewidth=1.5,
                color="tab:purple",
                label="Predicted encoder position",
            )

            axes[0].set_title("Encoder position")
            axes[0].set_ylabel("Relative position [pulses]")
            axes[0].grid(True)
            axes[0].legend(fontsize=8)

            # --------------------------------------------------
            # 2. MOTOR CURRENT
            # --------------------------------------------------

            axes[1].plot(
                t,
                raw_current,
                linestyle="-",
                linewidth=0.8,
                color="orange",
                alpha=0.30,
                label="Raw measured current",
            )

            axes[1].plot(
                t,
                filtered_current,
                linestyle="-",
                linewidth=1.4,
                color="orange",
                label="Filtered measured current",
            )

            axes[1].plot(
                t,
                current_pred,
                linestyle="--",
                linewidth=1.4,
                color="tab:purple",
                label="Predicted current",
            )

            axes[1].set_title("Motor current")
            axes[1].set_ylabel("Current [mA]")
            axes[1].grid(True)
            axes[1].legend(fontsize=8)

            # --------------------------------------------------
            # 3. ENCODER RESIDUAL
            # measured - predicted
            # --------------------------------------------------

            axes[2].plot(
                t,
                encoder_residual,
                linestyle="-",
                linewidth=1.2,
                color="#4C78A8",
                label="Position residual",
            )

            axes[2].axhline(
                0.0,
                color="black",
                linewidth=1.0,
            )

            axes[2].set_title(
                "Position residual: measured - predicted"
            )
            axes[2].set_ylabel("Residual [pulses]")
            axes[2].grid(True)
            axes[2].legend(fontsize=8)

            # --------------------------------------------------
            # 4. CURRENT RESIDUAL
            # filtered measured - predicted
            # --------------------------------------------------

            axes[3].plot(
                t,
                current_residual,
                linestyle="-",
                linewidth=1.2,
                color="#4C78A8",
                label="Current residual",
            )

            axes[3].axhline(
                0.0,
                color="black",
                linewidth=1.0,
            )

            axes[3].set_title(
                "Current residual: measured - predicted"
            )
            axes[3].set_ylabel("Residual [mA]")
            axes[3].set_xlabel("Time [s]")
            axes[3].grid(True)
            axes[3].legend(fontsize=8)

            plt.tight_layout(
                rect=[0, 0, 1, 0.94]
            )

            filename = (
                f"motor_{motor_index}_"
                f"{safe_name(test_type)}_"
                f"{safe_name(trial)}.png"
            )

            plot_path = plot_dir / filename

            fig.savefig(
                plot_path,
                dpi=200,
            )

            plt.close(fig)

            print(
                f"Saved motor-pattern prediction plot: {plot_path}"
            )


# ============================================================
# DOF PATTERN PLOTTING
# ============================================================

def motor_target_is_commanded(
    sample,
    tolerance_pulses=MOTOR_COMMAND_TOLERANCE_PULSES,
):
    """
    A motor is considered commanded only when its target repeatedly
    changes direction during the motion pattern.
    """
    metadata = sample[4]

    motor_target = np.asarray(
        metadata["target"],
        dtype=float,
    )

    finite_target = motor_target[
        np.isfinite(motor_target)
    ]

    if finite_target.size < 2:
        return False

    target_delta = np.diff(
        finite_target
    )

    significant_directions = np.sign(
        target_delta[
            np.abs(target_delta) > tolerance_pulses
        ]
    )

    if significant_directions.size < 2:
        return False

    number_of_reversals = np.sum(
        significant_directions[1:]
        != significant_directions[:-1]
    )

    return bool(
        number_of_reversals >= 2
    )

def plot_dof_pattern_predictions(
    samples_by_motor,
    trained_models,
    output_dir,
):
    plot_dir = output_dir / "prediction_plots" / "dof_patterns"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for motor_index, samples in samples_by_motor.items():
        if motor_index not in trained_models:
            continue

        _, test_samples = split_train_test(samples)

        encoder_model = trained_models[motor_index]["encoder_model"]
        current_model = trained_models[motor_index]["current_model"]

        # Group sinusoid and triangle that belong to the same
        # DOF + frequency + range combination.
        grouped = {}

        for sample in test_samples:
            X_encoder, X_current, y_encoder, y_current, metadata = sample

            dof_number = metadata.get("dof_number")
            frequency = metadata.get("frequency_factor")
            range_factor = metadata.get("range_factor")
            pattern_type = metadata.get("pattern_type")

            if (
                dof_number is None
                or frequency is None
                or range_factor is None
                or pattern_type not in {"sinusoid", "triangle"}
            ):
                continue

            group_key = (
                int(dof_number),
                float(frequency),
                float(range_factor),
                metadata["run_name"],
            )

            grouped.setdefault(group_key, {})[pattern_type] = sample

        for group_key, patterns in grouped.items():
            dof_number, frequency, range_factor, run_name = group_key

            # It is still useful to plot a group if one waveform is missing.
            sinusoid_sample = patterns.get("sinusoid")
            triangle_sample = patterns.get("triangle")

            available_samples = [
                sample
                for sample in (
                    sinusoid_sample,
                    triangle_sample,
                )
                if sample is not None
            ]

            motor_is_commanded = any(
                motor_target_is_commanded(sample)
                for sample in available_samples
            )

            # Only directly commanded motors need the system-level DOF
            # command above their motor response. Non-commanded motors are
            # still plotted because their encoder/current response can reveal
            # coupling, drag, or an unexpected reaction.
            row_offset = 1 if motor_is_commanded else 0
            encoder_row = row_offset
            current_row = row_offset + 1
            position_residual_row = row_offset + 2
            current_residual_row = row_offset + 3
            number_of_rows = 4 + row_offset

            fig, axes = plt.subplots(
                number_of_rows,
                2,
                figsize=(
                    16,
                    13 if motor_is_commanded else 10.5,
                ),
                squeeze=False,
            )

            command_status = (
                ""
                if motor_is_commanded
                else " - motor not actuated"
            )

            fig.suptitle(
                (
                    f"Motor DT - DOF{dof_number} - M{motor_index} - "
                    f"frequency x{frequency:g} - range x{range_factor:g}"
                    f"{command_status}"
                ),
                fontsize=14,
            )

            waveform_samples = [
                ("Sinusoid", sinusoid_sample),
                ("Triangle", triangle_sample),
            ]

            for column_index, (waveform_name, sample) in enumerate(
                waveform_samples
            ):
                if sample is None:
                    for row_index in range(number_of_rows):
                        axes[row_index, column_index].axis("off")

                    axes[0, column_index].set_title(
                        f"{waveform_name} - unavailable"
                    )
                    continue

                (
                    X_encoder,
                    X_current,
                    y_encoder,
                    y_current,
                    metadata,
                ) = sample

                t = np.asarray(metadata["time"], dtype=float)

                dof_command = np.asarray(
                    metadata["active_instrument_command_rad"],
                    dtype=float,
                )

                motor_target = np.asarray(
                    metadata["target"],
                    dtype=float,
                )

                raw_current = np.asarray(
                    metadata["raw_current"],
                    dtype=float,
                )

                filtered_current = np.asarray(
                    metadata["filtered_current"],
                    dtype=float,
                )

                encoder_pred = encoder_model.predict(X_encoder)
                current_pred = current_model.predict(X_current)

                encoder_residual = y_encoder - encoder_pred
                current_residual = filtered_current - current_pred

                axes[0, column_index].set_title(waveform_name)

                # --------------------------------------------------
                # 1. DOF COMMAND
                # --------------------------------------------------

                if motor_is_commanded:
                    axes[0, column_index].plot(
                        t,
                        dof_command,
                        linestyle="-",
                        linewidth=1.4,
                        color="black",
                        label=f"DOF{dof_number} command",
                    )

                    axes[0, column_index].set_ylabel(
                        "Command [rad]"
                    )
                    axes[0, column_index].grid(True)
                    axes[0, column_index].legend(fontsize=8)

                # --------------------------------------------------
                # 2. ENCODER POSITION
                # --------------------------------------------------
                axes[encoder_row, column_index].plot(
                    t,
                    motor_target,
                    linestyle=":",
                    linewidth=1.2,
                    color="black",
                    label="Commanded motor target",
                )
                
                axes[encoder_row, column_index].plot(
                    t,
                    y_encoder,
                    linestyle="-",
                    linewidth=1.5,
                    color="#6BAED6",
                    label="Measured encoder position",
                )

                axes[encoder_row, column_index].plot(
                    t,
                    encoder_pred,
                    linestyle="--",
                    linewidth=1.5,
                    color="tab:purple",
                    label="Predicted encoder position",
                )

                axes[encoder_row, column_index].set_ylabel(
                    "Relative position [pulses]"
                )
                axes[encoder_row, column_index].grid(True)
                axes[encoder_row, column_index].legend(fontsize=8)

                # --------------------------------------------------
                # 3. MOTOR CURRENT
                # --------------------------------------------------

                axes[current_row, column_index].plot(
                    t,
                    raw_current,
                    linestyle="-",
                    linewidth=0.8,
                    color="orange",
                    alpha=0.30,
                    label="Raw measured current",
                )

                axes[current_row, column_index].plot(
                    t,
                    filtered_current,
                    linestyle="-",
                    linewidth=1.4,
                    color="orange",
                    label="Filtered measured current",
                )

                axes[current_row, column_index].plot(
                    t,
                    current_pred,
                    linestyle="--",
                    linewidth=1.4,
                    color="tab:purple",
                    label="Predicted current",
                )

                axes[current_row, column_index].set_ylabel(
                    "Current [mA]"
                )
                axes[current_row, column_index].grid(True)
                axes[current_row, column_index].legend(fontsize=8)

                # --------------------------------------------------
                # 4. POSITION RESIDUAL
                # measured - predicted
                # --------------------------------------------------

                axes[position_residual_row, column_index].plot(
                    t,
                    encoder_residual,
                    linestyle="-",
                    linewidth=1.2,
                    color="#4C78A8",
                    label="Position residual",
                )

                axes[position_residual_row, column_index].axhline(
                    0.0,
                    color="black",
                    linewidth=1.0,
                )

                axes[position_residual_row, column_index].set_ylabel(
                    "Residual [pulses]"
                )
                axes[position_residual_row, column_index].grid(True)
                axes[position_residual_row, column_index].legend(fontsize=8)

                # --------------------------------------------------
                # 5. CURRENT RESIDUAL
                # filtered measured - predicted
                # --------------------------------------------------

                axes[current_residual_row, column_index].plot(
                    t,
                    current_residual,
                    linestyle="-",
                    linewidth=1.2,
                    color="#4C78A8",
                    label="Current residual",
                )

                axes[current_residual_row, column_index].axhline(
                    0.0,
                    color="black",
                    linewidth=1.0,
                )

                axes[current_residual_row, column_index].set_ylabel(
                    "Residual [mA]"
                )
                axes[current_residual_row, column_index].set_xlabel(
                    "Time [s]"
                )
                axes[current_residual_row, column_index].grid(True)
                axes[current_residual_row, column_index].legend(fontsize=8)

            # Row labels / titles only once conceptually
            if motor_is_commanded:
                axes[0, 0].text(
                    -0.14,
                    0.5,
                    "DOF command",
                    transform=axes[0, 0].transAxes,
                    rotation=90,
                    va="center",
                    fontweight="bold",
                )

            axes[encoder_row, 0].text(
                -0.14,
                0.5,
                "Encoder position",
                transform=axes[encoder_row, 0].transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )

            axes[current_row, 0].text(
                -0.14,
                0.5,
                "Motor current",
                transform=axes[current_row, 0].transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )

            axes[position_residual_row, 0].text(
                -0.14,
                0.5,
                "Position residual",
                transform=axes[
                    position_residual_row,
                    0,
                ].transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )

            axes[current_residual_row, 0].text(
                -0.14,
                0.5,
                "Current residual",
                transform=axes[
                    current_residual_row,
                    0,
                ].transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )

            plt.tight_layout(
                rect=[0.03, 0, 1, 0.96]
            )

            dof_dir = plot_dir / f"dof_{dof_number}"
            dof_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = (
                f"dof_{dof_number}_"
                f"motor_{motor_index}_"
                f"freq_{safe_name(f'{frequency:g}')}_"
                f"range_{safe_name(f'{range_factor:g}')}_"
                f"{safe_name(run_name)}.png"
            )

            plot_path = dof_dir / filename

            fig.savefig(
                plot_path,
                dpi=200,
            )

            plt.close(fig)

            print(
                f"Saved DOF-pattern prediction plot: {plot_path}"
            )

def train_motor_digital_twin(
    file_paths,
    output_dir,
    coupling_mode,
    conversion_parameters=None,
    controller_starting_positions=None,
):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    # Eerst sorteren: noodzakelijk om met dezelfde seed
    # altijd dezelfde run te selecteren.
    file_paths = sorted(
        Path(file_path).expanduser().resolve()
        for file_path in file_paths
    )

    if len(file_paths) < 2:
        raise RuntimeError(
            "At least two complete runs are needed for a train/test split."
        )

    segments = load_segments_from_files(
        file_paths=file_paths,
        coupling_mode=coupling_mode,
        conversion_parameters=conversion_parameters,
        controller_starting_positions=controller_starting_positions,
    )

    if coupling_mode != "motor_only":
        command_input_metadata = {
            "method": "logged_motor_target_features",
            "applies_to": ["gearbox_only", "full_setup"],
            "model_input": (
                "all_four_motor_target_positions_"
                "with_continuous_run_history"
            ),
            "target_reference": (
                "relative_to_motion_pattern_start"
            ),
            "encoder_target": (
                "encoder_displacement_relative_to_motion_pattern_start"
            ),
            "logged_motor_targets_used": True,
            "reconstructed_motor_targets_used": False,
            "controller_starting_positions_required": False,
            "gearbox_variant": (
                conversion_parameters["variant"]
                if conversion_parameters is not None
                else None
            ),
            "feature_names": coupled_motor_target_feature_names(),
            "log_order": [
                str(path)
                for path in sorted(
                    file_paths,
                    key=lambda path: (
                        re.search(
                            r"(\d{8})_(\d{6})",
                            Path(path).stem,
                        ).groups()
                        if re.search(
                            r"(\d{8})_(\d{6})",
                            Path(path).stem,
                        )
                        else ("99999999", "999999")
                    ),
                )
            ],
        }
        with open(output_dir / "command_input_metadata.json", "w") as f:
            json.dump(command_input_metadata, f, indent=2)

    assign_complete_run_splits(
        segments=segments,
        coupling_mode=coupling_mode,
    )

    print(f"\nTotal valid segments: {len(segments)}")

    samples_by_motor = build_dataset(
        segments=segments,
        coupling_mode=coupling_mode,
    )

    for motor_index, samples in samples_by_motor.items():
        print(f"Motor {motor_index}: {len(samples)} segments")

    trained_models, _ = train_motor_models(
        samples_by_motor=samples_by_motor,
        output_dir=output_dir,
        coupling_mode=coupling_mode,
    )
    plot_predictions(samples_by_motor, trained_models, output_dir)

    print(f"\nMotor Digital Twin training completed.")
    print(f"Output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train separate Motor Digital Twins for motor-only, "
            "gearbox-only and full-setup data."
        )
    )

    parser.add_argument(
        "--configuration",
        choices=["all", *VALID_CONFIGURATIONS],
        default="all",
        help="Configuration to train. Default: train all three configurations.",
    )
    parser.add_argument(
        "--training-config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="Training configuration YAML.",
    )

    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help=(
            "Optional path to a specific ROS .jsonl log. Can be repeated. "
            "Only valid when one configuration is selected."
        ),
    )

    parser.add_argument(
        "--input-dir",
        default=None,
        help="Optional input-folder override for one selected configuration.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-folder override for one selected configuration.",
    )

    parser.add_argument(
        "--gearbox-config",
        default=str(DEFAULT_GEARBOX_CONFIG_PATH),
        help=(
            "Path to gearbox_params.yaml, used only to record the selected "
            "gearbox variant in coupled-model metadata."
        ),
    )

    parser.add_argument(
        "--gearbox-variant",
        default=None,
        help=(
            "Gearbox variant from gearbox_params.yaml. Default: active_variant "
            "from that file."
        ),
    )

    parser.add_argument(
        "--controller-starting-positions",
        nargs=4,
        type=float,
        metavar=("M0", "M1", "M2", "M3"),
        default=None,
        help=(
            "Deprecated compatibility option. Coupled models now use logged "
            "motor target positions and do not require controller start positions."
        ),
    )

    args = parser.parse_args()

    training_config, training_data_root = load_training_config(
        args.training_config
    )

    motor_configurations_raw = training_config[
        "motor_dt"
    ]["configurations"]

    training_configurations = {}

    for coupling_mode in VALID_CONFIGURATIONS:
        config_entry = motor_configurations_raw[coupling_mode]

        training_configurations[coupling_mode] = {
            "input_dir": resolve_training_path(
                training_data_root,
                config_entry["input_dir"],
            ),
            "output_dir": resolve_training_path(
                training_data_root,
                config_entry["output_dir"],
            ),
        }

    if args.configuration == "all":
        if args.file is not None or args.input_dir is not None or args.output_dir is not None:
            raise RuntimeError(
                "--file, --input-dir and --output-dir require one specific "
                "--configuration. Edit training_config.yaml to change the default paths "
                "used by --configuration all."
            )
        selected_configurations = list(VALID_CONFIGURATIONS)
    else:
        selected_configurations = [args.configuration]

    for coupling_mode in selected_configurations:
        defaults = training_configurations[coupling_mode]
        input_dir = (
            Path(args.input_dir).expanduser()
            if args.input_dir is not None
            else defaults["input_dir"]
        )
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir is not None
            else defaults["output_dir"]
        )

        print("\n" + "=" * 80)
        print(f"Training Motor DT configuration: {coupling_mode}")
        print(f"Input folder:  {input_dir}")
        print(f"Output folder: {output_dir}")
        print("=" * 80)

        if args.file is not None:
            file_paths = [Path(path).expanduser() for path in args.file]
        else:
            file_paths = find_motor_jsonl_files(
                input_dir=input_dir,
                coupling_mode=coupling_mode,
            )

        conversion_parameters = None
        if coupling_mode != "motor_only":
            conversion_parameters = load_gearbox_conversion_parameters(
                config_path=args.gearbox_config,
                gearbox_variant=args.gearbox_variant,
            )
            print(
                "Coupled Motor-DT metadata: "
                f"gearbox variant {conversion_parameters['variant']}"
            )
            if args.controller_starting_positions is not None:
                print(
                    "Note: --controller-starting-positions is ignored because "
                    "no motor-target reconstruction is performed."
                )

        train_motor_digital_twin(
            file_paths=file_paths,
            output_dir=output_dir,
            coupling_mode=coupling_mode,
            conversion_parameters=conversion_parameters,
            controller_starting_positions=args.controller_starting_positions,
        )

if __name__ == "__main__":
    main()