#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

# Default directory for the stored motor digital twin model
DEFAULT_MODEL_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "setup_01_motors"
    / "training motor data NEW"
    / "motor_dt_results"
    / "models"
)


def safe_float(value):
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value

def safe_list(values):
    return [safe_float(value) for value in values]

def get_time_value(data):
    if "time" in data:
        return data["time"]

    if "timestamp" in data:
        return data["timestamp"]

    if "ros_timestamp" in data:
        return data["ros_timestamp"]

    return None


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
            if commands is None:
                commands = data.get("commands")

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


def relative_commands_to_target(commands):
    """
    Convert relative step commands into cumulative target positions.

    Example:
    raw command:        0, +300, 0, -300, 0
    commanded target:   0, +300, +300, 0, 0

    This reconstructed target is the command input used by the Motor DT.
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


def time_since_signal_change(t, signal):
    result = np.zeros_like(t, dtype=float)

    if len(t) == 0:
        return result

    last_change_time = t[0]
    previous_signal = signal[0]

    for i in range(len(t)):
        if signal[i] != previous_signal:
            last_change_time = t[i]
            previous_signal = signal[i]

        result[i] = t[i] - last_change_time

    return result


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


def segment_to_arrays(segment):
    rows = segment["rows"]

    t_global = np.array([row["time"] for row in rows], dtype=float)
    t_segment = t_global - t_global[0]

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

    for motor_index in range(4):
        positions[motor_index] = positions[motor_index] - positions[motor_index][0]

    return t_global, t_segment, commands, positions, currents


def load_motor_dt_models(model_dir):
    model_dir = Path(model_dir).expanduser()
    models = {}

    for motor_index in range(4):
        model_path = model_dir / f"motor_dt_m{motor_index}.pkl"

        if not model_path.exists():
            raise RuntimeError(f"Missing Motor DT model: {model_path}")

        models[motor_index] = joblib.load(model_path)

    return models

def build_motor_dt_features(t_segment, target):
    """
    Build the same input features as used during Motor DT training.

    The motor controller sends relative step commands. These are first
    reconstructed into a cumulative commanded target. From that target,
    the model uses position, step size, direction and timing features.

    Feature order must match the saved model package:
    1. commanded_target_pulses
    2. abs_commanded_target_pulses
    3. target_delta_pulses
    4. abs_target_delta_pulses
    5. target_direction
    6. time_since_target_change_s
    7. is_moving
    """
    abs_target = np.abs(target)

    # Step in commanded target at the exact moment the target changes.
    instant_target_delta = np.zeros_like(target)
    instant_target_delta[1:] = target[1:] - target[:-1]

    # Keep the last non-zero target step during the motor response phase.
    # This tells the model which movement caused the response.
    target_delta = np.zeros_like(target)
    last_delta = 0.0

    for sample_index in range(len(target)):
        if instant_target_delta[sample_index] != 0.0:
            last_delta = instant_target_delta[sample_index]

        target_delta[sample_index] = last_delta

    abs_target_delta = np.abs(target_delta)
    target_direction = np.sign(target_delta)
    time_since_target_change = time_since_signal_change(t_segment, target)

    movement_memory_s = 1.2
    target_based_is_moving = (
        (np.abs(target_delta) > 0.0)
        & (time_since_target_change < movement_memory_s)
    ).astype(float)

    return np.column_stack([
        target,
        abs_target,
        target_delta,
        abs_target_delta,
        target_direction,
        time_since_target_change,
        target_based_is_moving,
    ])

def predict_segment_offline(t_segment, raw_commands, models):
    """
    Replay the trained Motor DT offline for one segment.

    Input:
    - raw_commands: relative motor commands from the ROS log

    Output:
    - commanded_target: reconstructed cumulative target position
    - predicted_positions: offline predicted encoder response
    - predicted_currents: offline predicted filtered motor current
    """
    raw_commands = np.asarray(raw_commands, dtype=float)

    commanded_target = relative_commands_to_target(raw_commands)

    predicted_positions = np.full_like(raw_commands, np.nan, dtype=float)
    predicted_currents = np.full_like(raw_commands, np.nan, dtype=float)

    for motor_index in range(4):
        target = commanded_target[motor_index]
        motor_features = build_motor_dt_features(t_segment, target)

        model_package = models[motor_index]

        predicted_positions[motor_index] = model_package["encoder_model"].predict(
            motor_features
        )

        predicted_currents[motor_index] = model_package["current_model"].predict(
            motor_features
        )

    return commanded_target, predicted_positions, predicted_currents

def default_output_file(input_file):
    input_file = Path(input_file)
    name = input_file.name

    if name.endswith("_ros_log.jsonl"):
        output_name = name.replace("_ros_log.jsonl", "_motor_dt_replay.jsonl")
    else:
        output_name = input_file.stem + "_motor_dt_replay.jsonl"

    return input_file.with_name(output_name)


def replay_motor_digital_twin(input_file, model_dir, output_file, current_filter_window_size=5):
    input_file = Path(input_file).expanduser()
    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading log: {input_file}")
    rows = load_motor_log(input_file)

    print("Splitting into segments...")
    segments = split_into_segments(rows)

    print(f"Found {len(segments)} valid segments.")

    print(f"Loading Motor DT models from: {model_dir}")
    models = load_motor_dt_models(model_dir)

    written_rows = 0

    with open(output_file, "w") as f:
        for segment in segments:
            task_label = segment["task_label"]
            info = segment["info"]

            (
                t_global,
                t_segment,
                raw_commands,
                measured_positions,
                raw_currents,
            ) = segment_to_arrays(segment)

            filtered_currents = np.vstack([
                causal_moving_average(
                    raw_currents[motor_index],
                    window_size=current_filter_window_size,
                )
                for motor_index in range(4)
            ])

            (
                commanded_target,
                predicted_positions,
                predicted_currents,
            ) = predict_segment_offline(
                t_segment=t_segment,
                raw_commands=raw_commands,
                models=models,
            )

            encoder_deviation = measured_positions - predicted_positions
            # Current model is trained on filtered current, so this is the main current prediction error.
            filtered_current_deviation = filtered_currents - predicted_currents
            # Extra diagnostic error against raw measured current.
            raw_current_deviation = raw_currents - predicted_currents
            
            for sample_index in range(len(t_segment)):
                output_row = {
                    "time": safe_float(t_global[sample_index]),
                    "segment_time": safe_float(t_segment[sample_index]),
                    "task_label": task_label,
                    "test_type": info["test_type"],
                    "motor_name": info["motor_name"],
                    "trial": info["trial"],
                    "sample_index": int(sample_index),

                    "raw_commands": safe_list(raw_commands[:, sample_index]),
                    "offline_commanded_target": safe_list(commanded_target[:, sample_index]),

                    "measured_positions": safe_list(measured_positions[:, sample_index]),
                    "raw_measured_currents": safe_list(raw_currents[:, sample_index]),
                    "filtered_measured_currents": safe_list(filtered_currents[:, sample_index]),

                    "offline_predicted_positions": safe_list(predicted_positions[:, sample_index]),
                    "offline_predicted_currents": safe_list(predicted_currents[:, sample_index]),

                    "encoder_deviation": safe_list(encoder_deviation[:, sample_index]),
                    "filtered_current_deviation": safe_list(filtered_current_deviation[:, sample_index]),
                    "raw_current_deviation": safe_list(raw_current_deviation[:, sample_index]),
                    # Backward-compatible old name.
                    "current_deviation": safe_list(filtered_current_deviation[:, sample_index]),
                }

                f.write(json.dumps(output_row) + "\n")
                written_rows += 1

    print(f"Saved offline Motor DT replay: {output_file}")
    print(f"Written rows: {written_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay trained Motor DT models offline on a motor-only ROS log."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the motor-only *_ros_log.jsonl file.",
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Folder containing motor_dt_m0.pkl ... motor_dt_m3.pkl.",
    )

    parser.add_argument(
        "--output-file",
        default=None,
        help="Output replay .jsonl file. If omitted, it is saved next to the input log.",
    )

    parser.add_argument(
        "--current-filter-window-size",
        type=int,
        default=5,
        help="Causal moving average window size for measured current.",
    )

    args = parser.parse_args()

    input_file = Path(args.file).expanduser()

    if args.output_file is None:
        output_file = default_output_file(input_file)
    else:
        output_file = Path(args.output_file).expanduser()

    replay_motor_digital_twin(
        input_file=input_file,
        model_dir=Path(args.model_dir).expanduser(),
        output_file=output_file,
        current_filter_window_size=args.current_filter_window_size,
    )


if __name__ == "__main__":
    main()