#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


DEFAULT_MODEL_ROOT = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "trainings data V2"
    / "instrument_current_dt_results"
)


def default_model_file_for_dof(dof):
    return (
        DEFAULT_MODEL_ROOT
        / f"dof{dof}"
        / "models"
        / f"instrument_current_dt_dof{dof}_all_motors.pkl"
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
    if data.get("time") is not None:
        return data.get("time")

    if data.get("timestamp") is not None:
        return data.get("timestamp")

    if data.get("ros_timestamp") is not None:
        return data.get("ros_timestamp")

    return None


def as_float_array(value, expected_length):
    if value is None or len(value) < expected_length:
        return None

    array = np.array(value[:expected_length], dtype=float)

    if not np.all(np.isfinite(array)):
        return None

    return array


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


def time_since_signal_change(t, signal, threshold=1e-5):
    result = np.zeros_like(t, dtype=float)

    if len(t) == 0:
        return result

    last_change_time = t[0]
    previous_signal = signal[0]

    for i in range(len(t)):
        if abs(signal[i] - previous_signal) > threshold:
            last_change_time = t[i]
            previous_signal = signal[i]

        result[i] = t[i] - last_change_time

    return result


def build_instrument_current_features(t, target):
    """
    Build the same 7 command-only features as used during
    instrument current training.

    Feature order:
    1. target
    2. abs_target
    3. target_delta
    4. abs_target_delta
    5. target_direction
    6. time_after_change
    7. target_based_is_moving
    """
    command_change_threshold = 1e-5

    abs_target = np.abs(target)

    instant_target_delta = np.zeros_like(target)
    instant_target_delta[1:] = target[1:] - target[:-1]

    target_delta = np.zeros_like(target)
    last_delta = 0.0

    for sample_index in range(len(target)):
        if abs(instant_target_delta[sample_index]) > command_change_threshold:
            last_delta = instant_target_delta[sample_index]

        target_delta[sample_index] = last_delta

    abs_target_delta = np.abs(target_delta)
    target_direction = np.sign(target_delta)

    time_after_change = time_since_signal_change(
        t,
        target,
        threshold=command_change_threshold,
    )

    movement_memory_s = 1.2
    target_based_is_moving = (
        (np.abs(target_delta) > command_change_threshold)
        & (time_after_change < movement_memory_s)
    ).astype(float)

    return np.column_stack([
        target,
        abs_target,
        target_delta,
        abs_target_delta,
        target_direction,
        time_after_change,
        target_based_is_moving,
    ]), target_based_is_moving


def load_tool_log(file_path):
    times = []
    commanded = []
    currents = []
    task_labels = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            t_value = get_time_value(data)
            cmd = as_float_array(data.get("commanded_instrument_angles"), 4)
            cur = as_float_array(data.get("measured_currents"), 4)

            if t_value is None or cmd is None or cur is None:
                continue

            times.append(float(t_value))
            commanded.append(cmd)
            currents.append(cur)
            task_labels.append(data.get("task_label") or data.get("task") or "unlabeled")

    if len(times) < 20:
        raise RuntimeError(f"Not enough usable samples in {file_path}")

    t = np.array(times, dtype=float)
    t = t - t[0]

    return {
        "time": t,
        "commanded": np.vstack(commanded),
        "currents": np.vstack(currents),
        "task_labels": task_labels,
    }


def load_instrument_current_model(model_file):
    model_file = Path(model_file).expanduser()

    if not model_file.exists():
        raise RuntimeError(f"Missing Instrument Current DT model: {model_file}")

    model_package = joblib.load(model_file)

    expected_features = [
        "target",
        "abs_target",
        "target_delta",
        "abs_target_delta",
        "target_direction",
        "time_after_change",
        "target_based_is_moving",
    ]

    feature_names = model_package.get("feature_names")

    if feature_names != expected_features:
        raise RuntimeError(
            "The loaded model does not use the expected command-only features.\n"
            f"Expected: {expected_features}\n"
            f"Found:    {feature_names}"
        )

    return model_package


def default_output_file(input_file):
    input_file = Path(input_file)
    name = input_file.name

    if name.endswith("_ros_log.jsonl"):
        output_name = name.replace("_ros_log.jsonl", "_instrument_current_dt_replay.jsonl")
    else:
        output_name = input_file.stem + "_instrument_current_dt_replay.jsonl"

    return input_file.with_name(output_name)


def replay_instrument_current_digital_twin(
    input_file,
    model_file,
    output_file,
    current_filter_window_size=5,
    active_dof=None,
):
    input_file = Path(input_file).expanduser()
    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading log: {input_file}")
    log = load_tool_log(input_file)

    print(f"Loading Instrument Current DT model: {model_file}")
    model_package = load_instrument_current_model(model_file)

    model_active_dof = model_package.get("active_dof")

    if active_dof is None:
        if model_active_dof is None:
            raise RuntimeError(
                "No active_dof found in model package and no --dof was provided."
            )

        active_dof = int(model_active_dof)

    else:
        if model_active_dof is not None and int(model_active_dof) != int(active_dof):
            raise RuntimeError(
                f"DOF mismatch: --dof {active_dof} was requested, "
                f"but the loaded model was trained for DOF{model_active_dof}."
            )

    active_dof_index = active_dof - 1
    t = log["time"]
    commanded = log["commanded"]
    raw_currents = log["currents"]

    target = commanded[:, active_dof_index]

    features, is_moving = build_instrument_current_features(
        t=t,
        target=target,
    )

    filtered_currents = np.column_stack([
        causal_moving_average(
            raw_currents[:, motor_index],
            window_size=current_filter_window_size,
        )
        for motor_index in range(4)
    ])

    predicted_currents = np.full_like(filtered_currents, np.nan, dtype=float)

    models_by_motor = model_package["models_by_motor"]

    for motor_index in range(4):
        model = models_by_motor.get(motor_index)

        if model is None:
            model = models_by_motor.get(str(motor_index))

        if model is None:
            raise RuntimeError(f"Missing model for motor {motor_index}")

        predicted_currents[:, motor_index] = model.predict(features)

    filtered_current_deviation = filtered_currents - predicted_currents
    raw_current_deviation = raw_currents - predicted_currents

    written_rows = 0

    with open(output_file, "w") as f:
        for sample_index in range(len(t)):
            output_row = {
                "time": safe_float(t[sample_index]),
                "sample_index": int(sample_index),

                "active_dof": int(active_dof),
                "active_command_target": safe_float(target[sample_index]),
                "target_based_is_moving": safe_float(is_moving[sample_index]),

                "commanded_instrument_angles": safe_list(commanded[sample_index, :]),

                "raw_measured_currents": safe_list(raw_currents[sample_index, :]),
                "filtered_measured_currents": safe_list(filtered_currents[sample_index, :]),

                "offline_predicted_currents": safe_list(predicted_currents[sample_index, :]),

                "filtered_current_deviation": safe_list(filtered_current_deviation[sample_index, :]),
                "raw_current_deviation": safe_list(raw_current_deviation[sample_index, :]),

                # Backward-compatible old name.
                "current_deviation": safe_list(filtered_current_deviation[sample_index, :]),
            }

            f.write(json.dumps(output_row) + "\n")
            written_rows += 1

    print(f"Saved offline Instrument Current DT replay: {output_file}")
    print(f"Written rows: {written_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay trained Instrument Current DT offline on a tool ROS log."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the *_ros_log.jsonl file.",
    )

    parser.add_argument(
        "--model-file",
        default=None,
        help=(
            "Path to instrument_current_dt_dofX_all_motors.pkl. "
            "If omitted, the model is selected automatically from --dof."
        ),
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

    parser.add_argument(
        "--dof",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Override active DOF. If omitted, active_dof from the model package is used.",
    )

    args = parser.parse_args()

    input_file = Path(args.file).expanduser()

    if args.output_file is None:
        output_file = default_output_file(input_file)
    else:
        output_file = Path(args.output_file).expanduser()

    if args.model_file is None:
        if args.dof is None:
            raise RuntimeError(
                "--dof is required when --model-file is not provided."
            )

        model_file = default_model_file_for_dof(args.dof)
    else:
        model_file = Path(args.model_file).expanduser()

    replay_instrument_current_digital_twin(
        input_file=input_file,
        model_file=model_file,
        output_file=output_file,
        current_filter_window_size=args.current_filter_window_size,
        active_dof=args.dof,
    )

if __name__ == "__main__":
    main()