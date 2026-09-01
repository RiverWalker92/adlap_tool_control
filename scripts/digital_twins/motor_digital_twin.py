#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

# Default directory for the stored motor digital twin model depending on the setup.
DEFAULT_MOTOR_ONLY_MODEL_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "trainings data V4"
    / "motors_only"
    / "motors_only_results"
    / "models"
)

DEFAULT_GEARBOX_MODEL_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "trainings data V4"
    / "motors_gearbox"
    / "motors_gearbox_results"
    / "models"
)

DEFAULT_FULL_SETUP_MODEL_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "trainings data V4"
    / "full_setup"
    / "full_setup_results"
    / "models"
)

EVENT_TAIL_S = 1.0

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


def load_motor_log(file_path, coupling_mode):
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

            motor_targets = data.get("motor_target_positions")

            instrument_dof_commands = data.get("commanded_instrument_angles")
            # if instrument_dof_commands is None or len(instrument_dof_commands) < 4:
            #     instrument_dof_commands = [np.nan, np.nan, np.nan, np.nan]
            if (
                coupling_mode != "motor_only"
                and (
                    instrument_dof_commands is None
                    or len(instrument_dof_commands) < 4
                )
            ):
                continue
            if coupling_mode == "motor_only":
                instrument_dof_commands = [0.0, 0.0, 0.0, 0.0]

            if (
                coupling_mode != "motor_only"
                and (
                    motor_targets is None
                    or len(motor_targets) < 4
                )
            ):
                continue

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
                "motor_targets": (
                    [float(x) for x in motor_targets[:4]]
                    if motor_targets is not None and len(motor_targets) >= 4
                    else [0.0, 0.0, 0.0, 0.0]
                ),
                "instrument_dof_commands": [float(x) for x in instrument_dof_commands[:4]],
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


def time_since_signal_change(t, signal, threshold=0.0):
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

def safe_gradient(values, time):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)

    if len(values) < 2:
        return np.zeros_like(values)

    dt = np.diff(time)
    positive_dt = dt[dt > 0.0]
    median_dt = np.nanmedian(positive_dt) if len(positive_dt) else 0.01

    safe_time = time.copy()

    for sample_index in range(1, len(safe_time)):
        if safe_time[sample_index] <= safe_time[sample_index - 1]:
            safe_time[sample_index] = (
                safe_time[sample_index - 1] + median_dt
            )

    return np.gradient(values, safe_time)


def held_command_delta(command):
    command = np.asarray(command, dtype=float)

    instantaneous_delta = np.zeros_like(command)

    if len(command) > 1:
        instantaneous_delta[1:] = np.diff(command)

    held_delta = np.zeros_like(command)
    last_delta = 0.0

    for sample_index, delta in enumerate(instantaneous_delta):
        if abs(delta) > 1e-12:
            last_delta = delta

        held_delta[sample_index] = last_delta

    return instantaneous_delta, held_delta

def build_coupled_motor_dt_features(
    t_segment,
    absolute_motor_targets,
    motor_target_reference,
):
    """
    Build the 32 all_motor_targets_v1 features used during
    gearbox-only and full-setup training.
    """

    absolute_motor_targets = np.asarray(
        absolute_motor_targets,
        dtype=float,
    )

    motor_target_reference = np.asarray(
        motor_target_reference,
        dtype=float,
    )

    if (
        absolute_motor_targets.ndim != 2
        or absolute_motor_targets.shape[0] != 4
    ):
        raise RuntimeError(
            "Expected four motor_target_positions channels, "
            f"but received shape {absolute_motor_targets.shape}."
        )

    if motor_target_reference.shape != (4,):
        raise RuntimeError(
            "Expected four motor-target reference values."
        )

    if np.any(~np.isfinite(absolute_motor_targets)):
        raise RuntimeError(
            "The ROS log contains missing or invalid "
            "motor_target_positions."
        )

    feature_columns = []

    for target_motor_index in range(4):
        absolute_target = absolute_motor_targets[
            target_motor_index
        ]

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

        target_direction = np.sign(
            held_delta
        )

        target_velocity = safe_gradient(
            absolute_target,
            t_segment,
        )

        time_after_change = time_since_signal_change(
            t_segment,
            absolute_target,
            threshold=0.0,
        )

        target_is_moving = (
            (np.abs(held_delta) > 1e-12)
            & (time_after_change < EVENT_TAIL_S)
        ).astype(float)

        feature_columns.extend([
            relative_target,
            previous_relative_target,
            instantaneous_delta,
            np.abs(instantaneous_delta),
            target_direction,
            target_velocity,
            time_after_change,
            target_is_moving,
        ])

    features = np.column_stack(
        feature_columns
    )

    if features.shape[1] != 32:
        raise RuntimeError(
            f"Expected 32 all_motor_targets_v1 features, "
            f"but created {features.shape[1]}."
        )

    return features

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


def segment_to_arrays(segment, coupling_mode):
    rows = segment["rows"]

    t_global = np.array(
        [row["time"] for row in rows],
        dtype=float,
    )

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

    motor_targets = np.array(
        [row["motor_targets"] for row in rows],
        dtype=float,
    ).T

    instrument_dof_commands = np.array(
        [row["instrument_dof_commands"] for row in rows],
        dtype=float,
    ).T

    # Motor-only models are trained relative to the beginning
    # of the motor-pattern segment.
    if coupling_mode == "motor_only":
        for motor_index in range(4):
            positions[motor_index] = (
                positions[motor_index]
                - positions[motor_index, 0]
            )

    # Coupled configurations deliberately remain ABSOLUTE here.
    # They are converted to pattern-relative positions later,
    # using the neutral position immediately before each pattern.

    return (
        t_global,
        t_segment,
        commands,
        motor_targets,
        instrument_dof_commands,
        positions,
        currents,
    )

def make_coupled_pattern_relative_positions(
    t_segment,
    instrument_commands,
    absolute_positions,
    active_dof=None,
    command_tolerance=1e-4,
    minimum_pause_s=1.0,
):
    """
    Convert measured coupled encoder positions to the same reference
    used during Motor-DT training.

    Each individual DOF motion pattern is referenced to the first
    sample of that motion pattern, matching Motor-DT training.
    """

    t_segment = np.asarray(
        t_segment,
        dtype=float,
    )

    instrument_commands = np.asarray(
        instrument_commands,
        dtype=float,
    )

    absolute_positions = np.asarray(
        absolute_positions,
        dtype=float,
    )

    if len(t_segment) < 3:
        return absolute_positions.copy(), []

    # ----------------------------------------------------------
    # 1. Determine active DOF
    # ----------------------------------------------------------

    if active_dof is not None:
        dof_index = int(active_dof) - 1

    else:
        # Fallback: channel with largest command range.
        command_ranges = np.ptp(
            instrument_commands,
            axis=1,
        )

        dof_index = int(
            np.argmax(command_ranges)
        )

    command = instrument_commands[
        dof_index
    ]

    # ----------------------------------------------------------
    # 2. Determine neutral command
    # ----------------------------------------------------------

    initial_mask = (
        np.isfinite(command)
        & np.isfinite(t_segment)
        & (
            t_segment
            <= min(
                1.0,
                float(t_segment[-1]),
            )
        )
    )

    if np.any(initial_mask):
        baseline = float(
            np.median(
                command[initial_mask]
            )
        )
    else:
        baseline = float(command[0])

    moving = (
        np.isfinite(command)
        & (
            np.abs(
                command - baseline
            )
            > command_tolerance
        )
    )

    pause_mask = (
        np.isfinite(command)
        & ~moving
    )

    # ----------------------------------------------------------
    # 3. Find sustained neutral pauses
    # ----------------------------------------------------------

    pause_intervals = []
    pause_start = None

    for index, is_pause in enumerate(
        pause_mask
    ):
        if (
            is_pause
            and pause_start is None
        ):
            pause_start = index

        is_last = (
            index
            == len(pause_mask) - 1
        )

        if (
            pause_start is not None
            and (
                (not is_pause)
                or is_last
            )
        ):
            pause_end = (
                index
                if is_pause and is_last
                else index - 1
            )

            duration = (
                t_segment[pause_end]
                - t_segment[pause_start]
            )

            if duration >= minimum_pause_s:
                pause_intervals.append(
                    (
                        pause_start,
                        pause_end,
                    )
                )

            pause_start = None

    # ----------------------------------------------------------
    # 4. Convert pauses to the same motion blocks as training
    # ----------------------------------------------------------

    motion_blocks = []
    previous_pause_end = -1

    for pause_start, pause_end in pause_intervals:

        block_start = (
            previous_pause_end + 1
        )

        block_end = (
            pause_start - 1
        )

        if block_end >= block_start:

            block_moving = moving[
                block_start:block_end + 1
            ]

            if np.any(block_moving):

                first_motion = (
                    block_start
                    + np.flatnonzero(
                        block_moving
                    )[0]
                )

                # Same as training:
                # include the following neutral hold.
                motion_blocks.append(
                    (
                        first_motion,
                        pause_end,
                    )
                )

        previous_pause_end = (
            pause_end
        )

    # Possible final motion block.
    block_start = (
        previous_pause_end + 1
    )

    block_end = (
        len(t_segment) - 1
    )

    if block_end >= block_start:

        block_moving = moving[
            block_start:block_end + 1
        ]

        if np.any(block_moving):

            first_motion = (
                block_start
                + np.flatnonzero(
                    block_moving
                )[0]
            )

            last_motion = (
                block_start
                + np.flatnonzero(
                    block_moving
                )[-1]
            )

            motion_blocks.append(
                (
                    first_motion,
                    last_motion,
                )
            )

    # ----------------------------------------------------------
    # 5. Construct pattern-relative encoder positions
    # ----------------------------------------------------------

    relative_positions = np.full_like(
        absolute_positions,
        np.nan,
        dtype=float,
    )

    for start_index, end_index in motion_blocks:
        reference_index = start_index

        reference_position = (
            absolute_positions[
                :,
                reference_index,
            ].copy()
        )

        relative_positions[
            :,
            start_index:end_index + 1
        ] = (
            absolute_positions[
                :,
                start_index:end_index + 1
            ]
            - reference_position[:, None]
        )

    print(
        f"Detected {len(motion_blocks)} "
        "coupled Motor-DT motion patterns "
        "for pattern-relative encoder replay."
    )

    return relative_positions, motion_blocks

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

    # The model should consider the motor as moving for a short time after a target change.
    movement_memory_s = EVENT_TAIL_S
    target_based_is_moving = (
        (np.abs(target_delta) > 0.0)
        & (time_since_target_change < movement_memory_s)
    ).astype(float)

    features = np.column_stack([
        target,
        abs_target,
        target_delta,
        abs_target_delta,
        target_direction,
        time_since_target_change,
        target_based_is_moving,
    ])

    if features.shape[1] != 7:
        raise RuntimeError(
            f"Expected 7 motor-only features, but created "
            f"{features.shape[1]}."
        )

    return features


# def build_instrument_current_features(t, target):
#     """
#     Build the same 7 command-only features as used during
#     instrument current training.

#     Feature order:
#     1. target
#     2. abs_target
#     3. target_delta
#     4. abs_target_delta
#     5. target_direction
#     6. time_after_change
#     7. target_based_is_moving
#     """
#     command_change_threshold = 1e-5

#     abs_target = np.abs(target)

#     instant_target_delta = np.zeros_like(target)
#     instant_target_delta[1:] = target[1:] - target[:-1]

#     target_delta = np.zeros_like(target)
#     last_delta = 0.0

#     for sample_index in range(len(target)):
#         if abs(instant_target_delta[sample_index]) > command_change_threshold:
#             last_delta = instant_target_delta[sample_index]

#         target_delta[sample_index] = last_delta

#     abs_target_delta = np.abs(target_delta)
#     target_direction = np.sign(target_delta)

#     time_after_change = time_since_signal_change(
#         t,
#         target,
#         threshold=command_change_threshold,
#     )

#     movement_memory_s = 1.2
#     target_based_is_moving = (
#         (np.abs(target_delta) > command_change_threshold)
#         & (time_after_change < movement_memory_s)
#     ).astype(float)

#     return np.column_stack([
#         target,
#         abs_target,
#         target_delta,
#         abs_target_delta,
#         target_direction,
#         time_after_change,
#         target_based_is_moving,
#     ]), target_based_is_moving


# def load_instrument_current_model(model_file):
#     model_file = Path(model_file).expanduser()

#     if not model_file.exists():
#         raise RuntimeError(f"Missing Instrument Current DT model: {model_file}")

#     model_package = joblib.load(model_file)

#     expected_features = [
#         "target",
#         "abs_target",
#         "target_delta",
#         "abs_target_delta",
#         "target_direction",
#         "time_after_change",
#         "target_based_is_moving",
#     ]

#     feature_names = model_package.get("feature_names")

#     if feature_names != expected_features:
#         raise RuntimeError(
#             "The loaded model does not use the expected command-only features.\n"
#             f"Expected: {expected_features}\n"
#             f"Found:    {feature_names}"
#         )

#     return model_package

# def full_setup_current_model_file(active_dof):
#     return (
#         DEFAULT_FULL_SETUP_MODEL_DIR
#         / f"dof{active_dof}"
#         / "models"
#         / f"instrument_current_dt_dof{active_dof}_all_motors.pkl"
#     )

def predict_segment_offline(
    t_segment,
    raw_commands,
    motor_targets,
    motion_blocks,
    motor_models,
    coupling_mode,
):
    raw_commands = np.asarray(
        raw_commands,
        dtype=float,
    )

    motor_targets = np.asarray(
        motor_targets,
        dtype=float,
    )

    predicted_positions = np.full_like(
        raw_commands,
        np.nan,
        dtype=float,
    )

    predicted_currents = np.full_like(
        raw_commands,
        np.nan,
        dtype=float,
    )

    if coupling_mode == "motor_only":
        commanded_target = relative_commands_to_target(
            raw_commands
        )

        for motor_index in range(4):
            motor_features = build_motor_dt_features(
                t_segment,
                commanded_target[motor_index],
            )

            package = motor_models[motor_index]

            predicted_positions[motor_index] = (
                package["encoder_model"].predict(
                    motor_features
                )
            )

            predicted_currents[motor_index] = (
                package["current_model"].predict(
                    motor_features
                )
            )

    else:
        commanded_target = np.full_like(
            motor_targets,
            np.nan,
            dtype=float,
        )

        for start_index, end_index in motion_blocks:
            pattern_slice = slice(
                start_index,
                end_index + 1,
            )

            motor_target_reference = motor_targets[
                :,
                start_index,
            ].copy()

            coupled_features_run = (
                build_coupled_motor_dt_features(
                    t_segment=t_segment,
                    absolute_motor_targets=motor_targets,
                    motor_target_reference=motor_target_reference,
                )
            )

            pattern_features = coupled_features_run[
                pattern_slice
            ]

            commanded_target[
                :,
                pattern_slice,
            ] = (
                motor_targets[
                    :,
                    pattern_slice,
                ]
                - motor_target_reference[:, np.newaxis]
            )

            for motor_index in range(4):
                package = motor_models[motor_index]

                if (
                    package.get("feature_schema")
                    != "all_motor_targets_v1"
                ):
                    raise RuntimeError(
                        f"Motor {motor_index} model uses feature schema "
                        f"{package.get('feature_schema')!r}; expected "
                        "'all_motor_targets_v1'."
                    )

                predicted_positions[
                    motor_index,
                    pattern_slice,
                ] = package["encoder_model"].predict(
                    pattern_features
                )

                predicted_currents[
                    motor_index,
                    pattern_slice,
                ] = package["current_model"].predict(
                    pattern_features
                )

    return (
        commanded_target,
        predicted_positions,
        predicted_currents,
    )

def default_output_file(input_file):
    input_file = Path(input_file)
    name = input_file.name

    if name.endswith("_ros_log.jsonl"):
        output_name = name.replace("_ros_log.jsonl", "_motor_dt_replay.jsonl")
    else:
        output_name = input_file.stem + "_motor_dt_replay.jsonl"

    return input_file.with_name(output_name)


def replay_motor_digital_twin(
    input_file,
    output_file,
    coupling_mode,
    active_dof=None,
    current_filter_window_size=5,
):
    input_file = Path(input_file).expanduser()
    output_file = Path(output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading log: {input_file}")
    rows = load_motor_log(input_file, coupling_mode=coupling_mode)

    print("Splitting into segments...")
    segments = split_into_segments(rows)

    print(f"Found {len(segments)} valid segments.")

    if coupling_mode == "motor_only":
        position_model_dir = DEFAULT_MOTOR_ONLY_MODEL_DIR

    elif coupling_mode == "gearbox_only":
        position_model_dir = DEFAULT_GEARBOX_MODEL_DIR

    elif coupling_mode == "full_setup":
        # Full-system Motor DT packages contain encoder and current models.
        position_model_dir = DEFAULT_FULL_SETUP_MODEL_DIR

    else:
        raise RuntimeError(
            f"Unsupported coupling mode: {coupling_mode}"
        )

    print(f"Loading Motor DT models from: {position_model_dir}")
    motor_models = load_motor_dt_models(position_model_dir)
    
        
    written_rows = 0

    with open(output_file, "w") as f:
        for segment in segments:
            task_label = segment["task_label"]
            info = segment["info"]

            (
                t_global,
                t_segment,
                raw_commands,
                motor_targets,
                instrument_dof_commands,
                measured_positions,
                raw_currents,
            ) = segment_to_arrays(
                segment,
                coupling_mode=coupling_mode,
            )

            filtered_currents = np.vstack([
                causal_moving_average(
                    raw_currents[motor_index],
                    window_size=current_filter_window_size,
                )
                for motor_index in range(4)
            ])

            motion_blocks = []

            if coupling_mode != "motor_only":
                (
                    measured_positions,
                    motion_blocks,
                ) = make_coupled_pattern_relative_positions(
                    t_segment=t_segment,
                    instrument_commands=instrument_dof_commands,
                    absolute_positions=measured_positions,
                    active_dof=active_dof,
                )

                if not motion_blocks:
                    raise RuntimeError(
                        "No DOF motion patterns detected for coupled Motor-DT replay."
                    )
            (
                commanded_target,
                predicted_positions,
                predicted_currents,
            ) = predict_segment_offline(
                t_segment=t_segment,
                raw_commands=raw_commands,
                motor_targets=motor_targets,
                motion_blocks=motion_blocks,
                motor_models=motor_models,
                coupling_mode=coupling_mode,
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
                    "coupling_mode": coupling_mode,
                    "active_dof": int(active_dof) if active_dof is not None else None,

                    "raw_commands": safe_list(raw_commands[:, sample_index]),
                    "offline_commanded_target": safe_list(commanded_target[:, sample_index]),
                    "logged_motor_target_positions": safe_list(
                        motor_targets[:, sample_index]
                    ),

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
        description="Replay coupling-specific Motor DT models offline on a ROS log."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input *_ros_log.jsonl file."
    )

    parser.add_argument(
        "--coupling-mode",
        required=True,
        choices=["motor_only", "gearbox_only", "full_setup"],
        help="Detected physical coupling configuration.",
    )

    parser.add_argument(
        "--dof",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Active instrument DOF, stored as replay metadata.",
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
        coupling_mode=args.coupling_mode,
        active_dof=args.dof,
        output_file=output_file,
        current_filter_window_size=args.current_filter_window_size,
    )


if __name__ == "__main__":
    main()