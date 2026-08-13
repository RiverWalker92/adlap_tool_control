#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


TEST_DESCRIPTIONS = {
    "idle_baseline": "No commanded motion; position stability and baseline current of all motors.",
    "single_step_small": "Single relative motor command of +100 pulses, followed by settling.",
    "single_step_medium": "Single relative motor command of +300 pulses, followed by settling.",
    "single_step_large": "Single relative motor command of +600 pulses, followed by settling.",
    "back_and_forth_small": "Positive and negative relative motor command of ±100 pulses.",
    "back_and_forth_medium": "Positive and negative relative motor command of ±300 pulses.",
    "reversal_medium": "Short direction reversals using ±300 pulse commands.",
    "cyclic_medium": "Repeated forward/backward motor motion using ±300 pulses.",
}

MOTOR_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}

FINAL_HOLD_DURATION_S = {
    "idle_baseline": 3.0,

    "back_and_forth_small": 0.5,
    "back_and_forth_medium": 0.5,
    "back_and_forth_large": 0.5,

    "reversal_medium": 0.5,
    "cyclic_medium": 0.5,
    "full_turn_single_3x": 1.5,

    # Alleen behouden wanneer deze oude testnamen echt voorkomen:
    "single_step_small": 0.5,
    "single_step_medium": 0.5,
    "single_step_large": 0.5,
}

COMMAND_CHANGE_THRESHOLD = 1e-5
MOTION_MEMORY_S = 1.2

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

def float_or_none(value):
    if value is None:
        return None
    return float(value)

def load_motor_log(file_path):
    rows = []

    current_task_label = None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if data.get("task_label") is not None:
                current_task_label = data.get("task_label")

            t_value = get_time_value(data)
            ros_timestamp = data.get("ros_timestamp")
            positions = data.get("measured_motor_positions")

            currents = data.get("measured_currents")
            commands = data.get("commanded_motor_positions")

            motor_targets = data.get(
                "motor_target_positions"
            )

            instrument_commands = data.get(
                "commanded_instrument_angles"
            )
            if motor_targets is None or len(motor_targets) < 4:
                motor_targets = [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]

            if (
                instrument_commands is None
                or len(instrument_commands) < 4
            ):
                instrument_commands = [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            
            predicted_positions = data.get("predicted_motor_positions")
            predicted_currents = data.get("predicted_motor_currents")

            measured_positions_timestamp = data.get("measured_motor_positions_timestamp")
            measured_currents_timestamp = data.get("measured_currents_timestamp")
            commanded_motor_positions_timestamp = data.get("commanded_motor_positions_timestamp")
            motor_target_positions_timestamp = data.get("motor_target_positions_timestamp")
            predicted_positions_timestamp = data.get("predicted_motor_positions_timestamp")
            predicted_currents_timestamp = data.get("predicted_motor_currents_timestamp")

            # motor_dt_state_timestamp = data.get("motor_dt_state_timestamp")

            if predicted_positions is not None and len(predicted_positions) >= 4:
                predicted_positions = [float(x) for x in predicted_positions[:4]]
            else:
                predicted_positions = None

            if predicted_currents is not None and len(predicted_currents) >= 4:
                predicted_currents = [float(x) for x in predicted_currents[:4]]
            else:
                predicted_currents = None
            
            # motor_dt_state = data.get("motor_dt_state")

            # if motor_dt_state is not None and len(motor_dt_state) >= 24:
            #     motor_dt_state = [float(x) for x in motor_dt_state]
            # else:
            #     motor_dt_state = None
                
            if t_value is None or positions is None or currents is None:
                continue

            if len(positions) < 4 or len(currents) < 4:
                continue

            if commands is None or len(commands) < 4:
                commands = [0, 0, 0, 0]

            rows.append({
                "time": float(t_value),
                "ros_timestamp": float_or_none(ros_timestamp),
                "task_label": current_task_label,
                "positions": [float(x) for x in positions[:4]],
                "currents": [float(x) for x in currents[:4]],
                "commands": [float(x) for x in commands[:4]],
                "motor_targets": [
                    float(x) for x in motor_targets[:4]
                ],
                "instrument_commands": [
                    float(x) for x in instrument_commands[:4]
                ],
                "predicted_positions": predicted_positions,
                "predicted_currents": predicted_currents,
                "measured_positions_timestamp": float_or_none(measured_positions_timestamp),
                "measured_currents_timestamp": float_or_none(measured_currents_timestamp),
                "commanded_motor_positions_timestamp": float_or_none(commanded_motor_positions_timestamp),
                "motor_target_positions_timestamp": float_or_none(motor_target_positions_timestamp),
                "predicted_positions_timestamp": float_or_none(predicted_positions_timestamp),
                "predicted_currents_timestamp": float_or_none(predicted_currents_timestamp),

                # Optional/debug only
                # "motor_dt_state": motor_dt_state,
                # "motor_dt_state_timestamp": float_or_none(motor_dt_state_timestamp),
            })

    if not rows:
        raise RuntimeError(f"No usable motor samples found in {file_path}")

    t0 = rows[0]["time"]

    for row in rows:
        row["time"] -= t0

    return rows

def default_replay_file_from_log(log_file):
    log_file = Path(log_file)
    name = log_file.name

    if name.endswith("_ros_log.jsonl"):
        replay_name = name.replace("_ros_log.jsonl", "_motor_dt_replay.jsonl")
    else:
        replay_name = log_file.stem + "_motor_dt_replay.jsonl"

    return log_file.with_name(replay_name)


def load_replay_by_task_label(replay_file):
    replay_file = Path(replay_file).expanduser()

    if not replay_file.exists():
        return None

    replay_by_task_label = {}

    with open(replay_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            task_label = row.get("task_label")

            if task_label is None:
                continue

            replay_by_task_label.setdefault(task_label, []).append(row)

    return replay_by_task_label


def replay_rows_to_arrays(replay_rows):
    t_replay = np.array(
        [row["segment_time"] for row in replay_rows],
        dtype=float,
    )

    commanded_target = np.array(
        [row["offline_commanded_target"] for row in replay_rows],
        dtype=float,
    ).T

    predicted_positions = np.array(
        [row["offline_predicted_positions"] for row in replay_rows],
        dtype=float,
    ).T

    predicted_currents = np.array(
        [row["offline_predicted_currents"] for row in replay_rows],
        dtype=float,
    ).T

    filtered_currents = np.array(
        [row["filtered_measured_currents"] for row in replay_rows],
        dtype=float,
    ).T

    encoder_deviation = np.array(
        [row["encoder_deviation"] for row in replay_rows],
        dtype=float,
    ).T

    current_deviation = np.array(
        [row["current_deviation"] for row in replay_rows],
        dtype=float,
    ).T

    return (
        t_replay,
        commanded_target,
        predicted_positions,
        predicted_currents,
        filtered_currents,
        encoder_deviation,
        current_deviation,
    )

def relative_commands_to_target(commands):
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

def get_replay_arrays_for_segment(segment, replay_by_task_label):
    if replay_by_task_label is None:
        return None

    replay_rows = replay_by_task_label.get(
        segment["task_label"]
    )

    if replay_rows is None or len(replay_rows) < 2:
        return None

    # Motor-pattern segment:
    # task_label is already unique, so use the complete replay segment.
    if "segment_start_time_s" not in segment:
        return replay_rows_to_arrays(replay_rows)

    # DOF-pattern segment:
    # several waveform segments share the same task_label,
    # therefore slice the complete replay to this waveform window.
    start_s = float(segment["segment_start_time_s"])
    end_s = float(segment["segment_end_time_s"])

    selected_rows = [
        row
        for row in replay_rows
        if start_s
        <= float(row["segment_time"])
        <= end_s
    ]

    if len(selected_rows) < 2:
        print(
            "Warning: no matching replay samples for "
            f"DOF{segment['dof_number']} "
            f"{segment['pattern_type']} "
            f"freq={segment['frequency_factor']} "
            f"range={segment['range_factor']}"
        )
        return None

    arrays = replay_rows_to_arrays(selected_rows)

    (
        t_replay,
        commanded_target,
        predicted_positions,
        predicted_currents,
        filtered_currents,
        encoder_deviation,
        current_deviation,
    ) = arrays

    # Make every individual waveform start at t = 0.
    t_replay = t_replay - t_replay[0]

    return (
        t_replay,
        commanded_target,
        predicted_positions,
        predicted_currents,
        filtered_currents,
        encoder_deviation,
        current_deviation,
    )

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

    return [
        segment for segment in segments
        if segment["task_label"] is not None
        and segment["task_label"] not in [
            "between_trials",
            "motor_tests_finished",
            "tests_finished",
        ]
        and segment["info"]["test_type"] not in [
            "motor_tests_finished",
            "tests_finished",
            "unknown_test",
        ]
    ]

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

            # Keep the neutral hold after the waveform
            # as part of this DOF-pattern segment.
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
            "info": dict(info),

            "sequence_start_time_s": float(valid_rows[0]["time"]),
            "segment_start_time_s": float(
                segment_rows[0]["time"] - valid_rows[0]["time"]
            ),
            "segment_end_time_s": float(
                segment_rows[-1]["time"] - valid_rows[0]["time"]
            ),

            "encoder_reference_positions": [
                float(value)
                for value in valid_rows[
                    max(0, start_index - 1)
                ]["positions"][:4]
            ],

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

def list_or_nan4(value):
    if value is None:
        return [np.nan, np.nan, np.nan, np.nan]
    return value

# def list_or_nan32(value):
#     if value is None:
#         return [np.nan] * 32

#     if len(value) >= 32:
#         return value[:32]

#     if len(value) >= 24:
#         return value[:24] + [np.nan] * 8

#     return [np.nan] * 32

def zero_signal_if_available(signal):
    finite = np.isfinite(signal)

    if np.any(finite):
        signal = signal - signal[finite][0]

    return signal

def make_relative_time(rows, key, t0, fallback_key="time"):
    """
    Maakt een tijd-as relatief aan dezelfde gezamenlijke t0.
    Hierdoor blijft echte vertraging tussen signalen zichtbaar.
    """
    values = []

    for row in rows:
        value = row.get(key)

        if value is None:
            value = row.get(fallback_key)

        values.append(value)

    t = np.array(values, dtype=float)

    if t0 is not None and np.isfinite(t0):
        t = t - t0
    else:
        finite = np.isfinite(t)
        if np.any(finite):
            t = t - t[finite][0]

    return t

def make_relative_time_from_keys(rows, keys, t0, fallback_key="time"):
    values = []

    for row in rows:
        value = None

        for key in keys:
            value = row.get(key)
            if value is not None:
                break

        if value is None:
            value = row.get(fallback_key)

        values.append(value)

    t = np.array(values, dtype=float)

    if t0 is not None and np.isfinite(t0):
        t = t - t0
    else:
        finite = np.isfinite(t)
        if np.any(finite):
            t = t - t[finite][0]

    return t

def get_common_segment_t0(rows):
    """
    Kies één gezamenlijke nul-tijd voor het hele segment.

    Beste keuze:
    1. eerste command timestamp
    2. anders eerste absolute ros timestamp
    3. anders eerste log time
    """
    preferred_keys = [
        "ros_timestamp",
        "measured_positions_timestamp",
        "measured_currents_timestamp",
        "commanded_motor_positions_timestamp",
        "predicted_positions_timestamp",
        "predicted_currents_timestamp",
        "time",
    ]

    for key in preferred_keys:
        values = [
            row.get(key)
            for row in rows
            if row.get(key) is not None and np.isfinite(row.get(key))
        ]

        if values:
            return float(values[0])

    return None

def get_command_change_points(rows, motor_index, t0):
    t_values = []
    command_values = []

    previous_command = None

    for row in rows:
        commands = row.get("commands")
        timestamp = row.get("commanded_motor_positions_timestamp")

        if commands is None or timestamp is None:
            continue

        command = commands[motor_index]

        if previous_command is None or command != previous_command:
            t_values.append(timestamp - t0)
            command_values.append(command)
            previous_command = command

    return np.array(t_values, dtype=float), np.array(command_values, dtype=float)

def get_actual_command_times(segment):
    """
    Vind de unieke commandotimestamps die werkelijk binnen
    het huidige gelabelde trialsegment zijn gepubliceerd.

    Aan het begin van een segment kan de logger nog de timestamp
    van het vorige commando vasthouden. Die wordt hier verwijderd.
    """
    rows = segment["rows"]

    segment_ros_times = [
        float(row["ros_timestamp"])
        for row in rows
        if row.get("ros_timestamp") is not None
        and np.isfinite(row["ros_timestamp"])
    ]

    if not segment_ros_times:
        return np.array([], dtype=float)

    segment_start_abs = min(segment_ros_times)
    segment_end_abs = max(segment_ros_times)

    unique_command_times_abs = []
    previous_timestamp = None

    for row in rows:
        timestamp = row.get(
            "commanded_motor_positions_timestamp"
        )

        if timestamp is None or not np.isfinite(timestamp):
            continue

        timestamp = float(timestamp)

        # Negeer een commandotimestamp die vóór dit triallabel ligt.
        if timestamp < segment_start_abs - 1e-6:
            continue

        # Negeer ongeldige timestamps buiten het segment.
        if timestamp > segment_end_abs + 1e-6:
            continue

        # Dezelfde commandotimestamp staat vaak in meerdere logregels.
        if (
            previous_timestamp is None
            or abs(timestamp - previous_timestamp) > 1e-6
        ):
            unique_command_times_abs.append(timestamp)
            previous_timestamp = timestamp

    return np.asarray(
        unique_command_times_abs,
        dtype=float,
    )


def get_metric_window(segment, replay_time):
    """
    Maak het evaluatievenster op basis van de werkelijk gelogde
    commandotijden.

    Het venster bevat:
    - vanaf het eerste sequence-command;
    - alle tussenliggende commando's en holds;
    - de laatste vooraf ingestelde settlingperiode.
    """
    replay_time = np.asarray(replay_time, dtype=float)

    if replay_time.size < 2:
        return None

    test_type = segment["info"]["test_type"]

    final_hold = FINAL_HOLD_DURATION_S.get(test_type)

    if final_hold is None:
        if "segment_start_time_s" in segment:
            # Bij DOF-/instrumenttrials bevat het replaysegment de volledige
            # sequence, inclusief de afsluitende hold.
            evaluation_start = float(replay_time[0])
            evaluation_end = float(replay_time[-1])

            mask = (
                np.isfinite(replay_time)
                & (replay_time >= evaluation_start)
                & (replay_time <= evaluation_end)
            )

            if np.sum(mask) < 2:
                print(
                    f"Warning: fewer than two metric samples for "
                    f"{segment['task_label']}; skipping metrics."
                )
                return None

            command_times_abs = get_actual_command_times(segment)

            if command_times_abs.size >= 2:
                logged_command_span = float(
                    command_times_abs[-1] - command_times_abs[0]
                )
            else:
                logged_command_span = 0.0

            return {
                "start_s": evaluation_start,
                "end_s": evaluation_end,
                "mask": mask,
                "source": "full_replay_segment",
                "command_count": int(command_times_abs.size),
                "logged_command_span_s": logged_command_span,
            }

        print(
            f"Warning: no metric-window definition for "
            f"{test_type}; skipping metrics."
        )
        return None

    command_times_abs = get_actual_command_times(segment)

    if test_type == "idle_baseline":
        # De idle-test bevat één nulcommando en duurt exact 3 s.
        expected_duration = float(final_hold)

    elif command_times_abs.size >= 2:
        command_span = float(
            command_times_abs[-1]
            - command_times_abs[0]
        )

        expected_duration = (
            command_span
            + float(final_hold)
        )

    else:
        print(
            f"Warning: insufficient command timestamps for "
            f"{segment['task_label']}; skipping metrics."
        )
        return None

    # De replaytijd van ieder segment begint normaal bij nul.
    evaluation_start = float(replay_time[0])
    evaluation_end = (
        evaluation_start
        + expected_duration
    )

    available_end = float(replay_time[-1])
    tolerance_s = 0.1

    if evaluation_end > available_end + tolerance_s:
        print(
            f"Warning: replay shorter than reconstructed test for "
            f"{segment['task_label']}: "
            f"required {expected_duration:.3f} s, "
            f"available "
            f"{available_end - evaluation_start:.3f} s. "
            f"Skipping metrics."
        )
        return None

    evaluation_end = min(
        evaluation_end,
        available_end,
    )

    mask = (
        np.isfinite(replay_time)
        & (replay_time >= evaluation_start)
        & (replay_time <= evaluation_end)
    )

    if np.sum(mask) < 2:
        print(
            f"Warning: fewer than two metric samples for "
            f"{segment['task_label']}; skipping metrics."
        )
        return None

    if command_times_abs.size >= 2:
        logged_command_span = float(
            command_times_abs[-1]
            - command_times_abs[0]
        )
    else:
        logged_command_span = 0.0

    return {
        "start_s": evaluation_start,
        "end_s": evaluation_end,
        "mask": mask,
        "source": "logged_command_span_plus_final_hold",
        "command_count": int(command_times_abs.size),
        "logged_command_span_s": logged_command_span,
    }

def keep_unique_time_samples(t, data_arrays, min_dt=1e-6):
    """
    Houdt alleen samples waarbij de tijd echt veranderd is.
    Dit voorkomt dat dezelfde Motor DT prediction meerdere keren geplot wordt.
    """
    t = np.asarray(t, dtype=float)

    finite = np.isfinite(t)

    for data in data_arrays:
        finite = finite & np.all(np.isfinite(data), axis=0)

    t = t[finite]
    data_arrays = [data[:, finite] for data in data_arrays]

    if len(t) == 0:
        return t, data_arrays

    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > min_dt

    t = t[keep]
    data_arrays = [data[:, keep] for data in data_arrays]

    return t, data_arrays

def segment_to_arrays(segment):
    rows = segment["rows"]

    # Eén gezamenlijke t0 voor alle signalen in dit segment.
    # Hierdoor blijft echte vertraging tussen command, meting en prediction zichtbaar.
    t0 = get_common_segment_t0(rows)

    t_log = make_relative_time(rows, "ros_timestamp", t0)
    t_pos = make_relative_time(rows, "measured_positions_timestamp", t0)
    t_cur = make_relative_time(rows, "measured_currents_timestamp", t0)
    t_cmd = make_relative_time(rows, "commanded_motor_positions_timestamp", t0)

    # Prediction komt nu uit nette Motor DT topics.
    t_pred = make_relative_time_from_keys(
        rows,
        [
            "predicted_positions_timestamp",
            "predicted_currents_timestamp",
        ],
        t0,
    )

    positions = np.array(
        [row["positions"] for row in rows],
        dtype=float,
    ).T

    currents = np.array(
        [row["currents"] for row in rows],
        dtype=float,
    ).T

    raw_commands = np.array(
        [row["commands"] for row in rows],
        dtype=float,
    ).T

    commands_target = relative_commands_to_target(raw_commands)

    predicted_positions = np.array(
        [list_or_nan4(row.get("predicted_positions")) for row in rows],
        dtype=float,
    ).T

    predicted_currents = np.array(
        [list_or_nan4(row.get("predicted_currents")) for row in rows],
        dtype=float,
    ).T

    for motor_index in range(4):
        if positions.shape[1] > 0:
            positions[motor_index] = positions[motor_index] - positions[motor_index][0]

        predicted_positions[motor_index] = zero_signal_if_available(
            predicted_positions[motor_index]
        )

    # Verwijder dubbele Motor DT prediction samples.
    t_pred, pred_arrays = keep_unique_time_samples(
        t_pred,
        [predicted_positions, predicted_currents],
    )

    predicted_positions = pred_arrays[0]
    predicted_currents = pred_arrays[1]

    return (
        t_log,
        t_pos,
        t_cur,
        t_pred,
        t_cmd,
        commands_target,
        raw_commands,
        positions,
        currents,
        predicted_positions,
        predicted_currents,
    )

# def segment_to_motor_dt_state_arrays(segment):
#     rows = segment["rows"]

#     t = np.array([row["time"] for row in rows], dtype=float)

#     if len(t) > 0:
#         t = t - t[0]

#     state = np.array(
#         [list_or_nan32(row.get("motor_dt_state")) for row in rows],
#         dtype=float,
#     ).T

#     motor_dt = {
#         "predicted_positions": state[0:4],
#         "predicted_currents": state[4:8],
#         "raw_command": state[8:12],
#         "commanded_target": state[12:16],
#         "time_since_command_change": state[16:20],
#         "time_since_target_change": state[20:24],
#         "measured_positions": state[24:28],
#         "measured_currents": state[28:32],
#     }
#     return t, motor_dt

def get_active_motor_index(segment):
    motor_name = segment["info"]["motor_name"]

    if motor_name.startswith("m") and motor_name[1:].isdigit():
        motor_index = int(motor_name[1:])
        if 0 <= motor_index <= 3:
            return motor_index

    return None



def plot_motor_overview(rows, output_dir):
    t = np.array([row["time"] for row in rows], dtype=float)
    t = t - t[0]

    positions = np.array([row["positions"] for row in rows], dtype=float).T
    currents = np.array([row["currents"] for row in rows], dtype=float).T
    raw_commands = np.array([row["commands"] for row in rows], dtype=float).T
    commands_target = relative_commands_to_target(raw_commands)

    for motor_index in range(4):
        positions[motor_index] = positions[motor_index] - positions[motor_index][0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Motor-only overview", fontsize=16)

    for motor_index in range(4):
        axes[0].plot(
            t,
            commands_target[motor_index],
            color=MOTOR_COLORS[motor_index],
            linestyle="--",
            linewidth=1.4,
            label=f"motor {motor_index} commanded target",
        )
    axes[0].set_title("Commanded motor positions")
    axes[0].set_ylabel("Command [pulses]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8, ncol=4)

    for motor_index in range(4):
        axes[1].plot(
            t,
            positions[motor_index],
            color=MOTOR_COLORS[motor_index],
            linestyle="-",
            linewidth=1.6,
            label=f"measured motor {motor_index} encoder response",
        )

    axes[1].set_title("Relative encoder response")
    axes[1].set_ylabel("Δ position [pulses]")
    axes[1].grid(True)
    axes[1].legend(fontsize=8, ncol=4)

    for motor_index in range(4):
        axes[2].plot(
            t,
            currents[motor_index],
            color=MOTOR_COLORS[motor_index],
            linestyle="-",
            linewidth=1.2,
            label=f"measured motor {motor_index} current",
        )

    axes[2].set_title("Motor currents")
    axes[2].set_ylabel("Current [mA]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8, ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = Path(output_dir) / "motor_overview_all_tests.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {plot_path}")


def plot_segment(segment, output_dir, replay_by_task_label=None):
    info = segment["info"]
    (
        t_log,
        t_pos,
        t_cur,
        t_pred,
        t_cmd,
        commands,
        raw_commands,
        positions,
        currents,
        predicted_positions,
        predicted_currents,
    ) = segment_to_arrays(segment)
    
    replay_arrays = get_replay_arrays_for_segment(
        segment,
        replay_by_task_label,
    )

    if len(t_log) < 2:
        return

    motor_index = get_active_motor_index(segment)

    test_type = info["test_type"]
    motor_name = info["motor_name"]
    trial = info["trial"]

    description = TEST_DESCRIPTIONS.get(test_type, "")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{test_type} - {motor_name} - {trial}", fontsize=16)

    if description:
        fig.text(
            0.5,
            0.92,
            description,
            ha="center",
            fontsize=10,
        )

    if motor_index is None:
        motor_indices = range(4)
    else:
        motor_indices = [motor_index]

    rows = segment["rows"]
    t0 = get_common_segment_t0(rows)

    for idx in motor_indices:
        t_cmd_changes, actual_raw_command = get_command_change_points(rows, idx, t0)

        axes[0].step(
            t_cmd_changes,
            actual_raw_command,
            where="post",
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=f"motor {idx} actual raw command",
        )

        if replay_arrays is not None:
            (
                t_replay,
                replay_commanded_target,
                replay_predicted_positions,
                replay_predicted_currents,
                replay_filtered_currents,
                replay_encoder_deviation,
                replay_current_deviation,
            ) = replay_arrays

            axes[0].plot(
                t_replay,
                replay_commanded_target[idx],
                color="green",
                linewidth=1.3,
                label=f"motor {idx} commanded target",
            )
        else:
            axes[0].plot(
                t_pred,
                commands[idx],
                color="green",
                linewidth=1.3,
                label=f"motor {idx} online commanded target",
            )

        axes[0].plot(
            t_pos,
            positions[idx],
            color="#6BAED6",
            linewidth=1.5,
            label=f"motor {idx} measured encoder position",
        )

        if replay_arrays is not None:
            axes[0].plot(
                t_replay,
                replay_predicted_positions[idx],
                color="tab:purple",
                linestyle="--",
                linewidth=1.5,
                label=f"motor {idx} offline predicted position",
            )
        elif np.isfinite(predicted_positions[idx]).any():
            axes[0].plot(
                t_pred,
                predicted_positions[idx],
                color="tab:purple",
                linestyle="--",
                linewidth=1.5,
                label=f"motor {idx} online predicted position",
            )

    axes[0].set_title("Commanded target vs encoder response")
    axes[0].set_ylabel("Relative pulse count [pulses]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    for idx in motor_indices:
        axes[1].plot(
            t_cur,
            currents[idx],
            color="orange",
            linewidth=0.8,
            alpha=0.35,
            label=f"motor {idx} raw measured current",
        )

        if replay_arrays is not None:
            axes[1].plot(
                t_replay,
                replay_filtered_currents[idx],
                color="orange",
                linewidth=1.3,
                label=f"motor {idx} filtered measured current",
            )

            axes[1].plot(
                t_replay,
                replay_predicted_currents[idx],
                color="tab:purple",
                linestyle="--",
                linewidth=1.3,
                label=f"motor {idx} offline predicted current",
            )
        elif np.isfinite(predicted_currents[idx]).any():
            axes[1].plot(
                t_pred,
                predicted_currents[idx],
                color="tab:purple",
                linestyle="--",
                linewidth=1.2,
                label=f"motor {idx} online predicted current",
            )

    axes[1].set_title("Motor current")
    axes[1].set_ylabel("Current [mA]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_xlim(left=0)
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    filename = (
        f"{safe_name(test_type)}_"
        f"{safe_name(motor_name)}_"
        f"{safe_name(trial)}.png"
    )

    plot_path = Path(output_dir) / "segments" / filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {plot_path}")

# def plot_segment_command_debug(segment, output_dir):
#     info = segment["info"]
#     motor_index = get_active_motor_index(segment)

#     if motor_index is None:
#         return

#     t, commands, raw_commands, positions, currents, predicted_positions, predicted_currents = segment_to_arrays(segment)    
#     _, debug = segment_to_motor_dt_state_arrays(segment)

#     if len(t) < 2:
#         return

#     if not np.isfinite(debug["raw_command"][motor_index]).any():
#         return

#     test_type = info["test_type"]
#     motor_name = info["motor_name"]
#     trial = info["trial"]

#     fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
#     fig.suptitle(
#         f"Motor DT debug inputs - {test_type} - {motor_name} - {trial}",
#         fontsize=16,
#     )

#     axes[0].plot(
#         t,
#         debug["commanded_target"][motor_index],
#         color="green",
#         linewidth=1.3,
#         label="Motor DT commanded target",
#     )

#     axes[0].plot(
#         t,
#         commands[motor_index],
#         color="gray",
#         linestyle=":",
#         linewidth=1.0,
#         label="plotter reconstructed target",
#     )
#     axes[0].plot(
#         t,
#         positions[motor_index],
#         color="blue",
#         linewidth=1.4,
#         label="measured encoder",
#     )
#     if np.isfinite(debug["predicted_positions"][motor_index]).any():
#         axes[0].plot(
#             t,
#             debug["predicted_positions"][motor_index],
#             color="tab:purple",
#             linestyle="--",
#             linewidth=1.4,
#             label="predicted encoder",
#         )
#     axes[0].set_ylabel("Position [pulses]")
#     axes[0].grid(True)
#     axes[0].legend(fontsize=8)

#     axes[1].plot(
#         t,
#         debug["raw_command"][motor_index],
#         color="tab:orange",
#         linewidth=1.4,
#         label="raw command",
#     )
#     # axes[1].plot(
#     #     t_debug,
#     #     debug["effective_command"][motor_index],
#     #     color="tab:purple",
#     #     linestyle="--",
#     #     linewidth=1.4,
#     #     label="effective_command",
#     # )
#     axes[1].set_ylabel("Command [pulses]")
#     axes[1].grid(True)
#     axes[1].legend(fontsize=8)

#     axes[2].plot(
#         t,
#         debug["commanded_target"][motor_index],
#         color="green",
#         linewidth=1.4,
#         label="Motor DT commanded_target",
#     )
#     axes[2].plot(
#         t,
#         debug["time_since_command_change"][motor_index],
#         color="tab:red",
#         linewidth=1.2,
#         label="time_since_command_change",
#     )
#     axes[2].plot(
#         t,
#         debug["time_since_target_change"][motor_index],
#         color="tab:blue",
#         linestyle="--",
#         linewidth=1.2,
#         label="time_since_target_change",
#     )
#     axes[2].set_ylabel("Target / time")
#     axes[2].grid(True)
#     axes[2].legend(fontsize=8)

#     axes[3].plot(
#         t,
#         currents[motor_index],
#         color="orange",
#         linewidth=1.2,
#         label="measured current",
#     )
#     if np.isfinite(debug["predicted_currents"][motor_index]).any():
#         axes[3].plot(
#             t,
#             debug["predicted_currents"][motor_index],
#             color="tab:purple",
#             linestyle="--",
#             linewidth=1.2,
#             label="predicted current",
#         )
#     axes[3].set_ylabel("Current [mA]")
#     axes[3].set_xlabel("Time [s]")
#     axes[3].grid(True)
#     axes[3].legend(fontsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.94])

#     filename = (
#         f"{safe_name(test_type)}_"
#         f"{safe_name(motor_name)}_"
#         f"{safe_name(trial)}_debug_inputs.png"
#     )

#     plot_path = Path(output_dir) / "debug_command_inputs" / filename
#     plot_path.parent.mkdir(parents=True, exist_ok=True)

#     fig.savefig(plot_path, dpi=200)
#     plt.close(fig)

#     print(f"Saved debug plot: {plot_path}")

def plot_test_type_grid(segments, output_dir, replay_by_task_label=None):
    grouped = {}

    for segment in segments:
        info = segment["info"]
        key = (info["test_type"], info["trial"])
        grouped.setdefault(key, []).append(segment)

    for (test_type, trial), group_segments in grouped.items():
        if test_type == "idle_baseline":
            continue

        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey="row")
        fig.suptitle(f"{test_type} - {trial}", fontsize=16)
        fig.text(
            0.5,
            0.92,
            "Top: commanded motor target, measured encoder response and offline Motor DT prediction. "
            "Bottom: measured current and offline Motor DT current prediction.",
            ha="center",
            fontsize=10,
        )

        all_time_values = []
        all_position_values = []
        all_current_values = []

        # First pass: determine common axis limits
        for segment in group_segments:
            motor_index = get_active_motor_index(segment)

            if motor_index is None:
                continue

            (
                t_log,
                t_pos,
                t_cur,
                t_pred,
                t_cmd,
                commands,
                raw_commands,
                positions,
                currents,
                predicted_positions,
                predicted_currents,
            ) = segment_to_arrays(segment)

            if len(t_log) < 2:
                continue

            replay_arrays = get_replay_arrays_for_segment(
                segment,
                replay_by_task_label,
            )

            all_time_values.extend(t_pos)
            all_time_values.extend(t_cur)

            all_position_values.extend(positions[motor_index])
            all_current_values.extend(currents[motor_index])

            if replay_arrays is not None:
                (
                    t_replay,
                    replay_commanded_target,
                    replay_predicted_positions,
                    replay_predicted_currents,
                    replay_filtered_currents,
                    replay_encoder_deviation,
                    replay_current_deviation,
                ) = replay_arrays

                all_time_values.extend(t_replay)

                all_position_values.extend(replay_commanded_target[motor_index])
                all_position_values.extend(replay_predicted_positions[motor_index])

                all_current_values.extend(replay_filtered_currents[motor_index])
                all_current_values.extend(replay_predicted_currents[motor_index])

            else:
                all_time_values.extend(t_pred)

                all_position_values.extend(commands[motor_index])

                if np.isfinite(predicted_positions[motor_index]).any():
                    all_position_values.extend(predicted_positions[motor_index])

                if np.isfinite(predicted_currents[motor_index]).any():
                    all_current_values.extend(predicted_currents[motor_index])

        if all_position_values:
            position_min = float(np.nanmin(all_position_values))
            position_max = float(np.nanmax(all_position_values))
            position_margin = 0.10 * max(abs(position_min), abs(position_max), 1.0)

            position_ylim = (
                position_min - position_margin,
                position_max + position_margin,
            )
        else:
            position_ylim = None

        if all_current_values:
            current_min = float(np.nanmin(all_current_values))
            current_max = float(np.nanmax(all_current_values))
            current_margin = 0.10 * max(abs(current_min), abs(current_max), 1.0)

            current_ylim = (
                max(0.0, current_min - current_margin),
                current_max + current_margin,
            )
        else:
            current_ylim = None

        if all_time_values:
            xlim = (0.0, float(np.nanmax(all_time_values)))
        else:
            xlim = None

        # Second pass: plot each motor panel
        for segment in group_segments:
            motor_index = get_active_motor_index(segment)

            if motor_index is None:
                continue

            (
                t_log,
                t_pos,
                t_cur,
                t_pred,
                t_cmd,
                commands,
                raw_commands,
                positions,
                currents,
                predicted_positions,
                predicted_currents,
            ) = segment_to_arrays(segment)

            if len(t_log) < 2:
                continue

            replay_arrays = get_replay_arrays_for_segment(
                segment,
                replay_by_task_label,
            )

            ax_pos = axes[0, motor_index]
            ax_cur = axes[1, motor_index]

            if replay_arrays is not None:
                (
                    t_replay,
                    replay_commanded_target,
                    replay_predicted_positions,
                    replay_predicted_currents,
                    replay_filtered_currents,
                    replay_encoder_deviation,
                    replay_current_deviation,
                ) = replay_arrays

                ax_pos.plot(
                    t_replay,
                    replay_commanded_target[motor_index],
                    color="green",
                    linewidth=1.2,
                    label="offline target",
                )

            else:
                ax_pos.plot(
                    t_pred,
                    commands[motor_index],
                    color="green",
                    linewidth=1.2,
                    label="online target",
                )

            ax_pos.plot(
                t_pos,
                positions[motor_index],
                color="#6BAED6",
                linewidth=1.5,
                label="encoder response",
            )

            if replay_arrays is not None:
                ax_pos.plot(
                    t_replay,
                    replay_predicted_positions[motor_index],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.4,
                    label="offline predicted encoder",
                )

            elif np.isfinite(predicted_positions[motor_index]).any():
                ax_pos.plot(
                    t_pred,
                    predicted_positions[motor_index],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.4,
                    label="online predicted encoder",
                )

            ax_pos.set_title(f"Motor {motor_index}")
            ax_pos.set_ylabel("Relative pulse count [pulses]")

            if motor_index != 0:
                ax_pos.set_ylabel("")
                ax_cur.set_ylabel("")

            if position_ylim is not None:
                ax_pos.set_ylim(position_ylim)

            if xlim is not None:
                ax_pos.set_xlim(xlim)

            ax_pos.grid(True)
            ax_pos.legend(fontsize=8)

            ax_cur.plot(
                t_cur,
                currents[motor_index],
                color="orange",
                linewidth=0.8,
                alpha=0.35,
                label="raw current",
            )

            if replay_arrays is not None:
                ax_cur.plot(
                    t_replay,
                    replay_filtered_currents[motor_index],
                    color="orange",
                    linewidth=1.2,
                    label="filtered current",
                )

                ax_cur.plot(
                    t_replay,
                    replay_predicted_currents[motor_index],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.2,
                    label="offline predicted current",
                )

            elif np.isfinite(predicted_currents[motor_index]).any():
                ax_cur.plot(
                    t_pred,
                    predicted_currents[motor_index],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.2,
                    label="online predicted current",
                )

            ax_cur.set_xlabel("Time [s]")
            ax_cur.set_ylabel("Current [mA]")

            if current_ylim is not None:
                ax_cur.set_ylim(current_ylim)

            if xlim is not None:
                ax_cur.set_xlim(xlim)

            ax_cur.grid(True)
            ax_cur.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.90])

        filename = f"{safe_name(test_type)}_{safe_name(trial)}_grid.png"
        plot_path = Path(output_dir) / "test_type_grids" / filename
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        print(f"Saved plot: {plot_path}")

def get_metric_phase_target(
    segment,
    replay_time,
    motor_target,
):
    """
    Use the motor target for dynamic/stationary phase detection
    in both motor-only and DOF-controlled trials.
    """

    replay_time = np.asarray(
        replay_time,
        dtype=float,
    )

    motor_target = np.asarray(
        motor_target,
        dtype=float,
    )

    if (
        motor_target.ndim != 1
        or motor_target.size != replay_time.size
    ):
        return None, "missing_motor_target", "pulses"

    finite = (
        np.isfinite(replay_time)
        & np.isfinite(motor_target)
    )

    if np.sum(finite) < 2:
        return None, "missing_motor_target", "pulses"

    target_range = float(
        np.max(motor_target[finite])
        - np.min(motor_target[finite])
    )

    if (
        "segment_start_time_s" in segment
        and target_range <= 1.0
    ):
        return None, "constant_motor_target", "pulses"

    phase_source = (
        "motor_target_positions"
        if "segment_start_time_s" in segment
        else "offline_motor_target"
    )

    return (
        motor_target,
        phase_source,
        "pulses",
    )

def make_metric_zone_masks(
    time,
    commanded_target,
    full_mask,
    command_change_threshold=COMMAND_CHANGE_THRESHOLD,
    motion_memory_s=MOTION_MEMORY_S,
):
    """
    Verdeel het evaluatievenster in:
    - full: alle geldige evaluatiesamples;
    - dynamic: vanaf een targetverandering tot motion_memory_s daarna;
    - stationary: full minus dynamic.

    De classificatie wordt uitsluitend uit het commanded target afgeleid.
    """
    time = np.asarray(time, dtype=float)
    commanded_target = np.asarray(commanded_target, dtype=float)
    full_mask = np.asarray(full_mask, dtype=bool)

    valid = (
        full_mask
        & np.isfinite(time)
        & np.isfinite(commanded_target)
    )

    dynamic_mask = np.zeros(len(time), dtype=bool)

    valid_indices = np.flatnonzero(valid)

    if valid_indices.size > 0:
        first_index = valid_indices[0]
        change_indices = []

        # Als het eerste target al niet nul is, begint daar een beweging.
        if abs(commanded_target[first_index]) > command_change_threshold:
            change_indices.append(first_index)

        differences = np.abs(np.diff(commanded_target))

        detected_changes = np.flatnonzero(
            np.isfinite(differences)
            & (differences > command_change_threshold)
        ) + 1

        change_indices.extend(detected_changes.tolist())

        for change_index in change_indices:
            if not valid[change_index]:
                continue

            change_time = time[change_index]

            dynamic_mask |= (
                valid
                & (time >= change_time)
                & (time <= change_time + motion_memory_s)
            )

    stationary_mask = valid & ~dynamic_mask

    return {
        "full": valid,
        "dynamic": dynamic_mask,
        "stationary": stationary_mask,
    }


def calculate_residual_statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "samples": 0,
            "mae": None,
            "rmse": None,
            "max_abs": None,
            "bias": None,
            "sd": None,
        }

    return {
        "samples": int(values.size),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "max_abs": float(np.max(np.abs(values))),
        "bias": float(np.mean(values)),
        "sd": float(
            np.std(values, ddof=1)
            if values.size > 1
            else 0.0
        ),
    }


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(np.mean(values))


def finite_max(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(np.max(values))


def finite_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return float(np.min(values))


def percent_of_target_range(value, target_range_pulses):
    """Express a pulse error as percentage of one full target excursion."""
    if (
        value is None
        or target_range_pulses is None
        or not np.isfinite(value)
        or not np.isfinite(target_range_pulses)
        or target_range_pulses <= 1.0
    ):
        return None

    return float(100.0 * value / target_range_pulses)

def write_motor_metrics(
    segments,
    output_dir,
    replay_by_task_label=None,
):
    metrics = []

    for segment in segments:
        info = segment["info"]

        replay_arrays = get_replay_arrays_for_segment(
            segment,
            replay_by_task_label,
        )

        if replay_arrays is None:
            continue

        (
            t,
            commanded_target,
            predicted_positions,
            predicted_currents,
            filtered_currents,
            encoder_deviation,
            current_deviation,
        ) = replay_arrays

        t = np.asarray(t, dtype=float)

        if len(t) < 2:
            continue

        metric_window = get_metric_window(segment, t)

        if metric_window is None:
            continue

        full_window_mask = np.asarray(
            metric_window["mask"],
            dtype=bool,
        )

        evaluation_start = metric_window["start_s"]
        evaluation_end = metric_window["end_s"]

        motor_index = get_active_motor_index(segment)

        if motor_index is None:
            motor_indices = range(4)
        else:
            motor_indices = [motor_index]

        (
            t_log,
            t_pos,
            t_cur,
            t_pred,
            t_cmd,
            online_commands,
            raw_commands,
            measured_positions,
            raw_currents,
            online_predicted_positions,
            online_predicted_currents,
        ) = segment_to_arrays(segment)

        for idx in motor_indices:
            motor_target = np.asarray(
                commanded_target[idx],
                dtype=float,
            )

            (
                phase_target,
                phase_source,
                phase_unit,
            ) = get_metric_phase_target(
                segment=segment,
                replay_time=t,
                motor_target=motor_target,
            )

            if phase_target is None:
                print(
                    f"Warning: no usable phase command for "
                    f"{segment['task_label']}; "
                    f"skipping motor {idx} metrics."
                )
                continue

            zone_masks = make_metric_zone_masks(
                time=t,
                commanded_target=phase_target,
                full_mask=full_window_mask,
            )

            phase_differences = np.abs(np.diff(phase_target))

            phase_change_count = int(
                np.sum(
                    np.isfinite(phase_differences)
                    & (
                        phase_differences
                        > COMMAND_CHANGE_THRESHOLD
                    )
                    & full_window_mask[1:]
                )
            )

            # Use the complete evaluation window for the denominator in every
            # metric zone. This keeps full/dynamic/stationary comparable and
            # prevents a partial dynamic mask from shrinking the target range.
            full_phase_command = phase_target[
                full_window_mask & np.isfinite(phase_target)
            ]

            if phase_unit == "pulses" and full_phase_command.size >= 2:
                command_range_pulses = float(
                    np.max(full_phase_command)
                    - np.min(full_phase_command)
                )
            else:
                command_range_pulses = None

            position_error_all = np.asarray(
                encoder_deviation[idx],
                dtype=float,
            )

            filtered_current_error_all = np.asarray(
                current_deviation[idx],
                dtype=float,
            )

            filtered_current_all = np.asarray(
                filtered_currents[idx],
                dtype=float,
            )

            predicted_current_all = np.asarray(
                predicted_currents[idx],
                dtype=float,
            )

            # ------------------------------------------------------
            # Raw current op dezelfde tijdas vergelijken
            # ------------------------------------------------------
            raw_current = np.asarray(
                raw_currents[idx],
                dtype=float,
            )

            valid_raw = (
                np.isfinite(t_cur)
                & np.isfinite(raw_current)
                & (t_cur >= evaluation_start)
                & (t_cur <= evaluation_end)
            )

            valid_prediction = (
                np.isfinite(t)
                & np.isfinite(predicted_current_all)
                & full_window_mask
            )

            raw_current_times = np.array([], dtype=float)
            raw_current_eval = np.array([], dtype=float)
            raw_current_error = np.array([], dtype=float)

            if (
                np.sum(valid_raw) >= 2
                and np.sum(valid_prediction) >= 2
            ):
                t_raw_valid = t_cur[valid_raw]
                raw_valid = raw_current[valid_raw]

                t_prediction_valid = t[valid_prediction]
                predicted_valid = predicted_current_all[valid_prediction]

                overlap = (
                    (t_raw_valid >= t_prediction_valid[0])
                    & (t_raw_valid <= t_prediction_valid[-1])
                )

                if np.sum(overlap) >= 2:
                    raw_current_times = t_raw_valid[overlap]
                    raw_current_eval = raw_valid[overlap]

                    predicted_at_raw_time = np.interp(
                        raw_current_times,
                        t_prediction_valid,
                        predicted_valid,
                    )

                    raw_current_error = (
                        raw_current_eval
                        - predicted_at_raw_time
                    )

            # ------------------------------------------------------
            # Full, dynamic en stationary afzonderlijk opslaan
            # ------------------------------------------------------
            median_dt = np.median(
                np.diff(t[np.isfinite(t)])
            )

            for metric_zone, zone_mask in zone_masks.items():
                zone_mask = np.asarray(zone_mask, dtype=bool)

                position_error = position_error_all[zone_mask]
                filtered_current_error = (
                    filtered_current_error_all[zone_mask]
                )

                phase_command_eval = phase_target[zone_mask]
                filtered_current_eval = (
                    filtered_current_all[zone_mask]
                )
                predicted_current_eval = (
                    predicted_current_all[zone_mask]
                )

                position_stats = calculate_residual_statistics(
                    position_error
                )
                filtered_current_stats = (
                    calculate_residual_statistics(
                        filtered_current_error
                    )
                )

                # Zet het zonemask over op de raw-currenttijdas.
                if raw_current_times.size > 0:
                    replay_indices = np.searchsorted(
                        t,
                        raw_current_times,
                        side="right",
                    ) - 1

                    replay_indices = np.clip(
                        replay_indices,
                        0,
                        len(t) - 1,
                    )

                    raw_zone_mask = zone_mask[replay_indices]

                    raw_error_zone = raw_current_error[
                        raw_zone_mask
                    ]
                    raw_current_zone = raw_current_eval[
                        raw_zone_mask
                    ]
                else:
                    raw_error_zone = np.array([], dtype=float)
                    raw_current_zone = np.array([], dtype=float)

                raw_current_stats = calculate_residual_statistics(
                    raw_error_zone
                )

                metrics.append({
                    "prediction_source": "offline_replay",
                    "metric_window_source": metric_window["source"],
                    "metric_zone": metric_zone,

                    "test_type": info["test_type"],
                    "motor_name": info["motor_name"],
                    "trial": info["trial"],
                    "motor_index": idx,

                    "detected_command_count": (
                        metric_window["command_count"]
                    ),
                    "logged_command_span_s": (
                        metric_window["logged_command_span_s"]
                    ),

                    "evaluation_start_s": float(evaluation_start),
                    "evaluation_end_s": float(evaluation_end),
                    "evaluation_duration_s": float(
                        evaluation_end - evaluation_start
                    ),
                    "zone_duration_s": float(
                        np.sum(zone_mask) * median_dt
                    ),
                    "evaluation_samples": int(
                        np.sum(zone_mask)
                    ),

                    "phase_source": phase_source,
                    "phase_unit": phase_unit,
                    "phase_change_count": phase_change_count,

                    "command_min_pulses": (
                        finite_min(phase_command_eval)
                        if phase_unit == "pulses"
                        else None
                    ),
                    "command_max_pulses": (
                        finite_max(phase_command_eval)
                        if phase_unit == "pulses"
                        else None
                    ),
                    "command_range_pulses": command_range_pulses,

                    "instrument_command_min_rad": (
                        finite_min(phase_command_eval)
                        if phase_unit == "rad"
                        else None
                    ),
                    "instrument_command_max_rad": (
                        finite_max(phase_command_eval)
                        if phase_unit == "rad"
                        else None
                    ),

                    "position_prediction_mae_pulses": (
                        position_stats["mae"]
                    ),
                    "position_prediction_rmse_pulses": (
                        position_stats["rmse"]
                    ),
                    "position_prediction_max_abs_error_pulses": (
                        position_stats["max_abs"]
                    ),
                    "position_prediction_bias_pulses": (
                        position_stats["bias"]
                    ),
                    "position_prediction_residual_sd_pulses": (
                        position_stats["sd"]
                    ),
                    "position_prediction_nmae_percent": (
                        percent_of_target_range(
                            position_stats["mae"],
                            command_range_pulses,
                        )
                    ),
                    "position_prediction_nrmse_percent": (
                        percent_of_target_range(
                            position_stats["rmse"],
                            command_range_pulses,
                        )
                    ),
                    "position_prediction_max_abs_error_percent": (
                        percent_of_target_range(
                            position_stats["max_abs"],
                            command_range_pulses,
                        )
                    ),

                    "filtered_current_prediction_mae_mA": (
                        filtered_current_stats["mae"]
                    ),
                    "filtered_current_prediction_rmse_mA": (
                        filtered_current_stats["rmse"]
                    ),
                    "filtered_current_prediction_max_abs_error_mA": (
                        filtered_current_stats["max_abs"]
                    ),
                    "filtered_current_prediction_bias_mA": (
                        filtered_current_stats["bias"]
                    ),
                    "filtered_current_prediction_residual_sd_mA": (
                        filtered_current_stats["sd"]
                    ),

                    "raw_current_prediction_mae_mA": (
                        raw_current_stats["mae"]
                    ),
                    "raw_current_prediction_rmse_mA": (
                        raw_current_stats["rmse"]
                    ),
                    "raw_current_prediction_max_abs_error_mA": (
                        raw_current_stats["max_abs"]
                    ),
                    "raw_current_prediction_bias_mA": (
                        raw_current_stats["bias"]
                    ),
                    "raw_current_prediction_residual_sd_mA": (
                        raw_current_stats["sd"]
                    ),

                    "filtered_current_mean_mA": finite_mean(
                        filtered_current_eval
                    ),
                    "filtered_current_max_mA": finite_max(
                        filtered_current_eval
                    ),
                    "predicted_current_mean_mA": finite_mean(
                        predicted_current_eval
                    ),
                    "raw_current_mean_mA": finite_mean(
                        raw_current_zone
                    ),
                    "raw_current_max_mA": finite_max(
                        raw_current_zone
                    ),
                })

    metrics_path = Path(output_dir) / "motor_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics: {metrics_path}")

def plot_motor_position_prediction_residuals(
    segments,
    output_dir,
    replay_by_task_label=None,
):
    if replay_by_task_label is None:
        return

    grouped = {}

    for segment in segments:
        key = "full_run"
        grouped.setdefault(key, []).append(segment)

    for test_type, group_segments in grouped.items():
        motor_data = {}

        group_segments = sorted(
            group_segments,
            key=lambda segment: segment["rows"][0]["time"],
        )

        for segment in group_segments:
            motor_index = get_active_motor_index(segment)

            if motor_index is None:
                motor_indices = range(4)
            else:
                motor_indices = [motor_index]

            replay_arrays = get_replay_arrays_for_segment(
                segment,
                replay_by_task_label,
            )

            if replay_arrays is None:
                continue

            (
                t_replay,
                replay_commanded_target,
                replay_predicted_positions,
                replay_predicted_currents,
                replay_filtered_currents,
                replay_encoder_deviation,
                replay_current_deviation,
            ) = replay_arrays

            for motor_index in motor_indices:

                predicted_position = np.asarray(
                    replay_predicted_positions[motor_index],
                    dtype=float,
                )

                position_residual = np.asarray(
                    replay_encoder_deviation[motor_index],
                    dtype=float,
                )

                # encoder_deviation = measured - predicted
                measured_position = (
                    predicted_position + position_residual
                )

                t_local = np.asarray(t_replay, dtype=float)

                if len(t_local) < 2:
                    continue

                t_local = t_local - t_local[0]

                data = motor_data.setdefault(
                    motor_index,
                    {
                        "time": [],
                        "measured": [],
                        "predicted": [],
                        "residual": [],
                        "end_time": 0.0,
                    },
                )

                if data["time"]:
                    positive_dt = np.diff(t_local)
                    positive_dt = positive_dt[positive_dt > 0]

                    dt = (
                        np.median(positive_dt)
                        if len(positive_dt) > 0
                        else 0.01
                    )

                    t_local = t_local + data["end_time"] + dt

                data["time"].extend(t_local)
                data["measured"].extend(measured_position)
                data["predicted"].extend(predicted_position)
                data["residual"].extend(position_residual)
                data["end_time"] = t_local[-1]

                data["time"].append(np.nan)
                data["measured"].append(np.nan)
                data["predicted"].append(np.nan)
                data["residual"].append(np.nan)

        if not motor_data:
            continue

        sorted_motor_data = sorted(motor_data.items())

        fig, axes = plt.subplots(
            len(sorted_motor_data),
            2,
            figsize=(16, 12),
            sharex=True,
            sharey="col",
            squeeze=False,
        )

        fig.suptitle(
            f"Motor position prediction and residual: "
            f"{test_type}",
            fontsize=16,
        )

        for row_index, (motor_index, data) in enumerate(
            sorted_motor_data
        ):
            t = np.asarray(data["time"], dtype=float)
            measured = np.asarray(data["measured"], dtype=float)
            predicted = np.asarray(data["predicted"], dtype=float)
            residual = np.asarray(data["residual"], dtype=float)

            valid = (
                np.isfinite(t)
                & np.isfinite(measured)
                & np.isfinite(predicted)
                & np.isfinite(residual)
            )

            if np.sum(valid) < 2:
                continue

            mae = np.mean(np.abs(residual[valid]))
            rmse = np.sqrt(np.mean(residual[valid] ** 2))

            # Left: measured and predicted position
            axes[row_index, 0].plot(
                t,
                measured,
                linestyle="-",
                linewidth=1.3,
                color="#6BAED6",
                label="Measured encoder position",
            )

            axes[row_index, 0].plot(
                t,
                predicted,
                linestyle="--",
                linewidth=1.3,
                color="#8064A2",
                label="Predicted position",
            )

            axes[row_index, 0].set_title(
                f"Motor {motor_index}: measured vs predicted"
            )
            axes[row_index, 0].set_ylabel(
                "Relative position [pulses]"
            )
            axes[row_index, 0].grid(True)
            axes[row_index, 0].legend(fontsize=8)

            # Right: position residual
            axes[row_index, 1].plot(
                t,
                residual,
                linestyle="-",
                linewidth=1.2,
                color="#4C78A8",
                label="Position residual",
            )

            axes[row_index, 1].axhline(
                0.0,
                color="black",
                linewidth=1.0,
            )

            axes[row_index, 1].set_title(
                f"Motor {motor_index}: "
                "residual = measured - predicted"
            )
            axes[row_index, 1].set_ylabel(
                "Residual [pulses]"
            )
            axes[row_index, 1].grid(True)
            axes[row_index, 1].legend(fontsize=8)

            # axes[row_index, 1].text(
            #     0.01,
            #     0.95,
            #     f"MAE = {mae:.2f} pulses\n"
            #     f"RMSE = {rmse:.2f} pulses",
            #     transform=axes[row_index, 1].transAxes,
            #     va="top",
            #     ha="left",
            #     fontsize=8,
            #     bbox=dict(boxstyle="round", alpha=0.8),
            # )

        axes[-1, 0].set_xlabel("Elapsed time per motor [s]")
        axes[-1, 1].set_xlabel("Elapsed time per motor [s]")
        axes[-1, 0].set_xlim(left=0.0)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        plot_path = (
            Path(output_dir)
            / "prediction_residuals"
            / (
                "motor_position_prediction_residual.png"
            )
        )

        plot_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        print(
            f"Saved Motor DT position residual plot: {plot_path}"
        )

def plot_motor_current_prediction_residuals(
    segments,
    output_dir,
    replay_by_task_label=None,
):
    if replay_by_task_label is None:
        return

    grouped = {}

    for segment in segments:
        key = "full_run"
        grouped.setdefault(key, []).append(segment) 

    for test_type, group_segments in grouped.items():
        motor_data = {}

        group_segments = sorted(
            group_segments,
            key=lambda segment: segment["rows"][0]["time"],
        )

        for segment in group_segments:
            motor_index = get_active_motor_index(segment)

            if motor_index is None:
                motor_indices = range(4)
            else:
                motor_indices = [motor_index]

            replay_arrays = get_replay_arrays_for_segment(
                segment,
                replay_by_task_label,
            )

            if replay_arrays is None:
                continue

            (
                t_replay,
                replay_commanded_target,
                replay_predicted_positions,
                replay_predicted_currents,
                replay_filtered_currents,
                replay_encoder_deviation,
                replay_current_deviation,
            ) = replay_arrays

            for motor_index in motor_indices:

                measured_current = np.asarray(
                    replay_filtered_currents[motor_index],
                    dtype=float,
                )

                predicted_current = np.asarray(
                    replay_predicted_currents[motor_index],
                    dtype=float,
                )

                current_residual = np.asarray(
                    replay_current_deviation[motor_index],
                    dtype=float,
                )

                t_local = np.asarray(t_replay, dtype=float)
                if len(t_local) < 2:
                    continue

                t_local = t_local - t_local[0]

                data = motor_data.setdefault(
                    motor_index,
                    {
                        "time": [],
                        "measured": [],
                        "predicted": [],
                        "residual": [],
                        "end_time": 0.0,
                    },
                )

                if data["time"]:
                    positive_dt = np.diff(t_local)
                    positive_dt = positive_dt[positive_dt > 0]

                    dt = (
                        np.median(positive_dt)
                        if len(positive_dt) > 0
                        else 0.01
                    )

                    t_local = t_local + data["end_time"] + dt

                data["time"].extend(t_local)
                data["measured"].extend(measured_current)
                data["predicted"].extend(predicted_current)
                data["residual"].extend(current_residual)
                data["end_time"] = t_local[-1]

                # Onderbreking zodat trials niet met een rechte lijn worden verbonden.
                data["time"].append(np.nan)
                data["measured"].append(np.nan)
                data["predicted"].append(np.nan)
                data["residual"].append(np.nan)

        if not motor_data:
            continue

        sorted_motor_data = sorted(motor_data.items())

        fig, axes = plt.subplots(
            len(sorted_motor_data),
            2,
            figsize=(16, 12),
            sharex=True,
            sharey="col",
            squeeze=False,
        )

        fig.suptitle(
            f"Motor current prediction and residual: "
            f"{test_type}",
            fontsize=16,
        )

        for row_index, (motor_index, data) in enumerate(
            sorted_motor_data
        ):
            t = np.asarray(data["time"], dtype=float)
            measured = np.asarray(data["measured"], dtype=float)
            predicted = np.asarray(data["predicted"], dtype=float)
            residual = np.asarray(data["residual"], dtype=float)

            valid = (
                np.isfinite(t)
                & np.isfinite(measured)
                & np.isfinite(predicted)
                & np.isfinite(residual)
            )

            if np.sum(valid) < 2:
                continue

            mae = np.mean(np.abs(residual[valid]))
            rmse = np.sqrt(np.mean(residual[valid] ** 2))

            # Left: measured and predicted current
            axes[row_index, 0].plot(
                t,
                measured,
                linestyle="-",
                linewidth=1.3,
                color="orange",
                label="Filtered measured current",
            )

            axes[row_index, 0].plot(
                t,
                predicted,
                linestyle="--",
                linewidth=1.3,
                color="tab:purple",
                label="Predicted current",
            )

            axes[row_index, 0].set_title(
                f"Motor {motor_index}: measured vs predicted"
            )
            axes[row_index, 0].set_ylabel("Current [mA]")
            axes[row_index, 0].grid(True)
            axes[row_index, 0].legend(fontsize=8)

            # Right: current residual
            axes[row_index, 1].plot(
                t,
                residual,
                linestyle="-",
                linewidth=1.2,
                color="#4C78A8",
                label="Current residual",
            )

            axes[row_index, 1].axhline(
                0.0,
                color="black",
                linewidth=1.0,
            )

            axes[row_index, 1].set_title(
                f"Motor {motor_index}: "
                "residual = measured - predicted"
            )
            axes[row_index, 1].set_ylabel(
                "Residual [mA]"
            )
            axes[row_index, 1].grid(True)
            axes[row_index, 1].legend(fontsize=8)

            # axes[row_index, 1].text(
            #     0.01,
            #     0.95,
            #     f"MAE = {mae:.2f} mA\n"
            #     f"RMSE = {rmse:.2f} mA",
            #     transform=axes[row_index, 1].transAxes,
            #     va="top",
            #     ha="left",
            #     fontsize=8,
            #     bbox=dict(boxstyle="round", alpha=0.8),
            # )

        axes[-1, 0].set_xlabel("Elapsed time per motor [s]")
        axes[-1, 1].set_xlabel("Elapsed time per motor [s]")
        axes[-1, 0].set_xlim(left=0.0)

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        plot_path = (
            Path(output_dir)
            / "prediction_residuals"
            / (
                "motor_current_prediction_residual.png"
            )
        )

        plot_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        print(
            f"Saved Motor DT current residual plot: {plot_path}"
        )

def detect_pattern_type(rows):
    for row in rows:
        info = parse_task_label(row["task_label"])

        motor_name = str(
            info.get("motor_name", "")
        ).lower()

        if re.search(r"dof[\s_-]*[1-4]", motor_name):
            return "dof"

        if re.fullmatch(r"m[0-3]", motor_name):
            return "motor"

    return "unknown"

def plot_motor_pattern_results(
    segments,
    output_dir,
    replay_by_task_label,
):
    for segment in segments:
        plot_segment(
            segment,
            output_dir,
            replay_by_task_label=replay_by_task_label,
        )

    plot_test_type_grid(
        segments,
        output_dir,
        replay_by_task_label=replay_by_task_label,
    )

    plot_motor_position_prediction_residuals(
        segments,
        output_dir,
        replay_by_task_label,
    )

    plot_motor_current_prediction_residuals(
        segments,
        output_dir,
        replay_by_task_label,
    )

def get_commanded_motors_for_dof_patterns(
    patterns,
    replay_by_task_label,
    minimum_target_range_pulses=1.0,
):
    """
    Determine which motors received a changing motor target
    during a DOF-controlled motion pattern.
    """

    active_motors = set()

    for segment in patterns.values():
        if segment is None:
            continue

        rows = segment["rows"]

        if len(rows) < 2:
            continue

        motor_targets = np.asarray(
            [
                row.get(
                    "motor_targets",
                    [np.nan, np.nan, np.nan, np.nan],
                )
                for row in rows
            ],
            dtype=float,
        )

        if (
            motor_targets.ndim != 2
            or motor_targets.shape[1] < 4
        ):
            continue

        for motor_index in range(4):
            target = motor_targets[
                :,
                motor_index,
            ]

            finite = target[
                np.isfinite(target)
            ]

            if finite.size < 2:
                continue

            target_range = float(
                np.max(finite)
                - np.min(finite)
            )

            if target_range > minimum_target_range_pulses:
                active_motors.add(motor_index)

    return sorted(active_motors)

def plot_dof_pattern_results(
    segments,
    output_dir,
    replay_by_task_label,
):
    plot_root = (Path(output_dir))
    plot_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Group segments by:
    # DOF + frequency + range
    #
    # Within each group:
    #     sinusoid
    #     triangle
    # ----------------------------------------------------------

    grouped = {}

    for segment in segments:
        key = (
            int(segment["dof_number"]),
            float(segment["frequency_factor"]),
            float(segment["range_factor"]),
        )

        grouped.setdefault(
            key,
            {},
        )[segment["pattern_type"]] = segment

    # ----------------------------------------------------------
    # One figure per:
    # DOF + frequency + range + commanded motor
    #
    # Columns:
    #   left  = sinusoid
    #   right = triangle
    #
    # Rows:
    #   0 = DOF command
    #   1 = position
    #   2 = position residual
    #   3 = current
    #   4 = current residual
    # ----------------------------------------------------------

    for (
        dof_number,
        frequency,
        range_factor,
    ), patterns in sorted(grouped.items()):

        dof_dir = plot_root
        dof_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Only generate plots for motors that actually
        # received a motor target during this DOF pattern.
        active_motors = get_commanded_motors_for_dof_patterns(
            patterns,
            replay_by_task_label,
        )

        if not active_motors:
            print(
                f"Warning: no commanded motors detected for "
                f"DOF{dof_number}, "
                f"freq x{frequency:g}, "
                f"range x{range_factor:g}"
            )
            continue

        for motor_index in active_motors:

            # sharey="row":
            # sinusoid and triangle use exactly the same
            # y-axis limits for each type of signal.
            fig, axes = plt.subplots(
                5,
                2,
                figsize=(16, 13),
                sharey="row",
                squeeze=False,
            )

            fig.suptitle(
                (
                    f"Motor DT - DOF{dof_number} - "
                    f"Motor {motor_index} - "
                    f"frequency x{frequency:g} - "
                    f"range x{range_factor:g}"
                ),
                fontsize=15,
            )

            waveform_columns = [
                (
                    "Sinusoid",
                    patterns.get("sinusoid"),
                ),
                (
                    "Triangle",
                    patterns.get("triangle"),
                ),
            ]

            for column_index, (
                waveform_name,
                segment,
            ) in enumerate(waveform_columns):

                # --------------------------------------------------
                # Missing waveform
                # --------------------------------------------------

                if segment is None:
                    for row_index in range(5):
                        axes[
                            row_index,
                            column_index,
                        ].axis("off")

                    continue

                rows = segment["rows"]
                dof_index = segment["dof_index"]

                # --------------------------------------------------
                # Raw log data
                # --------------------------------------------------

                t_raw = np.asarray(
                    [
                        row["time"]
                        for row in rows
                    ],
                    dtype=float,
                )

                t_raw = (
                    t_raw
                    - t_raw[0]
                )

                dof_command = np.asarray(
                    [
                        row[
                            "instrument_commands"
                        ][dof_index]
                        for row in rows
                    ],
                    dtype=float,
                )

                raw_motor_target = np.asarray(
                    [
                        row["motor_targets"][
                            motor_index
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                finite_target = np.flatnonzero(
                    np.isfinite(raw_motor_target)
                )

                if finite_target.size > 0:
                    motor_target_reference = float(
                        raw_motor_target[finite_target[0]]
                    )

                    raw_motor_target = (
                        raw_motor_target
                        - motor_target_reference
                    )

                raw_position = np.asarray(
                    [
                        row["positions"][
                            motor_index
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                # Position is shown relative to the
                # beginning of this individual pattern.
                encoder_reference = float(
                    segment["encoder_reference_positions"][motor_index]
                )

                raw_position = (
                    raw_position
                    - encoder_reference
                )

                raw_current = np.asarray(
                    [
                        row["currents"][
                            motor_index
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                # --------------------------------------------------
                # Offline Motor DT replay
                # --------------------------------------------------

                replay_arrays = (
                    get_replay_arrays_for_segment(
                        segment,
                        replay_by_task_label,
                    )
                )

                # ==================================================
                # ROW 0 — DOF COMMAND
                # ==================================================

                ax_command = axes[
                    0,
                    column_index,
                ]

                ax_command.set_title(
                    waveform_name
                )

                ax_command.plot(
                    t_raw,
                    dof_command,
                    color="black",
                    linewidth=1.4,
                    label=(
                        f"DOF{dof_number} command"
                    ),
                )

                ax_command.set_ylabel(
                    "Command [rad]"
                )

                ax_command.grid(True)
                ax_command.legend(
                    fontsize=8
                )

                # ==================================================
                # ROW 1 — POSITION
                # ==================================================

                ax_position = axes[
                    1,
                    column_index,
                ]

                ax_position.plot(
                    t_raw,
                    raw_motor_target,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    label="Logged motor target",
                )

                ax_position.plot(
                    t_raw,
                    raw_position,
                    color="#6BAED6",
                    linewidth=1.5,
                    label=(
                        "Measured encoder position"
                    ),
                )

                # ==================================================
                # ROW 3 — RAW CURRENT
                # ==================================================

                ax_current = axes[
                    3,
                    column_index,
                ]

                ax_current.plot(
                    t_raw,
                    raw_current,
                    color="orange",
                    linewidth=0.8,
                    alpha=0.30,
                    label="Raw measured current",
                )

                # --------------------------------------------------
                # Offline prediction + residuals
                # --------------------------------------------------

                if replay_arrays is not None:
                    (
                        t_replay,
                        replay_commanded_target,
                        replay_predicted_positions,
                        replay_predicted_currents,
                        replay_filtered_currents,
                        replay_encoder_deviation,
                        replay_current_deviation,
                    ) = replay_arrays

                    # ==============================================
                    # ROW 1 — POSITION PREDICTION
                    # ==============================================

                    ax_position.plot(
                        t_replay,
                        replay_predicted_positions[
                            motor_index
                        ],
                        color="tab:purple",
                        linestyle="--",
                        linewidth=1.5,
                        label=(
                            "Predicted encoder position"
                        ),
                    )

                    # ==============================================
                    # ROW 2 — POSITION RESIDUAL
                    # ==============================================

                    ax_position_residual = axes[
                        2,
                        column_index,
                    ]

                    ax_position_residual.plot(
                        t_replay,
                        replay_encoder_deviation[
                            motor_index
                        ],
                        color="#4C78A8",
                        linewidth=1.2,
                        label="Position residual",
                    )

                    ax_position_residual.axhline(
                        0.0,
                        color="black",
                        linewidth=1.0,
                    )

                    # ==============================================
                    # ROW 3 — CURRENT
                    # ==============================================

                    ax_current.plot(
                        t_replay,
                        replay_filtered_currents[
                            motor_index
                        ],
                        color="orange",
                        linewidth=1.4,
                        label=(
                            "Filtered measured current"
                        ),
                    )

                    ax_current.plot(
                        t_replay,
                        replay_predicted_currents[
                            motor_index
                        ],
                        color="tab:purple",
                        linestyle="--",
                        linewidth=1.4,
                        label="Predicted current",
                    )

                    # ==============================================
                    # ROW 4 — CURRENT RESIDUAL
                    # ==============================================

                    ax_current_residual = axes[
                        4,
                        column_index,
                    ]

                    ax_current_residual.plot(
                        t_replay,
                        replay_current_deviation[
                            motor_index
                        ],
                        color="#4C78A8",
                        linewidth=1.2,
                        label="Current residual",
                    )

                    ax_current_residual.axhline(
                        0.0,
                        color="black",
                        linewidth=1.0,
                    )

                # --------------------------------------------------
                # Axis labels
                # --------------------------------------------------

                axes[
                    1,
                    column_index,
                ].set_ylabel(
                    "Position [pulses]"
                )

                axes[
                    2,
                    column_index,
                ].set_ylabel(
                    "Position residual [pulses]"
                )

                axes[
                    3,
                    column_index,
                ].set_ylabel(
                    "Current [mA]"
                )

                axes[
                    4,
                    column_index,
                ].set_ylabel(
                    "Current residual [mA]"
                )

                axes[
                    4,
                    column_index,
                ].set_xlabel(
                    "Time [s]"
                )

                # --------------------------------------------------
                # Grid + legends
                # --------------------------------------------------

                for row_index in range(
                    1,
                    5,
                ):
                    axes[
                        row_index,
                        column_index,
                    ].grid(True)

                    handles, labels = (
                        axes[
                            row_index,
                            column_index,
                        ].get_legend_handles_labels()
                    )

                    if handles:
                        axes[
                            row_index,
                            column_index,
                        ].legend(
                            fontsize=8
                        )

            # ------------------------------------------------------
            # Keep x-axis beginning at zero
            # ------------------------------------------------------

            for row_index in range(5):
                for column_index in range(2):
                    if (
                        axes[
                            row_index,
                            column_index,
                        ].axison
                    ):
                        axes[
                            row_index,
                            column_index,
                        ].set_xlim(
                            left=0.0
                        )

            plt.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.96,
                ]
            )

            filename = (
                f"dof_{dof_number}_"
                f"motor_{motor_index}_"
                f"freq_"
                f"{safe_name(f'{frequency:g}')}_"
                f"range_"
                f"{safe_name(f'{range_factor:g}')}"
                f".png"
            )

            plot_path = (
                dof_dir
                / filename
            )

            fig.savefig(
                plot_path,
                dpi=200,
            )

            plt.close(fig)

            print(
                "Saved DOF-pattern plot: "
                f"{plot_path}"
            )

def plot_dof_overall_residuals(
    segments,
    output_dir,
    replay_by_task_label,
):
    if replay_by_task_label is None or not segments:
        return

    dof_number = segments[0]["dof_number"]

    # Determine which motors were actually commanded.
    all_patterns = {
        f"{segment['pattern_type']}_{segment['pattern_index']}": segment
        for segment in segments
    }

    active_motors = get_commanded_motors_for_dof_patterns(
        all_patterns,
        replay_by_task_label,
    )

    if not active_motors:
        return

    segments = sorted(
        segments,
        key=lambda segment: segment["segment_start_time_s"],
    )

    output_dir = (
        Path(output_dir)
        / "overall_residuals"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for motor_index in active_motors:

        time_all = []
        measured_position_all = []
        predicted_position_all = []
        position_residual_all = []

        measured_current_all = []
        predicted_current_all = []
        current_residual_all = []

        elapsed_time = 0.0

        for segment in segments:
            replay_arrays = get_replay_arrays_for_segment(
                segment,
                replay_by_task_label,
            )

            if replay_arrays is None:
                continue

            (
                t,
                commanded_target,
                predicted_positions,
                predicted_currents,
                filtered_currents,
                encoder_deviation,
                current_deviation,
            ) = replay_arrays

            t = np.asarray(t, dtype=float)

            if len(t) < 2:
                continue

            t = t - t[0]

            if time_all:
                positive_dt = np.diff(t)
                positive_dt = positive_dt[positive_dt > 0]

                dt = (
                    float(np.median(positive_dt))
                    if positive_dt.size > 0
                    else 0.01
                )

                t = t + elapsed_time + dt

            predicted_position = np.asarray(
                predicted_positions[motor_index],
                dtype=float,
            )

            position_residual = np.asarray(
                encoder_deviation[motor_index],
                dtype=float,
            )

            measured_position = (
                predicted_position
                + position_residual
            )

            measured_current = np.asarray(
                filtered_currents[motor_index],
                dtype=float,
            )

            predicted_current = np.asarray(
                predicted_currents[motor_index],
                dtype=float,
            )

            current_residual = np.asarray(
                current_deviation[motor_index],
                dtype=float,
            )

            time_all.extend(t)
            measured_position_all.extend(measured_position)
            predicted_position_all.extend(predicted_position)
            position_residual_all.extend(position_residual)

            measured_current_all.extend(measured_current)
            predicted_current_all.extend(predicted_current)
            current_residual_all.extend(current_residual)

            elapsed_time = float(t[-1])

            # Visual separation between individual patterns
            time_all.append(np.nan)

            measured_position_all.append(np.nan)
            predicted_position_all.append(np.nan)
            position_residual_all.append(np.nan)

            measured_current_all.append(np.nan)
            predicted_current_all.append(np.nan)
            current_residual_all.append(np.nan)

        t = np.asarray(time_all, dtype=float)

        # ======================================================
        # POSITION
        # ======================================================

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 5),
            sharex=True,
        )

        fig.suptitle(
            f"Motor DT - DOF{dof_number} - Motor {motor_index} - position",
            fontsize=15,
        )

        axes[0].plot(
            t,
            measured_position_all,
            color="#6BAED6",
            linewidth=1.4,
            label="Measured encoder position",
        )

        axes[0].plot(
            t,
            predicted_position_all,
            color="tab:purple",
            linestyle="--",
            linewidth=1.4,
            label="Predicted encoder position",
        )

        axes[0].set_title(
            "Encoder position: measured vs predicted"
        )
        axes[0].set_ylabel(
            "Relative position [pulses]"
        )
        axes[0].set_xlabel("Time [s]")
        axes[0].grid(True)
        axes[0].legend(fontsize=8)

        axes[1].plot(
            t,
            position_residual_all,
            color="#4C78A8",
            linewidth=1.2,
            label="Position residual",
        )

        axes[1].axhline(
            0.0,
            color="black",
            linewidth=1.0,
        )

        axes[1].set_title(
            "Position residual = measured - predicted"
        )
        axes[1].set_ylabel(
            "Position residual [pulses]"
        )
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True)
        axes[1].legend(fontsize=8)

        plt.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        plot_path = (
            output_dir
            / f"motor_{motor_index}_position_prediction_residual.png"
        )

        fig.savefig(
            plot_path,
            dpi=200,
        )
        plt.close(fig)

        # ======================================================
        # CURRENT
        # ======================================================

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 5),
            sharex=True,
        )

        fig.suptitle(
            f"Motor DT - DOF{dof_number} - Motor {motor_index} - current",
            fontsize=15,
        )

        axes[0].plot(
            t,
            measured_current_all,
            color="orange",
            linewidth=1.4,
            label="Filtered measured current",
        )

        axes[0].plot(
            t,
            predicted_current_all,
            color="tab:purple",
            linestyle="--",
            linewidth=1.4,
            label="Predicted current",
        )

        axes[0].set_title(
            "Motor current: measured vs predicted"
        )
        axes[0].set_ylabel(
            "Current [mA]"
        )
        axes[0].set_xlabel("Time [s]")
        axes[0].grid(True)
        axes[0].legend(fontsize=8)

        axes[1].plot(
            t,
            current_residual_all,
            color="#4C78A8",
            linewidth=1.2,
            label="Current residual",
        )

        axes[1].axhline(
            0.0,
            color="black",
            linewidth=1.0,
        )

        axes[1].set_title(
            "Current residual = measured - predicted"
        )
        axes[1].set_ylabel(
            "Current residual [mA]"
        )
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True)
        axes[1].legend(fontsize=8)

        plt.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        plot_path = (
            output_dir
            / f"motor_{motor_index}_current_prediction_residual.png"
        )

        fig.savefig(
            plot_path,
            dpi=200,
        )
        plt.close(fig)

def plot_motor(file_path, output_dir, replay_file=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_motor_log(file_path)
    pattern_type = detect_pattern_type(rows)

    if pattern_type == "motor":
        print("Detected motor-pattern commands.")
        segments = split_motor_pattern_segments(rows)

    elif pattern_type == "dof":
        print("Detected DOF-pattern commands.")
        segments = split_dof_pattern_segments(rows)
        print(f"Number of DOF pattern segments: {len(segments)}")

        for segment in segments:
            print(
                f"  DOF{segment['dof_number']} | "
                f"{segment['pattern_type']} | "
                f"freq x{segment['frequency_factor']} | "
                f"range x{segment['range_factor']} | "
                f"{len(segment['rows'])} samples"
            )

    else:
        raise RuntimeError(
            "Could not determine whether this log contains "
            "motor-pattern or DOF-pattern commands."
        )

    if replay_file is None:
        replay_file = default_replay_file_from_log(file_path)

    replay_by_task_label = load_replay_by_task_label(replay_file)

    if replay_by_task_label is None:
        print(f"No offline Motor DT replay found: {replay_file}")
        print("Falling back to online Motor DT predictions from ROS log.")
    else:
        print(f"Loaded offline Motor DT replay: {replay_file}")

    if pattern_type == "motor":
        plot_motor_pattern_results(
            segments,
            output_dir,
            replay_by_task_label,
        )
        plot_motor_overview(
            rows, 
            output_dir)

    elif pattern_type == "dof":
        plot_dof_pattern_results(
            segments,
            output_dir,
            replay_by_task_label,
        )
        plot_dof_overall_residuals(
            segments,
            output_dir,
            replay_by_task_label,
        )

    write_motor_metrics(
        segments,
        output_dir,
        replay_by_task_label=replay_by_task_label,
    )
        

def main():
    parser = argparse.ArgumentParser(
        description="Plot motor-only trial data."
    )

    parser.add_argument("--file", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--replay-file",
        default=None,
        help="Optional offline Motor DT replay jsonl file.",
    )

    args = parser.parse_args()

    plot_motor(
        file_path=args.file,
        output_dir=args.output_dir,
        replay_file=args.replay_file,
    )


if __name__ == "__main__":
    main()