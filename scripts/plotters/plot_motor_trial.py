#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

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
            
            predicted_positions = data.get("predicted_motor_positions")
            predicted_currents = data.get("predicted_motor_currents")

            measured_positions_timestamp = data.get("measured_motor_positions_timestamp")
            measured_currents_timestamp = data.get("measured_currents_timestamp")
            commanded_motor_positions_timestamp = data.get("commanded_motor_positions_timestamp")
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
                "predicted_positions": predicted_positions,
                "predicted_currents": predicted_currents,
                "measured_positions_timestamp": float_or_none(measured_positions_timestamp),
                "measured_currents_timestamp": float_or_none(measured_currents_timestamp),
                "commanded_motor_positions_timestamp": float_or_none(commanded_motor_positions_timestamp),
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

    replay_rows = replay_by_task_label.get(segment["task_label"])

    if replay_rows is None or len(replay_rows) < 2:
        return None

    return replay_rows_to_arrays(replay_rows)

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

    return [
        segment for segment in segments
        if segment["task_label"] is not None
        and segment["task_label"] not in ["motor_tests_finished", "tests_finished"]
        and segment["info"]["test_type"] not in ["motor_tests_finished", "tests_finished", "unknown_test"]
    ]

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


def write_motor_metrics(segments, output_dir, replay_by_task_label=None):
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
            commands,
            predicted_positions,
            predicted_currents,
            filtered_currents,
            encoder_deviation,
            current_deviation,
        ) = replay_arrays

        if len(t) < 2:
            continue

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
            position_error = encoder_deviation[idx]

            # Dit is filtered current - predicted current uit de offline replay
            filtered_current_error = current_deviation[idx]

            # Extra: raw current - predicted current
            raw_current = raw_currents[idx]
            predicted_current = predicted_currents[idx]

            valid_raw = np.isfinite(t_cur) & np.isfinite(raw_current)
            valid_pred = np.isfinite(t) & np.isfinite(predicted_current)

            if np.sum(valid_raw) >= 2 and np.sum(valid_pred) >= 2:
                t_raw_valid = t_cur[valid_raw]
                raw_current_valid = raw_current[valid_raw]

                t_pred_valid = t[valid_pred]
                predicted_current_valid = predicted_current[valid_pred]

                overlap = (
                    (t_raw_valid >= t_pred_valid[0])
                    & (t_raw_valid <= t_pred_valid[-1])
                )

                if np.sum(overlap) >= 2:
                    predicted_current_at_raw_time = np.interp(
                        t_raw_valid[overlap],
                        t_pred_valid,
                        predicted_current_valid,
                    )

                    raw_current_error = (
                        raw_current_valid[overlap]
                        - predicted_current_at_raw_time
                    )
                else:
                    raw_current_error = np.array([np.nan])
            else:
                raw_current_error = np.array([np.nan])

            metrics.append({
                "prediction_source": "offline_replay",
                "test_type": info["test_type"],
                "motor_name": info["motor_name"],
                "trial": info["trial"],
                "motor_index": idx,

                "duration_s": float(t[-1] - t[0]),

                "command_min_pulses": float(np.nanmin(commands[idx])),
                "command_max_pulses": float(np.nanmax(commands[idx])),

                "position_prediction_mae_pulses": float(np.nanmean(np.abs(position_error))),
                "position_prediction_rmse_pulses": float(np.sqrt(np.nanmean(position_error ** 2))),
                "position_prediction_max_abs_error_pulses": float(np.nanmax(np.abs(position_error))),
                
                "filtered_current_prediction_mae_mA": float(np.nanmean(np.abs(filtered_current_error))),
                "filtered_current_prediction_rmse_mA": float(np.sqrt(np.nanmean(filtered_current_error ** 2))),
                "filtered_current_prediction_max_abs_error_mA": float(np.nanmax(np.abs(filtered_current_error))),

                "raw_current_prediction_mae_mA": float(np.nanmean(np.abs(raw_current_error))),
                "raw_current_prediction_rmse_mA": float(np.sqrt(np.nanmean(raw_current_error ** 2))),
                "raw_current_prediction_max_abs_error_mA": float(np.nanmax(np.abs(raw_current_error))),
                
                "filtered_current_mean_mA": float(np.nanmean(filtered_currents[idx])),
                "filtered_current_max_mA": float(np.nanmax(filtered_currents[idx])),
                "raw_current_mean_mA": float(np.nanmean(raw_current)),
                "raw_current_max_mA": float(np.nanmax(raw_current)),
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
                continue

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

            axes[row_index, 1].text(
                0.01,
                0.95,
                f"MAE = {mae:.2f} pulses\n"
                f"RMSE = {rmse:.2f} pulses",
                transform=axes[row_index, 1].transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", alpha=0.8),
            )

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
                continue

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

            axes[row_index, 1].text(
                0.01,
                0.95,
                f"MAE = {mae:.2f} mA\n"
                f"RMSE = {rmse:.2f} mA",
                transform=axes[row_index, 1].transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", alpha=0.8),
            )

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

def plot_motor_trial(file_path, output_dir, replay_file=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_motor_log(file_path)
    segments = split_into_segments(rows)

    if replay_file is None:
        replay_file = default_replay_file_from_log(file_path)

    replay_by_task_label = load_replay_by_task_label(replay_file)

    if replay_by_task_label is None:
        print(f"No offline Motor DT replay found: {replay_file}")
        print("Falling back to online Motor DT predictions from ROS log.")
    else:
        print(f"Loaded offline Motor DT replay: {replay_file}")

    plot_motor_overview(rows, output_dir)

    for segment in segments:
        plot_segment(
            segment,
            output_dir,
            replay_by_task_label=replay_by_task_label,
        )
        # plot_segment_command_debug(segment, output_dir)

    plot_test_type_grid(
        segments,
        output_dir,
        replay_by_task_label=replay_by_task_label)

    plot_motor_position_prediction_residuals(
        segments=segments,
        output_dir=output_dir,
        replay_by_task_label=replay_by_task_label,
    )

    plot_motor_current_prediction_residuals(
        segments=segments,
        output_dir=output_dir,
        replay_by_task_label=replay_by_task_label,
    )

    write_motor_metrics(
        segments, 
        output_dir, 
        replay_by_task_label=replay_by_task_label)

    print(f"Saved motor-only plots in: {output_dir}")

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

    plot_motor_trial(
        file_path=args.file,
        output_dir=args.output_dir,
        replay_file=args.replay_file,
    )


if __name__ == "__main__":
    main()