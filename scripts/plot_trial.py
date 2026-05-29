#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import os

SEQUENCE_FOLDER = "/home/leanne/ros2_ws_roel_split/test_data/unstructured"

sequence_tasks = [
    "coupling_sequence",
    "initialization_sequence",
    "update_starting_positions",
]

setup_folder = "/home/leanne/ros2_ws_roel_split/test_data/setup_01_motors_only"
time_filters = [
    # "20260504_135",
    "20260528_133",
    # "20260527_151",
    # "20260511_102",
    # "20260511_103",

]

PLOT_MODE = "sequence"  
# opties: "idle", "duty_current", "test_type", "small_vs_medium", "sequence"


trial_numbers = [1, 2, 3]

run_id = "_".join(time_filters)

run_dir = os.path.join(setup_folder, "plots", run_id)
os.makedirs(run_dir, exist_ok=True)


test_types = [
    # "idle_baseline",
    "single_step_small",
    "single_step_medium",
    "backlash_small",
    "reversal_medium",
    "cyclic_medium",
]

test_descriptions = {
    "single_step_small": "Single relative motor step of +100 pulses, followed by settling.",
    "single_step_medium": "Single relative motor step of +300 pulses, followed by settling.",
    "backlash_small": "Positive step of +100 pulses followed by negative step of -100 pulses.",
    "reversal_medium": "Direction reversal using +300 and -300 pulse commands.",
    "cyclic_medium": "Repeated forward/backward motion using ±300 pulses for 15 cycles.",
    "idle_baseline": "No commanded motion; baseline current and position stability."
}

motor_folders = ["m1", "m2", "m3", "m4"]

def get_sequence_timestamp(file_path):
    name = os.path.basename(file_path)
    name = name.replace(".jsonl", "")

    # voorbeeld: coupling_sequence_20260508_100151
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[-2] + "_" + parts[-1]

    return "unknown_time"

def load_sequence_file(file_path):
    print(f"Loading: {file_path}")

    timestamps = []
    positions = [[], [], [], []]
    currents = [[], [], [], []]
    commands = [[], [], [], []]
    duty = [[], [], [], []]

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            timestamps.append(data["timestamp"])

            pos_data = data.get("measured_motor_positions")
            cur_data = data.get("measured_currents")
            cmd_data = data.get("commanded_motor_positions")
            duty_data = data.get("commanded_duty_cycle")

            for i in range(4):
                positions[i].append(pos_data[i] if pos_data is not None else None)
                currents[i].append(cur_data[i] if cur_data is not None else None)
                commands[i].append(cmd_data[i] if cmd_data is not None else None)
                duty[i].append(duty_data[i] if duty_data is not None else None)

    if not timestamps:
        return None

    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    for i in range(4):
        if positions[i][0] is not None:
            p0 = positions[i][0]
            positions[i] = [p - p0 if p is not None else None for p in positions[i]]

        if commands[i][0] is not None:
            c0 = commands[i][0]
            commands[i] = [c - c0 if c is not None else None for c in commands[i]]

    return timestamps, positions, currents, commands, duty, file_path

def get_sequence_files(task_name):
    folder = os.path.join(SEQUENCE_FOLDER, task_name)

    if not os.path.isdir(folder):
        print(f"No folder found: {folder}")
        return []

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".jsonl")
    ]

    files.sort()
    return files

def moving_average(values, window=11):
    """
    Symmetric moving average.
    Beter dan alleen backwards averaging, omdat de curve minder verschuift in de tijd.
    """
    result = []
    n = len(values)
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        subset = [v for v in values[start:end] if v is not None]

        if len(subset) == 0:
            result.append(None)
        else:
            result.append(sum(subset) / len(subset))

    return result

def compute_velocity(timestamps, positions, position_window=11, velocity_window=11):
    velocities = [[], [], [], []]

    for motor_index in range(4):
        pos = positions[motor_index]
        vel = [0.0]

        # 1. Smooth eerst de positie
        pos_smooth = moving_average(pos, window=position_window)

        # 2. Bereken velocity met central difference
        vel = []

        for i in range(len(timestamps)):
            if i == 0 or i == len(timestamps) - 1:
                vel.append(0.0)
                continue

            t_prev = timestamps[i - 1]
            t_next = timestamps[i + 1]
            p_prev = pos_smooth[i - 1]
            p_next = pos_smooth[i + 1]

            dt = t_next - t_prev

            if dt <= 0 or p_prev is None or p_next is None:
                vel.append(0.0)
            else:
                vel.append((p_next - p_prev) / dt)

        # 3. Smooth velocity nog een keer licht
        vel_smooth = moving_average(vel, window=velocity_window)

        velocities[motor_index] = [
            v if v is not None else 0.0
            for v in vel_smooth
        ]

    return velocities

def plot_sequence(task_name, file_path):
    data = load_sequence_file(file_path)

    if data is None:
        return

    timestamps, positions, currents, commands, duty, file_path = data
    velocities = compute_velocity(timestamps, positions,  position_window=15,
    velocity_window=15)

    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    fig.suptitle(task_name, fontsize=16, y=0.98)

    fig.text(
        0.5, 0.94,
        f"File: {os.path.basename(file_path)}",
        ha="center",
        fontsize=9
    )

    for motor_index in range(4):
        ax_pos = axes[0, motor_index]
        ax_vel = axes[1, motor_index]
        ax_cur = axes[2, motor_index]
        
        # ax_cmd = axes[3, motor_index]
        # ax_duty = axes[4, motor_index]

        ax_pos.plot(timestamps, positions[motor_index], marker="o", markersize=2)
        ax_pos.set_title(f"Motor {motor_index}")
        ax_pos.set_ylabel("Δ position [pulses]")
        ax_pos.grid(True)

        ax_vel.plot(timestamps, velocities[motor_index], marker="o", markersize=2)
        ax_vel.set_ylabel("Velocity [pulses/s]")
        ax_vel.grid(True)

        ax_cur.plot(timestamps, currents[motor_index], color="orange", marker="o", markersize=2)
        ax_cur.set_ylabel("Current [ ]")
        ax_cur.set_xlabel("Time [s]")
        ax_cur.grid(True)

        # ax_cmd.plot(timestamps, commands[motor_index], marker="o", markersize=2)
        # ax_cmd.set_ylabel("Δ command [pulses]")
        # ax_cmd.grid(True)

        # ax_duty.plot(timestamps, duty[motor_index], marker="o", markersize=2)
        # ax_duty.set_ylabel("Duty cycle")
        # ax_duty.set_xlabel("Time [s]")
        # ax_duty.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    sequence_timestamp = get_sequence_timestamp(file_path)
    sequence_plot_dir = os.path.join(SEQUENCE_FOLDER, "plots", task_name, sequence_timestamp)
    os.makedirs(sequence_plot_dir, exist_ok=True)

    plot_path = os.path.join(sequence_plot_dir, f"{task_name}_timeseries.png")

    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")

    plt.close(fig)

def plot_duty_current():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle("Current vs commanded duty cycle", fontsize=16, y=0.98)
    fig.text(
        0.5, 0.94,
        "All available sequence samples. Current depends on duty cycle, load, friction and motion state.",
        ha="center",
        fontsize=9
    )

    duty_all = [[], [], [], []]
    current_all = [[], [], [], []]

    for task_name in sequence_tasks:
        files = get_sequence_files(task_name)

        for file_path in files:
            data = load_sequence_file(file_path)

            if data is None:
                continue

            timestamps, positions, currents, commands, duty, _ = data

            for motor_index in range(4):
                for d, c in zip(duty[motor_index], currents[motor_index]):
                    if d is not None and c is not None:
                        duty_all[motor_index].append(d)
                        current_all[motor_index].append(c)

    for motor_index in range(4):
        ax = axes[motor_index]

        ax.scatter(
            duty_all[motor_index],
            current_all[motor_index],
            s=8,
            alpha=0.5
        )

        ax.set_title(f"Motor {motor_index}")
        ax.set_xlabel("Commanded duty cycle")
        ax.set_ylabel("Measured current [Pico units]")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    plot_dir = os.path.join(SEQUENCE_FOLDER, "plots", "duty_current")
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, "current_vs_duty_cycle_all_sequences.png")
    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")

    plt.close(fig)

def plot_idle_baseline(trial_number):
    base_folder = os.path.join(setup_folder, "idle_baseline", "all_motors")

    prefix = f"trial_{trial_number:02d}_"
    files = [
        f for f in os.listdir(base_folder)
        if f.endswith(".jsonl")
        and f.startswith(prefix)
        and any(time_filter in f for time_filter in time_filters)
    ]
    files.sort()

    if not files:
        print(f"No idle baseline file found for {prefix}")
        return

    file_path = os.path.join(base_folder, files[-1])
    print(f"idle_baseline: {file_path}")

    timestamps = []
    positions = [[], [], [], []]
    currents = [[], [], [], []]

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            timestamps.append(data["timestamp"])

            for i in range(4):
                positions[i].append(data["measured_motor_positions"][i])
                currents[i].append(data["measured_currents"][i])

    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("idle_baseline", fontsize=16, y=0.98)
    fig.text(
        0.5, 0.94,
        "No commanded motion; position stability and baseline current of all motors.",
        ha="center",
        fontsize=10
    )

    # normaliseer posities per motor
    for i in range(4):
        p0 = positions[i][0]
        positions[i] = [p - p0 for p in positions[i]]

 
    all_pos = [v for motor_data in positions for v in motor_data]
    all_cur = [v for motor_data in currents for v in motor_data]

    pos_min, pos_max = min(all_pos), max(all_pos)
    cur_min, cur_max = min(all_cur), max(all_cur)

    pos_margin = 0.1 * (pos_max - pos_min) if pos_max != pos_min else 1
    cur_margin = 0.1 * (cur_max - cur_min) if cur_max != cur_min else 1

    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)

        ax.plot(timestamps, positions[i], color="blue", label="Δ position")
        ax.set_ylim(pos_min - pos_margin, pos_max + pos_margin)
        ax.set_ylabel("Δ position [pulses]")

        ax2 = ax.twinx()
        ax2.plot(timestamps, currents[i], color="orange", marker="o", markersize=2, linestyle="none", label="current")
        ax2.set_ylim(cur_min - cur_margin, cur_max + cur_margin)
        ax2.set_ylabel("Current [?]")

        ax.set_title(f"Motor {i}")
        ax.set_xlabel("Time [s]")

        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
    os.makedirs(trial_folder, exist_ok=True)

    plot_path = os.path.join(
        trial_folder,
        f"idle_baseline_t{trial_number}.png"
    )
    fig.savefig(plot_path, dpi=200)

    print(f"Saved plot: {plot_path}")
    plt.close(fig)


def plot_test_type(test_type, trial_number):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)

    fig.suptitle(f"{test_type} - Trial {trial_number}", fontsize=16, y=0.98)

    description = test_descriptions.get(test_type, "")
    fig.text(0.5, 0.94, description, ha="center", fontsize=10)

    for motor_index, motor in enumerate(motor_folders):
        data = load_motor_file(test_type, motor, trial_number)

        if data is None:
            continue

        timestamps, pos, current = data

        ax_pos = axes[0, motor_index]
        ax_cur = axes[1, motor_index]

        ax_pos.plot(
            timestamps,
            pos,
            color="blue",
            linewidth=1.5,
            marker="o",
            markersize=2,
            label="motor position"
            )


        ax_pos.set_title(f"Motor {motor_index}")
        ax_pos.set_ylabel("position (relative) [pulses]")
        ax_pos.grid(True)
        ax_pos.legend(loc="best", fontsize=8)

        ax_cur.plot(
            timestamps,
            current,
            color="orange",
            linewidth=1.2,
            marker="o",
            markersize=2,
            label="current"
        )

        ax_cur.set_xlabel("Time [s]")
        ax_cur.set_ylabel("Current [?]")
        ax_cur.grid(True)
        ax_cur.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
    os.makedirs(trial_folder, exist_ok=True)

    plot_path = os.path.join(
        trial_folder,
        f"{test_type}_t{trial_number}.png"
    )

    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")

    plt.close(fig)


def load_motor_file(test_type, motor, trial_number):
    folder = os.path.join(setup_folder, test_type, motor)
    prefix = f"trial_{trial_number:02d}_"

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".jsonl")
        and f.startswith(prefix)
        and any(time_filter in f for time_filter in time_filters)
    ]
    files.sort()

    if not files:
        print(f"No file found for {test_type}/{motor}, {prefix}, filters={time_filters}")
        return None

    file_path = os.path.join(folder, files[-1])
    print(f"{test_type}/{motor}: {file_path}")

    motor_index = motor_folders.index(motor)

    timestamps = []
    pos = []
    current = []
    command = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            timestamps.append(data["timestamp"])
            pos.append(data["measured_motor_positions"][motor_index])
            current.append(data["measured_currents"][motor_index])

    t0 = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    p0 = pos[0]
    pos = [p - p0 for p in pos]

    return timestamps, pos, current

def plot_small_vs_medium(trial_number):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False)
    fig.suptitle(
        f"Single step response: small vs medium, trial {trial_number:02d}",
        fontsize=16,
        y=0.98
    )

    fig.text(
        0.5, 0.94,
        "Comparison of +100 pulse and +300 pulse relative motor commands.",
        ha="center",
        fontsize=10
    )

    for motor_index, motor in enumerate(motor_folders):
        small_data = load_motor_file("single_step_small", motor, trial_number)
        medium_data = load_motor_file("single_step_medium", motor, trial_number)

        if small_data is None or medium_data is None:
            continue

        t_small, pos_small, cur_small = small_data
        t_medium, pos_medium, cur_medium = medium_data

        ax_pos = axes[0, motor_index]
        ax_cur = axes[1, motor_index]

        ax_pos.plot(
            t_small,
            pos_small,
            color="blue",
            linewidth=1.5,
            marker="o",
            markersize=2,
            label="small position"
        )

        ax_pos.plot(
            t_medium,
            pos_medium,
            color="green",
            linewidth=1.5,
            marker="o",
            markersize=2,
            label="medium position"
        )

        ax_pos.set_title(f"Motor {motor_index}")
        ax_pos.set_ylabel("relative position [pulses]")
        ax_pos.grid(True)
        ax_pos.legend(loc="best", fontsize=8)

        ax_cur.plot(
            t_small,
            cur_small,
            color="orange",
            linewidth=1.2,
            marker="o",
            markersize=2,
            label="small current"
        )

        ax_cur.plot(
            t_medium,
            cur_medium,
            color="red",
            linewidth=1.2,
            marker="o",
            markersize=2,
            label="medium current"
        )

        ax_cur.set_xlabel("Time [s]")
        ax_cur.set_ylabel("Current [?]")
        ax_cur.grid(True)
        ax_cur.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
    os.makedirs(trial_folder, exist_ok=True)

    plot_path = os.path.join(
        trial_folder,
        f"single_step_small_vs_medium_t{trial_number}.png"
    )

    fig.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")

    plt.close(fig)



if PLOT_MODE == "idle":
    for trial_number in trial_numbers:
        plot_idle_baseline(trial_number)
    print("Idle baseline plots saved.")

elif PLOT_MODE == "test_type":
    for trial_number in trial_numbers:
        for test_type in test_types:
            plot_test_type(test_type, trial_number)
    print("Test type plots saved.")

elif PLOT_MODE == "small_vs_medium":
    for trial_number in trial_numbers:
        plot_small_vs_medium(trial_number)
    print("Small vs medium comparison plots saved.")

elif PLOT_MODE == "sequence":
    for task_name in sequence_tasks:
        files = get_sequence_files(task_name)

        for file_path in files:
            plot_sequence(task_name, file_path)
    print("Sequence plots saved.")

elif PLOT_MODE == "duty_current":
    plot_duty_current()
    print("Duty-current plots saved.")

print("All plots saved.")
