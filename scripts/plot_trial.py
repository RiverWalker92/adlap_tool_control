#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# SEQUENCE_FOLDER = "/home/leanne/ros2_ws/test_data/unstructured"
# CONTINUOUS_FOLDER = "/home/leanne/ros2_ws/test_data/setup_03/gearbox_instrument"
VIDEO_DATA_FOLDER = "/home/leanne/ros2_ws/test_data/video_data"
VIDEO_ANGLE_FILE = None #"/home/leanne/ros2_ws/test_data/video_data/sequence_DOF4_trial_14_angles.jsonl"

GEARBOX_LABELS = [
    "inner shaft rotation [deg]",
    "collet cam rotation [deg]",
    "inner shaft translation [mm]",
    "middle shaft rotation [deg]",
    "outer shaft rotation [deg]",
    "middle-outer relative rotation [deg]",
]
TRIM_BY_DURATION = True
TRIM_DURATION = 550
TRIM_MARGIN_AFTER = 0.5


# sequence_tasks = [
#     "coupling_sequence",
#     "initialization_sequence",
#     "update_starting_positions",
# ]

# continuous_time_filters = [
#     "20260612_0954",
# ]

# time_filters = [
#     "20260529_12",
# ]

# PLOT_MODE = "continuous_dof" 
# # opties: "continuous_dof", "idle", "duty_current", "test_type", "small_vs_medium", "sequence"


# trial_numbers = [1, 2, 3]

# test_types = [
#     # "idle_baseline",
#     "single_step_small",
#     "single_step_medium",
#     "backlash_small",
#     "reversal_medium",
#     "cyclic_medium",
# ]

# test_descriptions = {
#     "single_step_small": "Single relative motor step of +100 pulses, followed by settling.",
#     "single_step_medium": "Single relative motor step of +300 pulses, followed by settling.",
#     "backlash_small": "Positive step of +100 pulses followed by negative step of -100 pulses.",
#     "reversal_medium": "Direction reversal using +300 and -300 pulse commands.",
#     "cyclic_medium": "Repeated forward/backward motion using ±300 pulses for 15 cycles.",
#     "idle_baseline": "No commanded motion; baseline current and position stability."
# }

# motor_folders = ["m1", "m2", "m3", "m4"]
def radians_to_deg_array(values):
    clean_values = [
        np.nan if v is None else v
        for v in values
    ]
    return np.degrees(np.array(clean_values, dtype=float))

def load_continuous_file(file_path):
    t = []
    commanded = [[], [], [], []]
    current_angles = [[], [], [], []]
    motor_pos = [[], [], [], []]
    currents = [[], [], [], []]
    gearbox = [[], [], [], [], [], []]
    predicted_angles = [[] for _ in range(8)]
    hybrid_predicted_angles = [[] for _ in range(8)]
    ros_timestamps = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            cmd = data.get("commanded_instrument_angles")
            cur_ang = data.get("current_instrument_angles")
            pos = data.get("measured_motor_positions")
            cur = data.get("measured_currents")
            gb = data.get("gearbox_state")
            pred_ang = data.get("predicted_instrument_angles")
            hybrid_pred_ang = data.get("hybrid_predicted_instrument_angles")

            if cmd is None or pos is None or cur is None or gb is None or pred_ang is None:
                continue

            t.append(data["time"])
            ros_timestamps.append(data["ros_timestamp"])

            for i in range(4):
                commanded[i].append(cmd[i])
                current_angles[i].append(cur_ang[i] if cur_ang is not None else None)
                motor_pos[i].append(pos[i])
                currents[i].append(cur[i])

            for i in range(6):
                gearbox[i].append(gb[i])

            for i in range(8):
                if pred_ang is not None and len(pred_ang) > i:
                    predicted_angles[i].append(pred_ang[i])
                else:
                    predicted_angles[i].append(None)
            
            for i in range(8):
                if hybrid_pred_ang is not None and len(hybrid_pred_ang) > i:
                    hybrid_predicted_angles[i].append(hybrid_pred_ang[i])
                else:
                    hybrid_predicted_angles[i].append(None)

    motor_ros_t0 = ros_timestamps[0]
    t0 = t[0]
    t = [x - t0 for x in t]

    for i in range(4):
        p0 = motor_pos[i][0]
        motor_pos[i] = [p - p0 for p in motor_pos[i]]
    
    for i in range(8):
        first_valid = next((v for v in predicted_angles[i] if v is not None), None)
        if first_valid is not None:
            predicted_angles[i] = [
                v - first_valid if v is not None else None
                for v in predicted_angles[i]
            ]
    for i in range(8):
        first_valid = next((v for v in hybrid_predicted_angles[i] if v is not None), None)
        if first_valid is not None:
            hybrid_predicted_angles[i] = [
                v - first_valid if v is not None else None
                for v in hybrid_predicted_angles[i]
            ]

    if TRIM_BY_DURATION:
        t_end = TRIM_DURATION + TRIM_MARGIN_AFTER

        keep = [
            i for i, ti in enumerate(t)
            if ti <= t_end
        ]

        t = [t[i] for i in keep]

        for k in range(4):
            commanded[k] = [commanded[k][i] for i in keep]
            current_angles[k] = [current_angles[k][i] for i in keep]
            motor_pos[k] = [motor_pos[k][i] for i in keep]
            currents[k] = [currents[k][i] for i in keep]

        for k in range(6):
            gearbox[k] = [gearbox[k][i] for i in keep]

        for k in range(8):
            predicted_angles[k] = [predicted_angles[k][i] for i in keep]
        
        for k in range(8):
            hybrid_predicted_angles[k] = [
                hybrid_predicted_angles[k][i]
                for i in keep
            ]

    return (
        t,
        commanded,
        current_angles,
        predicted_angles,
        hybrid_predicted_angles,
        motor_pos,
        currents,
        gearbox,
        motor_ros_t0,
        )

# def get_continuous_files():
#     files = []

#     for root, dirs, filenames in os.walk(CONTINUOUS_FOLDER):
#         for filename in filenames:
#             if not filename.endswith(".jsonl"):
#                 continue

#             if continuous_time_filters and not any(f in filename for f in continuous_time_filters):
#                 continue

#             files.append(os.path.join(root, filename))

#     files.sort()
#     return files

# def get_sequence_timestamp(file_path):
#     name = os.path.basename(file_path)
#     name = name.replace(".jsonl", "")

#     # voorbeeld: coupling_sequence_20260508_100151
#     parts = name.split("_")
#     if len(parts) >= 3:
#         return parts[-2] + "_" + parts[-1]

#     return "unknown_time"

def load_video_angle_file(file_path, motor_ros_t0):
    t_video = []
    red_shaft_angle = []
    jaw_angle = []
    yellow_shaft_angle = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            ros_time = data.get("ros_time_s")

            red_angle = data.get("measured_angle_red_shaft_zeroed")
            if red_angle is None:
                red_angle = data.get("relative_marker_to_shaft_zeroed_deg")
            raw_red_shaft = data.get("measured_angle_red_shaft")

            yellow_angle = data.get("measured_angle_yellow_shaft_zeroed")
            if yellow_angle is None:
                yellow_angle = data.get("yellow_relative_angle_zeroed_deg")
            raw_yellow_shaft = data.get("measured_angle_yellow_shaft")

            measured_jaw_angle = data.get("measured_angle_between_jaws")

            if ros_time is None:
                continue
            
            t_rel = ros_time - motor_ros_t0

            if t_rel < -1.0 or t_rel > TRIM_DURATION + 1.0:
                continue

            t_video.append(t_rel)
            red_shaft_angle.append(red_angle)
            jaw_angle.append(measured_jaw_angle)
            yellow_shaft_angle.append(yellow_angle)
            # blue_shaft_angle.append(blue_angle)

    print("video samples:", len(t_video))
    print("valid red shaft angles:", sum(v is not None for v in red_shaft_angle))
    print("valid jaw angles:", sum(v is not None for v in jaw_angle))
    return t_video, red_shaft_angle, jaw_angle, yellow_shaft_angle

# def load_sequence_file(file_path):
#     print(f"Loading: {file_path}")

#     timestamps = []
#     positions = [[], [], [], []]
#     currents = [[], [], [], []]
#     commands = [[], [], [], []]
#     duty = [[], [], [], []]

#     with open(file_path, "r") as f:
#         for line in f:
#             data = json.loads(line)

#             timestamps.append(data["timestamp"])

#             pos_data = data.get("measured_motor_positions")
#             cur_data = data.get("measured_currents")
#             cmd_data = data.get("commanded_motor_positions")
#             duty_data = data.get("commanded_duty_cycle")

#             for i in range(4):
#                 positions[i].append(pos_data[i] if pos_data is not None else None)
#                 currents[i].append(cur_data[i] if cur_data is not None else None)
#                 commands[i].append(cmd_data[i] if cmd_data is not None else None)
#                 duty[i].append(duty_data[i] if duty_data is not None else None)

#     if not timestamps:
#         return None

#     t0 = timestamps[0]
#     timestamps = [t - t0 for t in timestamps]

#     for i in range(4):
#         if positions[i][0] is not None:
#             p0 = positions[i][0]
#             positions[i] = [p - p0 if p is not None else None for p in positions[i]]

#         if commands[i][0] is not None:
#             c0 = commands[i][0]
#             commands[i] = [c - c0 if c is not None else None for c in commands[i]]

#     return timestamps, positions, currents, commands, duty, file_path

# def get_sequence_files(task_name):
#     folder = os.path.join(SEQUENCE_FOLDER, task_name)

#     if not os.path.isdir(folder):
#         print(f"No folder found: {folder}")
#         return []

#     files = [
#         os.path.join(folder, f)
#         for f in os.listdir(folder)
#         if f.endswith(".jsonl")
#     ]

#     files.sort()
#     return files

def extract_ros_parameters(config: dict) -> dict:
    if not isinstance(config, dict):
        raise RuntimeError("YAML config is empty or invalid.")

    preferred_keys = [
        "tool_controller_node",
        "/tool_controller_node",
        "pattern_runner_node",
        "/pattern_runner_node",
        "/**",
    ]

    for key in preferred_keys:
        if key in config and isinstance(config[key], dict):
            if "ros__parameters" in config[key]:
                return config[key]["ros__parameters"]

    if "ros__parameters" in config:
        return config["ros__parameters"]

    return config


def safe_number_for_filename(value):
    return f"{float(value):.3g}".replace(".", "p").replace("-", "m")


def get_frequency_segments_from_params(params_file, active_dof, t, commanded):
    """
    Returns time windows for every range-frequency block in the sequence.

    Example:
    sequence_range_factors: [0.5, 1.0]
    sequence_frequency_factors: [2.0, 1.0, 0.5]

    Output:
    range 0.5, freq 2.0
    range 0.5, freq 1.0
    range 0.5, freq 0.5
    range 1.0, freq 2.0
    range 1.0, freq 1.0
    range 1.0, freq 0.5
    """
    if params_file is None or active_dof is None:
        return []

    import yaml
    import numpy as np

    with open(params_file, "r") as f:
        config = yaml.safe_load(f)

    params = extract_ros_parameters(config)

    dof_name = f"dof{active_dof + 1}"

    if dof_name not in params:
        return []

    dof_params = params[dof_name]

    if dof_params.get("mode", "constant") != "sequence":
        return []

    base_frequency = float(dof_params.get("frequency", 0.1))
    value = float(dof_params.get("value", 0.0))

    frequency_factors = [
        float(x)
        for x in dof_params.get("sequence_frequency_factors", [1.0])
    ]

    range_factors = [
        float(x)
        for x in dof_params.get("sequence_range_factors", [1.0])
    ]

    modes = list(dof_params.get("sequence_modes", ["sinusoid", "triangle"]))

    cycles = [
        float(x)
        for x in dof_params.get("sequence_cycles", [5, 5])
    ]

    stage_pause = float(dof_params.get("sequence_pause_duration", 5.0))
    between_frequency_pause = float(
        dof_params.get("sequence_between_frequency_pause", 5.0)
    )
    between_range_pause = float(
        dof_params.get("sequence_between_range_pause", 5.0)
    )

    if len(modes) != len(cycles):
        return []

    # Estimate where the real sequence starts in the plotted time axis.
    # For DOF4 this is after preroll/sync; command starts changing after that.
    cmd = np.array(commanded[active_dof], dtype=float)
    tt = np.array(t, dtype=float)

    valid = np.isfinite(cmd)
    moving_indices = np.where(valid & (np.abs(cmd - value) > 1e-4))[0]

    if len(moving_indices) == 0:
        motion_start = float(tt[0])
    else:
        first_idx = max(0, int(moving_indices[0]) - 2)
        motion_start = float(tt[first_idx])

    segments = []
    elapsed = 0.0
    sequence_block_index = 0

    for range_index, range_factor in enumerate(range_factors):
        for frequency_index, frequency_factor in enumerate(frequency_factors):
            frequency = base_frequency * frequency_factor

            motion_duration = sum(
                cycle_count / frequency
                for cycle_count in cycles
            )

            number_of_stage_pauses = max(0, len(modes) - 1)
            block_duration = (
                motion_duration
                + number_of_stage_pauses * stage_pause
            )

            start = motion_start + elapsed
            end = start + block_duration

            segments.append(
                {
                    "sequence_block_index": sequence_block_index,
                    "range_index": range_index,
                    "range_factor": range_factor,
                    "frequency_index": frequency_index,
                    "frequency_factor": frequency_factor,
                    "frequency": frequency,
                    "start": start,
                    "end": end,
                }
            )

            sequence_block_index += 1
            elapsed += block_duration

            # Pause between frequency blocks within the same range block.
            is_last_frequency = (
                frequency_index == len(frequency_factors) - 1
            )

            if not is_last_frequency:
                elapsed += between_frequency_pause

        # Pause between range blocks.
        is_last_range = range_index == len(range_factors) - 1

        if not is_last_range:
            elapsed += between_range_pause

    return segments
    
# def moving_average(values, window=11):
#     """
#     Symmetric moving average.
#     Beter dan alleen backwards averaging, omdat de curve minder verschuift in de tijd.
#     """
#     result = []
#     n = len(values)
#     half = window // 2

#     for i in range(n):
#         start = max(0, i - half)
#         end = min(n, i + half + 1)
#         subset = [v for v in values[start:end] if v is not None]

#         if len(subset) == 0:
#             result.append(None)
#         else:
#             result.append(sum(subset) / len(subset))

#     return result

# def compute_velocity(timestamps, positions, position_window=11, velocity_window=11):
#     velocities = [[], [], [], []]

#     for motor_index in range(4):
#         pos = positions[motor_index]
#         vel = [0.0]

#         # 1. Smooth eerst de positie
#         pos_smooth = moving_average(pos, window=position_window)

#         # 2. Bereken velocity met central difference
#         vel = []

#         for i in range(len(timestamps)):
#             if i == 0 or i == len(timestamps) - 1:
#                 vel.append(0.0)
#                 continue

#             t_prev = timestamps[i - 1]
#             t_next = timestamps[i + 1]
#             p_prev = pos_smooth[i - 1]
#             p_next = pos_smooth[i + 1]

#             dt = t_next - t_prev

#             if dt <= 0 or p_prev is None or p_next is None:
#                 vel.append(0.0)
#             else:
#                 vel.append((p_next - p_prev) / dt)

#         # 3. Smooth velocity nog een keer licht
#         vel_smooth = moving_average(vel, window=velocity_window)

#         velocities[motor_index] = [
#             v if v is not None else 0.0
#             for v in vel_smooth
#         ]

#     return velocities

def detect_active_dof(commanded, threshold=0.01):
    for i in range(4):
        if max(commanded[i]) - min(commanded[i]) > threshold:
            return i
    return None

def compute_prediction_metrics(video_t, video_angle_deg, prediction_t, prediction_deg):
    video_t = np.array(video_t, dtype=float)
    video_angle_deg = np.array(
        [np.nan if v is None else v for v in video_angle_deg],
        dtype=float,
    )

    prediction_t = np.array(prediction_t, dtype=float)
    prediction_deg = np.array(prediction_deg, dtype=float)

    valid_video = np.isfinite(video_t) & np.isfinite(video_angle_deg)
    valid_prediction = np.isfinite(prediction_t) & np.isfinite(prediction_deg)

    if np.sum(valid_video) < 2 or np.sum(valid_prediction) < 2:
        return None

    video_t = video_t[valid_video]
    video_angle_deg = video_angle_deg[valid_video]

    prediction_t = prediction_t[valid_prediction]
    prediction_deg = prediction_deg[valid_prediction]

    overlap_start = max(video_t[0], prediction_t[0])
    overlap_end = min(video_t[-1], prediction_t[-1])

    overlap_mask = (video_t >= overlap_start) & (video_t <= overlap_end)

    if np.sum(overlap_mask) < 2:
        return None

    video_t_overlap = video_t[overlap_mask]
    video_angle_overlap = video_angle_deg[overlap_mask]

    prediction_at_video_time = np.interp(
        video_t_overlap,
        prediction_t,
        prediction_deg,
    )

    error = prediction_at_video_time - video_angle_overlap

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    max_abs_error = float(np.max(np.abs(error)))

    return {
        "n_samples": int(len(error)),
        "mae_deg": mae,
        "rmse_deg": rmse,
        "bias_deg": bias,
        "max_abs_error_deg": max_abs_error,
        "overlap_start_s": float(overlap_start),
        "overlap_end_s": float(overlap_end),
    }


def write_dof2_prediction_metrics(
    plot_dir,
    physics_metrics,
    hybrid_metrics,
    used_prediction_name,
):
    metrics_path = os.path.join(plot_dir, "dof2_prediction_metrics.txt")

    def write_block(f, title, metrics):
        f.write(f"{title}\n")
        f.write("-" * len(title) + "\n")

        if metrics is None:
            f.write("Not available\n\n")
            return

        f.write(f"MAE:             {metrics['mae_deg']:.3f} deg\n")
        f.write(f"RMSE:            {metrics['rmse_deg']:.3f} deg\n")
        f.write(f"Bias:            {metrics['bias_deg']:.3f} deg\n")
        f.write(f"Max abs error:   {metrics['max_abs_error_deg']:.3f} deg\n")
        f.write(f"Samples:         {metrics['n_samples']}\n")
        f.write(
            f"Time window:     "
            f"{metrics['overlap_start_s']:.3f} - "
            f"{metrics['overlap_end_s']:.3f} s\n\n"
        )

    with open(metrics_path, "w") as f:
        f.write("DOF2 prediction metrics\n")
        f.write("=======================\n\n")
        f.write("Error definition:\n")
        f.write("prediction_error = prediction_deg - video_measurement_deg\n\n")
        f.write(f"Prediction shown in overview plot: {used_prediction_name}\n\n")

        write_block(f, "Physics-based DT", physics_metrics)
        write_block(f, "Hybrid DT", hybrid_metrics)

    print(f"Saved DOF2 prediction metrics: {metrics_path}")
    return metrics_path


def add_metrics_text_box(ax, model_name, metrics):
    if metrics is None:
        return

    text = (
        f"{model_name}\n"
        f"MAE = {metrics['mae_deg']:.2f}°\n"
        f"RMSE = {metrics['rmse_deg']:.2f}°"
    )

    ax.text(
        0.01,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", alpha=0.8),
    )

def plot_continuous_file(file_path, video_angle_file=None, output_dir=None, params_file=None):
    t, commanded, current_angles, predicted_angles, hybrid_predicted_angles, motor_pos, currents, gearbox, motor_ros_t0 = load_continuous_file(file_path)
    basename = os.path.basename(file_path).replace(".jsonl", "")
    pattern_name = os.path.basename(os.path.dirname(file_path))

    if output_dir is None:
        plot_dir = os.path.join(os.path.dirname(file_path), "plots", basename)
    else:
        plot_dir = output_dir

    os.makedirs(plot_dir, exist_ok=True)


    # Plot 1: commanded vs current instrument angles
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Instrument angles: commanded vs current via encoder ouput", fontsize=16)

    for i in range(4):
        axes[i].plot(t, commanded[i], label=f"commanded dof{i+1}")
        axes[i].plot(t, current_angles[i], label=f"current dof{i+1}", linestyle="--")
        axes[i].set_ylabel(f"dof{i+1} [rad]")
        axes[i].grid(True)
        axes[i].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "instrument_angles.png"), dpi=200)
    plt.close(fig)

    # Plot 2: motor positions
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Measured motor positions", fontsize=16)

    for i in range(4):
        axes[i].plot(t, motor_pos[i], label=f"motor {i}")
        axes[i].set_ylabel("Δ pulses")
        axes[i].grid(True)
        axes[i].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "motor_positions.png"), dpi=200)
    plt.close(fig)

    # Plot 3: motor currents
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Measured motor currents", fontsize=16)

    for i in range(4):
        axes[i].plot(t, currents[i], label=f"motor {i}")
        axes[i].set_ylabel("current")
        axes[i].grid(True)
        axes[i].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "motor_currents.png"), dpi=200)
    plt.close(fig)

    # Plot 4: gearbox state
    fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    fig.suptitle("Gearbox prediction", fontsize=16)

    for i in range(6):
        axes[i].plot(t, gearbox[i], label=GEARBOX_LABELS[i])
        axes[i].set_ylabel(GEARBOX_LABELS[i])
        axes[i].grid(True)
        axes[i].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "gearbox_state.png"), dpi=200)
    plt.close(fig)


    # Plot 5: OVERVIEW: main plot, general DOF mapping overview
    active_dof = detect_active_dof(commanded)
    # If DOF4 also clearly moves, prefer DOF4 over the artificial DOF3 compensation.
    if max(commanded[3]) - min(commanded[3]) > 0.01:
        active_dof = 3
    active_dof_name = f"dof{active_dof + 1}" if active_dof is not None else "unknown_dof"

    if video_angle_file is not None:
        video_t, red_shaft_video_angle, jaw_video_angle, yellow_shaft_video_angle = load_video_angle_file(
            video_angle_file,
            motor_ros_t0
        )
    else:
        video_t = []
        red_shaft_video_angle = []
        jaw_video_angle = []
        yellow_shaft_video_angle = []
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

    fig.suptitle(f"Continuous sequence sinusoid-triangle test: {active_dof_name}", fontsize=16)

    # 1. Active command
    if active_dof is not None:
        axes[0].plot(
            t,
            commanded[active_dof],
            label=f"requested {active_dof_name}",
            linewidth=1.5,
            color="green"
        )

        if current_angles[active_dof] and current_angles[active_dof][0] is not None:
            axes[0].plot(
                t,
                current_angles[active_dof],
                linestyle="--",
                label=f"controller output {active_dof_name}",
                linewidth=1.2,
                color = "black"
            )

    axes[0].set_title("Instrument command")
    axes[0].set_ylabel("Command [rad]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # 2. All motor positions
    for motor_index in range(4):
        axes[1].plot(
            t,
            motor_pos[motor_index],
            label=f"motor {motor_index}",
            linewidth=1.3
        )

    axes[1].set_title("Measured motor positions")
    axes[1].set_ylabel("Motor position\nΔ pulses")
    axes[1].grid(True)
    axes[1].legend(fontsize=8, ncol=4)

    # 3. All motor currents
    for motor_index in range(4):
        axes[2].plot(
            t,
            currents[motor_index],
            label=f"motor {motor_index}",
            linewidth=1.0
        )

    axes[2].set_title("Measured motor currents")
    axes[2].set_ylabel("Motor current\n[mA]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8, ncol=4)

    # 4. Relevant gearbox output
    if active_dof == 3: # dof4
        axes[3].plot(t, gearbox[2], label="Gearbox prediction: inner shaft translation [mm]", color = "blue")
        axes[3].set_ylabel("Translation [mm]")
        axes[3].set_title("Gearbox output: predicted inner shaft translation")

    elif active_dof == 2: # dof3
        axes[3].plot(t, gearbox[0], label="Gearbox prediction: inner shaft rotation [deg]", color = "blue")
        axes[3].set_ylabel("Rotation [deg]")
        axes[3].set_title("Gearbox output: predicted inner shaft rotation")

    else: # dof1/dof2
        axes[3].plot(t, gearbox[3], label="Gearbox prediction: middle shaft rotation [deg]")
        axes[3].plot(t, gearbox[4], label="Gearbox prediction: outer shaft rotation [deg]")
        axes[3].plot(t, gearbox[5], label="Gearbox prediction: middle-outer relative rotation [deg]", color = "blue")
        axes[3].set_ylabel("Rotation [deg]")
        axes[3].set_title("Gearbox output: predicted shaft rotations")

        # axes[3].plot(
        #     video_t,
        #     video_angle,
        #     linewidth=2,
        #     linestyle="--",
        #     label="video measured angle"
        # )

    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True)
    axes[3].legend(fontsize=8)


    # Plot 5. Instrument DT prediction vs video measurement
    if active_dof == 1: # DOF2 
        predicted_pitch_deg = radians_to_deg_array(predicted_angles[1])
        predicted_bend_deg = radians_to_deg_array(predicted_angles[4])
        hybrid_bend_deg = radians_to_deg_array(hybrid_predicted_angles[4])

        has_hybrid_bend = np.isfinite(hybrid_bend_deg).any()

        physics_metrics = compute_prediction_metrics(
            video_t,
            red_shaft_video_angle,
            t,
            predicted_bend_deg,
        )

        hybrid_metrics = None

        if has_hybrid_bend:
            hybrid_metrics = compute_prediction_metrics(
                video_t,
                red_shaft_video_angle,
                t,
                hybrid_bend_deg,
            )

            used_prediction_name = "Hybrid DT bend"

            axes[4].plot(
                t,
                hybrid_bend_deg,
                label="Hybrid DT bend [deg]",
                linewidth=1.5,
                color="green",
            )

            add_metrics_text_box(
                axes[4],
                "Hybrid DT",
                hybrid_metrics,
            )

        else:
            used_prediction_name = "Physics-based DT bend"

            axes[4].plot(
                t,
                predicted_bend_deg,
                label="Physics-based DT bend [deg]",
                linewidth=1.5,
                color="purple",
            )

            add_metrics_text_box(
                axes[4],
                "Physics DT",
                physics_metrics,
            )

        write_dof2_prediction_metrics(
            plot_dir,
            physics_metrics,
            hybrid_metrics,
            used_prediction_name,
        )
        # axes[4].plot(
        #     t,
        #     predicted_pitch_deg,
        #     label="DT predicted pitch projection [deg]",
        #     linewidth=1.2,
        #     linestyle="-.",
        #     color = "purple"
        # )

        axes[4].plot(
            video_t,
            red_shaft_video_angle,
            "--",
            linewidth=2,
            color="orange",   
            label="video measured red marker vs shaft [deg]",
        )

    elif active_dof == 3: # DOF4
        commanded_deg = np.degrees(np.array(commanded[3], dtype=float))
        predicted_deg = np.degrees(np.array(predicted_angles[3], dtype=float))

        # axes[4].plot(
        #     t,
        #     commanded_deg,
        #     label="commanded gripper/articulation [deg]",
        #     linewidth=1.2,
        # )

        axes[4].plot(
            t,
            predicted_deg,
            label="instrument DT predicted gripper/articulation [deg]",
            linewidth=1.5,
            color = "purple"
        )

        axes[4].plot(
            video_t,
            jaw_video_angle,
            ".",
            markersize=3,
            color="orange",
            label="video measured angle between jaws [deg]",
        )

    elif active_dof is not None: # Fallback for DOF1/DOF3
        predicted_deg = np.degrees(np.array(predicted_angles[active_dof], dtype=float))
        commanded_deg = np.degrees(np.array(commanded[active_dof], dtype=float))

        # axes[4].plot(
        #     t,
        #     commanded_deg,
        #     label=f"commanded {active_dof_name} [deg]",
        #     linewidth=1.2,
        # )

        axes[4].plot(
            t,
            predicted_deg,
            label=f"instrument DT predicted {active_dof_name} [deg]",
            linewidth=1.5,
            color = "purple"
        )

        axes[4].plot(
            video_t,
            red_shaft_video_angle,
            "--",
            linewidth=2,
            color="orange",
            label="video measured red marker vs shaft [deg]",
        )

    axes[4].set_title("Instrument output: DT prediction vs video measurement")
    axes[4].set_ylabel("Angle [deg]")
    axes[4].set_xlabel("Time [s]")
    axes[4].grid(True)
    axes[4].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    safe_pattern_name = pattern_name.replace("/", "_").replace("|", "_").replace(" ", "_")
    safe_active_dof_name = active_dof_name.replace(" ", "_")

    overview_filename = f"overview_{safe_pattern_name}_{safe_active_dof_name}.png"

    fig.savefig(os.path.join(plot_dir, overview_filename), dpi=200)
    
    # Cropped overview per frequency block
    frequency_segments = get_frequency_segments_from_params(
        params_file=params_file,
        active_dof=active_dof,
        t=t,
        commanded=commanded,
    )

    original_xlim = axes[0].get_xlim()

    for segment in frequency_segments:
        for ax in axes:
            ax.set_xlim(segment["start"], segment["end"])

        freq_text = safe_number_for_filename(segment["frequency"])
        freq_factor_text = safe_number_for_filename(segment["frequency_factor"])
        range_text = safe_number_for_filename(segment["range_factor"])

        segment_filename = (
            f"overview_{safe_pattern_name}_{safe_active_dof_name}"
            f"_range{range_text}"
            f"_freq{freq_text}Hz"
            f"_freqfactor{freq_factor_text}.png"
        )

        fig.savefig(os.path.join(plot_dir, segment_filename), dpi=200)


    # Restore full view in case the figure is reused before closing
    for ax in axes:
        ax.set_xlim(original_xlim)

    plt.close(fig)

    if active_dof == 1:
        plot_dof2_video_validation(
        plot_dir,
        t,
        commanded,
        current_angles,
        motor_pos,
        video_t,
        red_shaft_video_angle,
        predicted_angles,
        hybrid_predicted_angles,
    )
    if active_dof == 3:
        plot_gripper_video_measurements(
            plot_dir,
            video_t,
            jaw_video_angle,
            yellow_shaft_video_angle, #yellow_shaft_video_angle,
            red_shaft_video_angle,
        )    
    print(f"Saved plots in: {plot_dir}")

def plot_gripper_video_measurements(
    plot_dir,
    video_t,
    jaw_video_angle,
    yellow_shaft_video_angle, #yellow_shaft_video_angle,
    red_shaft_video_angle,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Gripper video measurements", fontsize=16)

    axes[0].plot(
        video_t,
        jaw_video_angle,
        "--",
        linewidth=2,
        label="measured jaw angle: yellow vs red [deg]",
    )
    axes[0].set_ylabel("Jaw angle [deg]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        video_t,
        yellow_shaft_video_angle, #yellow_shaft_video_angle,
        "--",
        linewidth=2,
        label="measured yellow marker vs shaft [deg]",
    )
    axes[1].set_ylabel("Yellow-shaft [deg]")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        video_t,
        red_shaft_video_angle,
        "--",
        linewidth=2,
        label="measured red marker vs shaft [deg]",
    )
    axes[2].set_ylabel("Red-shaft [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(plot_dir, "gripper_video_measurements.png"), dpi=200)
    plt.close(fig)

def plot_dof2_video_validation(plot_dir, t, commanded, current_angles, motor_pos, video_t, video_angle, predicted_angles=None, hybrid_predicted_angles=None,):
    motor2_minus_motor1 = [
        m2 - m1 for m1, m2 in zip(motor_pos[1], motor_pos[2])
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("DOF2 validation: command, motor difference, and video angle", fontsize=16)

    axes[0].plot(t, commanded[1], label="commanded dof2 / pitch [rad]")
    axes[0].plot(t, current_angles[1], linestyle="--", label="current dof2 / pitch from motors [rad]")
    axes[0].set_ylabel("Pitch [rad]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].plot(t, motor_pos[1], label="motor 1")
    axes[1].plot(t, motor_pos[2], label="motor 2")
    axes[1].plot(t, motor2_minus_motor1, linewidth=2, label="motor2 - motor1")
    axes[1].set_ylabel("Δ pulses")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    pitch_rad = np.array(current_angles[1], dtype=float)
    command_rad = np.array(commanded[1], dtype=float)

    pitch_deg = np.degrees(pitch_rad)
    command_deg = np.degrees(command_rad)
    axes[2].plot(
        t,
        command_deg,
        label="commanded pitch [deg]"
    )

    axes[2].plot(
        t,
        pitch_deg,
        label="current pitch from motors [deg]"
    )

    axes[2].plot(
        video_t,
        video_angle,
        "--",
        linewidth=2,
        label="video measured angle [deg]"
    )
    has_hybrid_bend = False

    if hybrid_predicted_angles is not None:
        hybrid_bend_deg = radians_to_deg_array(hybrid_predicted_angles[4])
        has_hybrid_bend = np.isfinite(hybrid_bend_deg).any()

    if has_hybrid_bend:
        axes[2].plot(
            t,
            hybrid_bend_deg,
            label="Hybrid DT bend [deg]",
            linewidth=1.2,
            color="green",
        )

    elif predicted_angles is not None:
        predicted_bend_deg = radians_to_deg_array(predicted_angles[4])
        axes[2].plot(
            t,
            predicted_bend_deg,
            label="Physics-based DT bend [deg]",
            linewidth=1.2,
            color="purple",
        )

    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(plot_dir, "dof2_video_validation.png"), dpi=200)
    plt.close(fig)

# def plot_sequence(task_name, file_path):
#     data = load_sequence_file(file_path)

#     if data is None:
#         return

#     timestamps, positions, currents, commands, duty, file_path = data
#     velocities = compute_velocity(timestamps, positions,  position_window=15,
#     velocity_window=15)

#     fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
#     fig.suptitle(task_name, fontsize=16, y=0.98)

#     fig.text(
#         0.5, 0.94,
#         f"File: {os.path.basename(file_path)}",
#         ha="center",
#         fontsize=9
#     )

#     for motor_index in range(4):
#         ax_pos = axes[0, motor_index]
#         ax_vel = axes[1, motor_index]
#         ax_cur = axes[2, motor_index]

#         ax_pos.plot(timestamps, positions[motor_index], marker="o", markersize=2)
#         ax_pos.set_title(f"Motor {motor_index}")
#         ax_pos.set_ylabel("Δ position [pulses]")
#         ax_pos.grid(True)

#         ax_vel.plot(timestamps, velocities[motor_index], marker="o", markersize=2)
#         ax_vel.set_ylabel("Velocity [pulses/s]")
#         ax_vel.grid(True)

#         ax_cur.plot(timestamps, currents[motor_index], color="orange", marker="o", markersize=2)
#         ax_cur.set_ylabel("Current [ ]")
#         ax_cur.set_xlabel("Time [s]")
#         ax_cur.grid(True)

#     plt.tight_layout(rect=[0, 0, 1, 0.91])

#     sequence_timestamp = get_sequence_timestamp(file_path)
#     sequence_plot_dir = os.path.join(SEQUENCE_FOLDER, "plots", task_name, sequence_timestamp)
#     os.makedirs(sequence_plot_dir, exist_ok=True)

#     plot_path = os.path.join(sequence_plot_dir, f"{task_name}_timeseries.png")

#     fig.savefig(plot_path, dpi=200)
#     print(f"Saved plot: {plot_path}")

#     plt.close(fig)

# def plot_duty_current():
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
#     axes = axes.flatten()

#     fig.suptitle("Current vs commanded duty cycle", fontsize=16, y=0.98)
#     fig.text(
#         0.5, 0.94,
#         "All available sequence samples. Current depends on duty cycle, load, friction and motion state.",
#         ha="center",
#         fontsize=9
#     )

#     duty_all = [[], [], [], []]
#     current_all = [[], [], [], []]

#     for task_name in sequence_tasks:
#         files = get_sequence_files(task_name)

#         for file_path in files:
#             data = load_sequence_file(file_path)

#             if data is None:
#                 continue

#             timestamps, positions, currents, commands, duty, _ = data

#             for motor_index in range(4):
#                 for d, c in zip(duty[motor_index], currents[motor_index]):
#                     if d is not None and c is not None:
#                         duty_all[motor_index].append(d)
#                         current_all[motor_index].append(c)

#     for motor_index in range(4):
#         ax = axes[motor_index]

#         ax.scatter(
#             duty_all[motor_index],
#             current_all[motor_index],
#             s=8,
#             alpha=0.5
#         )

#         ax.set_title(f"Motor {motor_index}")
#         ax.set_xlabel("Commanded duty cycle")
#         ax.set_ylabel("Measured current [Pico units]")
#         ax.grid(True)

#     plt.tight_layout(rect=[0, 0, 1, 0.91])

#     plot_dir = os.path.join(SEQUENCE_FOLDER, "plots", "duty_current")
#     os.makedirs(plot_dir, exist_ok=True)

#     plot_path = os.path.join(plot_dir, "current_vs_duty_cycle_all_sequences.png")
#     fig.savefig(plot_path, dpi=200)
#     print(f"Saved plot: {plot_path}")

#     plt.close(fig)

# def plot_idle_baseline(trial_number):
#     base_folder = os.path.join(setup_folder, "idle_baseline", "all_motors")

#     prefix = f"trial_{trial_number:02d}_"
#     files = [
#         f for f in os.listdir(base_folder)
#         if f.endswith(".jsonl")
#         and f.startswith(prefix)
#         and any(time_filter in f for time_filter in time_filters)
#     ]
#     files.sort()

#     if not files:
#         print(f"No idle baseline file found for {prefix}")
#         return

#     file_path = os.path.join(base_folder, files[-1])
#     print(f"idle_baseline: {file_path}")

#     timestamps = []
#     positions = [[], [], [], []]
#     currents = [[], [], [], []]

#     with open(file_path, "r") as f:
#         for line in f:
#             data = json.loads(line)
#             timestamps.append(data["timestamp"])

#             for i in range(4):
#                 positions[i].append(data["measured_motor_positions"][i])
#                 currents[i].append(data["measured_currents"][i])

#     t0 = timestamps[0]
#     timestamps = [t - t0 for t in timestamps]

#     fig = plt.figure(figsize=(12, 8))
#     fig.suptitle("idle_baseline", fontsize=16, y=0.98)
#     fig.text(
#         0.5, 0.94,
#         "No commanded motion; position stability and baseline current of all motors.",
#         ha="center",
#         fontsize=10
#     )

#     # normaliseer posities per motor
#     for i in range(4):
#         p0 = positions[i][0]
#         positions[i] = [p - p0 for p in positions[i]]

 
#     all_pos = [v for motor_data in positions for v in motor_data]
#     all_cur = [v for motor_data in currents for v in motor_data]

#     pos_min, pos_max = min(all_pos), max(all_pos)
#     cur_min, cur_max = min(all_cur), max(all_cur)

#     pos_margin = 0.1 * (pos_max - pos_min) if pos_max != pos_min else 1
#     cur_margin = 0.1 * (cur_max - cur_min) if cur_max != cur_min else 1

#     for i in range(4):
#         ax = plt.subplot(2, 2, i + 1)

#         ax.plot(timestamps, positions[i], color="blue", label="Δ position")
#         ax.set_ylim(pos_min - pos_margin, pos_max + pos_margin)
#         ax.set_ylabel("Δ position [pulses]")

#         ax2 = ax.twinx()
#         ax2.plot(timestamps, currents[i], color="orange", marker="o", markersize=2, linestyle="none", label="current")
#         ax2.set_ylim(cur_min - cur_margin, cur_max + cur_margin)
#         ax2.set_ylabel("Current [?]")

#         ax.set_title(f"Motor {i}")
#         ax.set_xlabel("Time [s]")

#         ax.legend(loc="upper left")
#         ax2.legend(loc="upper right")

#     plt.tight_layout(rect=[0, 0, 1, 0.95])

#     trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
#     os.makedirs(trial_folder, exist_ok=True)

#     plot_path = os.path.join(
#         trial_folder,
#         f"idle_baseline_t{trial_number}.png"
#     )
#     fig.savefig(plot_path, dpi=200)

#     print(f"Saved plot: {plot_path}")
#     plt.close(fig)


# def plot_test_type(test_type, trial_number):
#     fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)

#     fig.suptitle(f"{test_type} - Trial {trial_number}", fontsize=16, y=0.98)

#     description = test_descriptions.get(test_type, "")
#     fig.text(0.5, 0.94, description, ha="center", fontsize=10)

#     for motor_index, motor in enumerate(motor_folders):
#         data = load_motor_file(test_type, motor, trial_number)

#         if data is None:
#             continue

#         timestamps, pos, current = data

#         ax_pos = axes[0, motor_index]
#         ax_cur = axes[1, motor_index]

#         ax_pos.plot(
#             timestamps,
#             pos,
#             color="blue",
#             linewidth=1.5,
#             marker="o",
#             markersize=2,
#             label="motor position"
#             )


#         ax_pos.set_title(f"Motor {motor_index}")
#         ax_pos.set_ylabel("position (relative) [pulses]")
#         ax_pos.grid(True)
#         ax_pos.legend(loc="best", fontsize=8)

#         ax_cur.plot(
#             timestamps,
#             current,
#             color="orange",
#             linewidth=1.2,
#             marker="o",
#             markersize=2,
#             label="current"
#         )

#         ax_cur.set_xlabel("Time [s]")
#         ax_cur.set_ylabel("Current [?]")
#         ax_cur.grid(True)
#         ax_cur.legend(loc="best", fontsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.92])

#     trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
#     os.makedirs(trial_folder, exist_ok=True)

#     plot_path = os.path.join(
#         trial_folder,
#         f"{test_type}_t{trial_number}.png"
#     )

#     fig.savefig(plot_path, dpi=200)
#     print(f"Saved plot: {plot_path}")

#     plt.close(fig)


# def load_motor_file(test_type, motor, trial_number):
#     folder = os.path.join(setup_folder, test_type, motor)
#     prefix = f"trial_{trial_number:02d}_"

#     files = [
#         f for f in os.listdir(folder)
#         if f.endswith(".jsonl")
#         and f.startswith(prefix)
#         and any(time_filter in f for time_filter in time_filters)
#     ]
#     files.sort()

#     if not files:
#         print(f"No file found for {test_type}/{motor}, {prefix}, filters={time_filters}")
#         return None

#     file_path = os.path.join(folder, files[-1])
#     print(f"{test_type}/{motor}: {file_path}")

#     motor_index = motor_folders.index(motor)

#     timestamps = []
#     pos = []
#     current = []
#     command = []

#     with open(file_path, "r") as f:
#         for line in f:
#             data = json.loads(line)

#             timestamps.append(data["timestamp"])
#             pos.append(data["measured_motor_positions"][motor_index])
#             current.append(data["measured_currents"][motor_index])

#     t0 = timestamps[0]
#     timestamps = [t - t0 for t in timestamps]

#     p0 = pos[0]
#     pos = [p - p0 for p in pos]

#     return timestamps, pos, current

# def plot_small_vs_medium(trial_number):
#     fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False)
#     fig.suptitle(
#         f"Single step response: small vs medium, trial {trial_number:02d}",
#         fontsize=16,
#         y=0.98
#     )

#     fig.text(
#         0.5, 0.94,
#         "Comparison of +100 pulse and +300 pulse relative motor commands.",
#         ha="center",
#         fontsize=10
#     )

#     for motor_index, motor in enumerate(motor_folders):
#         small_data = load_motor_file("single_step_small", motor, trial_number)
#         medium_data = load_motor_file("single_step_medium", motor, trial_number)

#         if small_data is None or medium_data is None:
#             continue

#         t_small, pos_small, cur_small = small_data
#         t_medium, pos_medium, cur_medium = medium_data

#         ax_pos = axes[0, motor_index]
#         ax_cur = axes[1, motor_index]

#         ax_pos.plot(
#             t_small,
#             pos_small,
#             color="blue",
#             linewidth=1.5,
#             marker="o",
#             markersize=2,
#             label="small position"
#         )

#         ax_pos.plot(
#             t_medium,
#             pos_medium,
#             color="green",
#             linewidth=1.5,
#             marker="o",
#             markersize=2,
#             label="medium position"
#         )

#         ax_pos.set_title(f"Motor {motor_index}")
#         ax_pos.set_ylabel("relative position [pulses]")
#         ax_pos.grid(True)
#         ax_pos.legend(loc="best", fontsize=8)

#         ax_cur.plot(
#             t_small,
#             cur_small,
#             color="orange",
#             linewidth=1.2,
#             marker="o",
#             markersize=2,
#             label="small current"
#         )

#         ax_cur.plot(
#             t_medium,
#             cur_medium,
#             color="red",
#             linewidth=1.2,
#             marker="o",
#             markersize=2,
#             label="medium current"
#         )

#         ax_cur.set_xlabel("Time [s]")
#         ax_cur.set_ylabel("Current [?]")
#         ax_cur.grid(True)
#         ax_cur.legend(loc="best", fontsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.90])

#     trial_folder = os.path.join(run_dir, f"trial_{trial_number:02d}")
#     os.makedirs(trial_folder, exist_ok=True)

#     plot_path = os.path.join(
#         trial_folder,
#         f"single_step_small_vs_medium_t{trial_number}.png"
#     )

#     fig.savefig(plot_path, dpi=200)
#     print(f"Saved plot: {plot_path}")

#     plt.close(fig)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--video-angles", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--params-file",
        default=str(
            Path.home()
            / "ros2_ws"
            / "src"
            / "adlap_tool_control"
            / "config"
            / "tool_params.yaml"
        ),
    )
    args = parser.parse_args()

    plot_continuous_file(
        file_path=args.file,
        video_angle_file=args.video_angles,
        output_dir=args.output_dir,
        params_file=args.params_file,
    )

    print("All plots saved.")