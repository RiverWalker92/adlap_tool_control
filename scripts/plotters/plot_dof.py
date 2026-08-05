#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import yaml

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
TRIM_DURATION = 750
TRIM_MARGIN_AFTER = 0.5
SEGMENT_MARGIN_BEFORE = 0.2
SEGMENT_MARGIN_AFTER = 1.0

# General utility functions
# -----------------------------------------
def radians_to_deg_array(values):
    clean_values = [
        np.nan if v is None else v
        for v in values
    ]
    return np.degrees(np.array(clean_values, dtype=float))

def safe_number_for_filename(value):
    return f"{float(value):.3g}".replace(".", "p").replace("-", "m")

def safe_text_for_filename(value):
    return str(value).replace("/", "_").replace("|", "_").replace(" ", "_")

def clean_numeric_array(values):
    return np.array(
        [np.nan if value is None else value for value in values],
        dtype=float,
    )

def wrap_to_pi(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

def extract_ros_parameters(config: dict) -> dict:
    if not isinstance(config, dict):
        raise RuntimeError("YAML config is empty or invalid.")

    preferred_keys = [
        "dof_pattern_runner_node",
        "/dof_pattern_runner_node",
        "/**",
    ]

    for key in preferred_keys:
        if key in config and isinstance(config[key], dict):
            if "ros__parameters" in config[key]:
                return config[key]["ros__parameters"]

    if "ros__parameters" in config:
        return config["ros__parameters"]

    return config

def detect_active_dof(commanded, threshold=0.01):
    """
    Determine which DOF is actively commanded.

    The active DOF is detected from the command range. DOF4 is preferred
    if it moves, because DOF4 sequences can also introduce artificial DOF3
    compensation motion.
    """
    dof_ranges = []

    for dof_index in range(4):
        values = np.array(
            [np.nan if value is None else value for value in commanded[dof_index]],
            dtype=float,
        )

        valid_values = values[np.isfinite(values)]

        if len(valid_values) == 0:
            dof_ranges.append(0.0)
        else:
            dof_ranges.append(float(np.max(valid_values) - np.min(valid_values)))

    # Prefer DOF4 if it moves, because DOF4 can also create DOF3-like compensation.
    if dof_ranges[3] > threshold:
        return 3

    for dof_index, dof_range in enumerate(dof_ranges):
        if dof_range > threshold:
            return dof_index

    return None

# Continuous sequence data loading
# -----------------------------------------
def load_continuous_file(file_path):
    t = []
    commanded = [[], [], [], []]
    current_angles = [[], [], [], []]
    motor_pos = [[], [], [], []]
    # currents = [[], [], [], []]
    gearbox = [[], [], [], [], [], []]
    predicted_angles = [[] for _ in range(8)]
    hybrid_predicted_angles = [[] for _ in range(8)]
    ros_timestamps = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            cmd = data.get("commanded_instrument_angles")
            cur_ang = data.get("measured_instrument_angles")
            if cur_ang is None:
                cur_ang = data.get("current_instrument_angles")
            pos = data.get("measured_motor_positions")
            # cur = data.get("measured_currents")
            gb = data.get("gearbox_state")
            pred_ang = data.get("predicted_instrument_angles")
            hybrid_pred_ang = data.get("hybrid_predicted_instrument_angles")

            if cmd is None or pos is None or gb is None:
                continue

            t.append(data["time"])
            ros_timestamps.append(data["ros_timestamp"])

            for i in range(4):
                commanded[i].append(cmd[i])
                current_angles[i].append(cur_ang[i] if cur_ang is not None else None)
                motor_pos[i].append(pos[i])
                # currents[i].append(cur[i])

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
            # currents[k] = [currents[k][i] for i in keep]

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
        # currents,
        gearbox,
        motor_ros_t0,
        )

# Video angle data loading
# -----------------------------------------
def load_video_angle_file(file_path, motor_ros_t0):
    t_video = []
    red_shaft_angle = []
    jaw_angle = []
    # yellow_shaft_angle = []
    green_shaft_angle = []

    detected_dof = None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if detected_dof is None:
                detected_dof = data.get("active_dof")

            ros_time = data.get("ros_time_s")

            # red_angle = data.get("measured_angle_red_shaft_zeroed")
            # if red_angle is None:
            #     red_angle = data.get("relative_marker_to_shaft_zeroed_deg")
            # raw_red_shaft = data.get("measured_angle_red_shaft")

            # yellow_angle = data.get("measured_angle_yellow_shaft_zeroed")
            # if yellow_angle is None:
            #     yellow_angle = data.get("yellow_relative_angle_zeroed_deg")
            # raw_yellow_shaft = data.get("measured_angle_yellow_shaft")

            # measured_jaw_angle = data.get("measured_angle_between_jaws")
            # Use only the filtered video measurements for validation,
            
            # Filtered DOF2 video reference
            red_angle = data.get(
                "measured_angle_red_shaft_zeroed_filtered"
            )

            # Filtered additional DOF4 diagnostic signal
            # yellow_angle = data.get(
            #     "measured_angle_yellow_shaft_zeroed_filtered"
            # )

            # Nieuwe groene veldnaam
            green_angle = data.get(
                "measured_angle_green_shaft_zeroed_filtered"
            )

            # Ondersteun ook oudere bestanden waarin de groene marker
            # nog onder de legacy yellow-veldnaam werd opgeslagen.
            if green_angle is None:
                green_angle = data.get(
                    "measured_angle_yellow_shaft_zeroed_filtered"
                )

            # Filtered DOF4 validation reference
            measured_jaw_angle = data.get(
                "measured_angle_between_jaws_filtered"
            )

            if ros_time is None:
                continue

            t_rel = ros_time - motor_ros_t0

            if t_rel < -1.0 or t_rel > TRIM_DURATION + 1.0:
                continue

            t_video.append(t_rel)
            red_shaft_angle.append(red_angle)
            jaw_angle.append(measured_jaw_angle)
            # yellow_shaft_angle.append(yellow_angle)
            green_shaft_angle.append(green_angle)

    if detected_dof == 4:
        valid_filtered = sum(
            value is not None
            for value in jaw_angle
        )

        if valid_filtered == 0:
            raise RuntimeError(
                "No filtered DOF4 jaw-angle measurements found in "
                f"{file_path}. Re-run the webcam angle detector."
            )

    if detected_dof == 2:
        valid_filtered = sum(
            value is not None
            for value in red_shaft_angle
        )

        if valid_filtered == 0:
            raise RuntimeError(
                "No filtered DOF2 video-angle measurements found in "
                f"{file_path}. Re-run the webcam angle detector."
            )

    print("video samples:", len(t_video))
    print(
        "valid filtered red-shaft angles:",
        sum(value is not None for value in red_shaft_angle),
    )
    print(
        "valid filtered jaw angles:",
        sum(value is not None for value in jaw_angle),
    )

    return (
        t_video,
        red_shaft_angle,
        jaw_angle,
        # yellow_shaft_angle,
        green_shaft_angle,
    )


# Instrument Current DT replay data loading
# -----------------------------------------
# def default_instrument_current_replay_file(file_path):
#     file_path = Path(file_path)
#     name = file_path.name

#     if name.endswith("_ros_log.jsonl"):
#         replay_name = name.replace(
#             "_ros_log.jsonl",
#             "_motor_dt_replay.jsonl",
#         )
#     else:
#         replay_name = file_path.stem + "_motor_dt_replay.jsonl"
#     return file_path.with_name(replay_name)

# def load_instrument_current_dt_replay(file_path):
#     t = []
#     predicted_currents = [[], [], [], []]
#     filtered_currents = [[], [], [], []]
#     current_deviation = [[], [], [], []]
#     active_dof_values = []

#     if file_path is None:
#         return None

#     if not os.path.exists(file_path):
#         print(f"Instrument Current DT replay not found: {file_path}")
#         return None

#     with open(file_path, "r") as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             data = json.loads(line)

#             time_value = data.get("time")
#             pred = data.get("offline_predicted_currents")
#             filt = data.get("filtered_measured_currents")
#             dev = data.get("filtered_current_deviation")
#             active_dof = data.get("active_dof")

#             if time_value is None or pred is None:
#                 continue

#             if len(pred) < 4:
#                 continue

#             t.append(float(time_value))
#             active_dof_values.append(active_dof)

#             for motor_index in range(4):
#                 predicted_currents[motor_index].append(pred[motor_index])

#                 if filt is not None and len(filt) > motor_index:
#                     filtered_currents[motor_index].append(filt[motor_index])
#                 else:
#                     filtered_currents[motor_index].append(None)

#                 if dev is not None and len(dev) > motor_index:
#                     current_deviation[motor_index].append(dev[motor_index])
#                 else:
#                     current_deviation[motor_index].append(None)

#     if not t:
#         print(f"No usable Instrument Current DT replay samples found in: {file_path}")
#         return None

#     t0 = t[0]
#     t = [x - t0 for x in t]

#     return {
#         "time": t,
#         "predicted_currents": predicted_currents,
#         "filtered_currents": filtered_currents,
#         "current_deviation": current_deviation,
#         "active_dof_values": active_dof_values,
#     }

# Sequence parameter parsing
# -----------------------------------------
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

def get_sequence_evaluation_window(
    params_file,
    active_dof,
    t,
    commanded,
):
    """
    Bepaal het primaire evaluatievenster van een instrumentsequence.

    Start:
    eerste daadwerkelijke afwijking van de neutrale commandowaarde.

    Einde:
    start + volledige sequence-duur uit de YAML, inclusief:
    - alle waveform-stages;
    - pauzes tussen waveform-stages;
    - pauzes tussen frequenties;
    - pauzes tussen ranges;
    - laatste settlingpauze.
    """
    if params_file is None or active_dof is None:
        return None

    with open(params_file, "r") as f:
        config = yaml.safe_load(f)

    params = extract_ros_parameters(config)

    dof_name = f"dof{active_dof + 1}"

    if dof_name not in params:
        return None

    dof_params = params[dof_name]

    if dof_params.get("mode", "constant") != "sequence":
        return None

    tt = np.asarray(t, dtype=float)
    command = clean_numeric_array(commanded[active_dof])

    if len(tt) < 2:
        return None

    neutral_value = float(
        dof_params.get("value", 0.0)
    )

    moving_indices = np.where(
        np.isfinite(command)
        & (np.abs(command - neutral_value) > 1e-4)
    )[0]

    if len(moving_indices) == 0:
        print(
            f"Warning: no movement detected for {dof_name}; "
            "metrics will not be calculated."
        )
        return None

    # Twee samples terug om de eerste commandostijging volledig mee te nemen.
    first_index = max(
        0,
        int(moving_indices[0]) - 2,
    )

    evaluation_start = float(tt[first_index])

    base_frequency = float(
        dof_params.get("frequency", 0.1)
    )

    frequency_factors = [
        float(value)
        for value in dof_params.get(
            "sequence_frequency_factors",
            [1.0],
        )
    ]

    range_factors = [
        float(value)
        for value in dof_params.get(
            "sequence_range_factors",
            [1.0],
        )
    ]

    modes = list(
        dof_params.get(
            "sequence_modes",
            ["sinusoid", "triangle"],
        )
    )

    cycles = [
        float(value)
        for value in dof_params.get(
            "sequence_cycles",
            [3, 3],
        )
    ]

    if len(modes) != len(cycles):
        raise ValueError(
            f"{dof_name}: sequence_modes and sequence_cycles "
            "must have the same length."
        )

    stage_pause = float(
        dof_params.get(
            "sequence_pause_duration",
            0.0,
        )
    )

    between_frequency_pause = float(
        dof_params.get(
            "sequence_between_frequency_pause",
            0.0,
        )
    )

    between_range_pause = float(
        dof_params.get(
            "sequence_between_range_pause",
            0.0,
        )
    )

    final_pause = float(
        dof_params.get(
            "sequence_final_pause_duration",
            0.0,
        )
    )

    total_duration = 0.0

    for range_index, _range_factor in enumerate(
        range_factors
    ):
        for frequency_index, frequency_factor in enumerate(
            frequency_factors
        ):
            frequency = (
                base_frequency
                * frequency_factor
            )

            motion_duration = sum(
                cycle_count / frequency
                for cycle_count in cycles
            )

            number_of_stage_pauses = max(
                0,
                len(modes) - 1,
            )

            block_duration = (
                motion_duration
                + number_of_stage_pauses
                * stage_pause
            )

            total_duration += block_duration

            is_last_frequency = (
                frequency_index
                == len(frequency_factors) - 1
            )

            if not is_last_frequency:
                total_duration += (
                    between_frequency_pause
                )

        is_last_range = (
            range_index
            == len(range_factors) - 1
        )

        if not is_last_range:
            total_duration += between_range_pause

    total_duration += final_pause

    requested_end = (
        evaluation_start
        + total_duration
    )

    available_end = float(tt[-1])
    tolerance_s = 0.2

    if requested_end > available_end + tolerance_s:
        print(
            f"Warning: log shorter than expected sequence for "
            f"{dof_name}: required until "
            f"{requested_end:.3f} s, available until "
            f"{available_end:.3f} s."
        )

        return None

    evaluation_end = min(
        requested_end,
        available_end,
    )

    return {
        "start_s": evaluation_start,
        "end_s": evaluation_end,
        "duration_s": (
            evaluation_end
            - evaluation_start
        ),
        "source": "command_start_plus_yaml_sequence_duration",
        "active_dof": dof_name,
        "base_frequency_hz": base_frequency,
        "frequency_factors": frequency_factors,
        "range_factors": range_factors,
    }

# Prediction metrics computation
# -----------------------------------------
def compute_prediction_metrics(
    video_t,
    video_angle_deg,
    prediction_t,
    prediction_deg,
    evaluation_window=None,
):
    video_t = np.asarray(video_t, dtype=float)

    video_angle_deg = np.asarray(
        [
            np.nan if value is None else value
            for value in video_angle_deg
        ],
        dtype=float,
    )

    prediction_t = np.asarray(
        prediction_t,
        dtype=float,
    )

    prediction_deg = np.asarray(
        prediction_deg,
        dtype=float,
    )

    valid_video = (
        np.isfinite(video_t)
        & np.isfinite(video_angle_deg)
    )

    valid_prediction = (
        np.isfinite(prediction_t)
        & np.isfinite(prediction_deg)
    )

    if np.sum(valid_video) < 2:
        return None

    if np.sum(valid_prediction) < 2:
        return None

    video_t = video_t[valid_video]
    video_angle_deg = video_angle_deg[valid_video]

    prediction_t = prediction_t[valid_prediction]
    prediction_deg = prediction_deg[valid_prediction]

    overlap_start = max(
        video_t[0],
        prediction_t[0],
    )

    overlap_end = min(
        video_t[-1],
        prediction_t[-1],
    )

    if evaluation_window is not None:
        overlap_start = max(
            overlap_start,
            evaluation_window["start_s"],
        )

        overlap_end = min(
            overlap_end,
            evaluation_window["end_s"],
        )

    if overlap_end <= overlap_start:
        return None

    overlap_mask = (
        (video_t >= overlap_start)
        & (video_t <= overlap_end)
    )

    if np.sum(overlap_mask) < 2:
        return None

    video_t_overlap = video_t[overlap_mask]
    video_angle_overlap = video_angle_deg[overlap_mask]

    prediction_at_video_time = np.interp(
        video_t_overlap,
        prediction_t,
        prediction_deg,
    )

    # residual = measurement - prediction
    error = (
        video_angle_overlap
        - prediction_at_video_time
    )

    return {
        "n_samples": int(len(error)),
        "mae_deg": float(
            np.mean(np.abs(error))
        ),
        "rmse_deg": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "bias_deg": float(
            np.mean(error)
        ),
        "max_abs_error_deg": float(
            np.max(np.abs(error))
        ),
        "overlap_start_s": float(
            video_t_overlap[0]
        ),
        "overlap_end_s": float(
            video_t_overlap[-1]
        ),
    }

# def write_dof2_prediction_metrics(
#     plot_dir,
#     physics_metrics,
#     hybrid_metrics,
#     used_prediction_name,
# ):
#     metrics_path = os.path.join(plot_dir, "dof2_prediction_metrics.txt")

#     def write_block(f, title, metrics):
#         f.write(f"{title}\n")
#         f.write("-" * len(title) + "\n")

#         if metrics is None:
#             f.write("Not available\n\n")
#             return

#         f.write(f"MAE:             {metrics['mae_deg']:.3f} deg\n")
#         f.write(f"RMSE:            {metrics['rmse_deg']:.3f} deg\n")
#         f.write(f"Bias:            {metrics['bias_deg']:.3f} deg\n")
#         f.write(f"Max abs error:   {metrics['max_abs_error_deg']:.3f} deg\n")
#         f.write(f"Samples:         {metrics['n_samples']}\n")
#         f.write(
#             f"Time window:     "
#             f"{metrics['overlap_start_s']:.3f} - "
#             f"{metrics['overlap_end_s']:.3f} s\n\n"
#         )

#     with open(metrics_path, "w") as f:
#         f.write("DOF2 prediction metrics\n")
#         f.write("=======================\n\n")
#         f.write("Error definition:\n")
#         f.write("prediction_error = prediction_deg - video_measurement_deg\n\n")
#         f.write(f"Prediction shown in overview plot: {used_prediction_name}\n\n")

#         write_block(f, "Physics-based DT", physics_metrics)
#         write_block(f, "Hybrid DT", hybrid_metrics)

#     print(f"Saved DOF2 prediction metrics: {metrics_path}")
#     return metrics_path

def write_video_prediction_metrics(
    plot_dir,
    active_dof_name,
    video_signal_name,
    prediction_blocks,
):
    metrics_path = os.path.join(
        plot_dir,
        "instrument_video_prediction_metrics.txt",
    )

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
        f.write("Instrument video prediction metrics\n")
        f.write("===================================\n\n")
        f.write(f"DOF:          {active_dof_name}\n")
        f.write(f"Video signal: {video_signal_name}\n\n")
        f.write("Video preprocessing:\n")
        f.write("centred median filter, window = 5 frames\n")
        f.write("offline centred window; no fixed causal filter delay\n\n")

        f.write("Residual definition:\n")
        f.write(
            "residual = median_filtered_video_measurement_deg "
            "- prediction_deg\n\n"
        )
        for title, metrics in prediction_blocks:
            write_block(f, title, metrics)

    print(f"Saved video prediction metrics: {metrics_path}")
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


# Basic diagnostic plotting functions
#  -----------------------------------------
def plot_instrument_angles(plot_dir, t, commanded, measured_instrument_angles):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Instrument angles: commanded vs measured/controller output",
        fontsize=16,
    )

    for dof_index in range(4):
        axes[dof_index].plot(
            t,
            commanded[dof_index],
            label=f"commanded dof{dof_index + 1}",
        )

        axes[dof_index].plot(
            t,
            measured_instrument_angles[dof_index],
            linestyle="-",
            label=f"measured/controller dof{dof_index + 1}",
        )

        axes[dof_index].set_ylabel(f"DOF{dof_index + 1} [rad]")
        axes[dof_index].grid(True)
        axes[dof_index].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "instrument_angles.png"), dpi=200)
    plt.close(fig)

def plot_motor_positions(plot_dir, t, motor_pos):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Measured motor positions from encoders", fontsize=16)

    for motor_index in range(4):
        axes[motor_index].plot(
            t,
            motor_pos[motor_index],
            label=f"motor {motor_index}",
        )

        axes[motor_index].set_ylabel("Δ pulses")
        axes[motor_index].grid(True)
        axes[motor_index].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "motor_positions.png"), dpi=200)
    plt.close(fig)

# def plot_motor_currents(plot_dir, t, currents):
#     fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle("Measured motor currents", fontsize=16)

#     for motor_index in range(4):
#         axes[motor_index].plot(
#             t,
#             currents[motor_index],
#             label=f"motor {motor_index}",
#         )

#         axes[motor_index].set_ylabel("Current [mA]")
#         axes[motor_index].grid(True)
#         axes[motor_index].legend(fontsize=8)

#     axes[-1].set_xlabel("Time [s]")

#     plt.tight_layout()
#     fig.savefig(os.path.join(plot_dir, "motor_currents.png"), dpi=200)
#     plt.close(fig)

def plot_gearbox_state(plot_dir, t, gearbox):
    fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    fig.suptitle("Gearbox DT prediction", fontsize=16)

    for gearbox_index in range(6):
        axes[gearbox_index].plot(
            t,
            gearbox[gearbox_index],
            linestyle="--",
            label=GEARBOX_LABELS[gearbox_index],
        )

        axes[gearbox_index].set_ylabel(GEARBOX_LABELS[gearbox_index])
        axes[gearbox_index].grid(True)
        axes[gearbox_index].legend(fontsize=8)

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "gearbox_state.png"), dpi=200)
    plt.close(fig)


# DOF specific plotting functions
# -----------------------------------------

# DOF2 specific video validation plotting
def plot_dof2_video_validation(plot_dir, t, commanded, current_angles, motor_pos, video_t, video_angle, predicted_angles=None, hybrid_predicted_angles=None,):
    motor2_minus_motor1 = [
        m2 - m1 for m1, m2 in zip(motor_pos[1], motor_pos[2])
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("DOF2 validation: command, motor difference, and video angle", fontsize=16)

    axes[0].plot(t, commanded[1], linestyle="-", label="commanded dof2 / pitch [rad]")
    axes[0].plot(t, current_angles[1], linestyle="-", label="current dof2 / pitch from motors [rad]")
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
        linestyle="-",
        linewidth=2,
        label="video measured angle [deg]"
    )
    has_hybrid_bend = False

    if hybrid_predicted_angles is not None:
        hybrid_bend_deg = radians_to_deg_array(hybrid_predicted_angles[1])
        has_hybrid_bend = np.isfinite(hybrid_bend_deg).any()

    if has_hybrid_bend:
        axes[2].plot(
            t,
            hybrid_bend_deg,
            label="Hybrid DT bend [deg]",
            linestyle="--",
            linewidth=1.2,
            color="green",
        )

    elif predicted_angles is not None:
        predicted_bend_deg = radians_to_deg_array(predicted_angles[1])
        axes[2].plot(
            t,
            predicted_bend_deg,
            label="Physics-based DT bend [deg]",
            linestyle="--",
            linewidth=1.2,
            color="purple",
        )

    axes[2].set_ylabel("Angle [deg]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(plot_dir, "dof2_video_validation.png"), dpi=200)
    plt.close(fig)


# DOF1 and DOF3 specific wrapped rotation plotting
# def plot_current_vs_wrapped_angle(
#     plot_dir,
#     t,
#     angle_deg,
#     motor_current,
#     filename,
#     title,
#     current_label,
#     y_limit=(0, 250),
# ):
#     angle_deg = np.array(angle_deg, dtype=float)
#     angle_rad = np.radians(angle_deg)
#     wrapped_rad = wrap_to_pi(angle_rad)

#     motor_current = np.array(motor_current, dtype=float)
#     t_arr = np.array(t, dtype=float)

#     valid = (
#         np.isfinite(wrapped_rad)
#         & np.isfinite(motor_current)
#         & np.isfinite(t_arr)
#     )

#     wrapped_rad = wrapped_rad[valid]
#     motor_current = motor_current[valid]
#     t_arr = t_arr[valid]

#     # Alleen het bewegende deel gebruiken.
#     moving = np.abs(wrapped_rad) > 0.15

#     if np.sum(moving) < 10:
#         return

#     first = np.where(moving)[0][0]
#     last = np.where(moving)[0][-1] + 1

#     wrapped_rad = wrapped_rad[first:last]
#     motor_current = motor_current[first:last]
#     t_arr = t_arr[first:last]

#     # Split op wrap-jumps: van +pi naar -pi of andersom.
#     jumps = np.where(np.abs(np.diff(wrapped_rad)) > np.pi)[0] + 1
#     cycle_edges = np.r_[0, jumps, len(wrapped_rad)]

#     cycle_edges = [
#         (start, end)
#         for start, end in zip(cycle_edges[:-1], cycle_edges[1:])
#         if end - start > 20
#     ]

#     # Eerste en laatste onvolledige rotatie weggooien.
#     if len(cycle_edges) > 2:
#         cycle_edges = cycle_edges[1:-1]

#     fig, ax = plt.subplots(figsize=(10, 6))

#     cycle_count = 0

#     for start, end in cycle_edges:
#         if end - start < 20:
#             continue

#         angle_cycle = wrapped_rad[start:end]
#         current_cycle = motor_current[start:end]

#         angle_range = np.nanmax(angle_cycle) - np.nanmin(angle_cycle)

#         # Alleen bijna volledige -pi tot pi stukken gebruiken.
#         if angle_range < 0.8 * 2.0 * np.pi:
#             continue

#         cycle_count += 1

#         ax.plot(
#             angle_cycle,
#             current_cycle,
#             linewidth=1.2,
#             alpha=0.8,
#             # label=f"Rotation {cycle_count}",
#         )

#     ax.set_title(title)
#     ax.set_xlabel("Wrapped angle [rad]")
#     ax.set_ylabel(current_label)
#     ax.set_xlim(-np.pi, np.pi)

#     if y_limit is not None:
#         ax.set_ylim(*y_limit)

#     ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
#     ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
#     ax.grid(True)

#     if cycle_count > 0:
#         ax.legend(fontsize=8)

#     plt.tight_layout()
#     fig.savefig(os.path.join(plot_dir, filename), dpi=200)
#     plt.close(fig)


# later all dof? 
# def plot_dof3_instrument_current_prediction(plot_dir, instrument_current_dt):
#     if instrument_current_dt is None:
#         return

#     t = np.array(instrument_current_dt["time"], dtype=float)
#     predicted_currents = instrument_current_dt["predicted_currents"]
#     filtered_currents = instrument_current_dt["filtered_currents"]
#     current_deviation = instrument_current_dt["current_deviation"]

#     def clean_array(values):
#         return np.array(
#             [np.nan if value is None else value for value in values],
#             dtype=float,
#         )

#     fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
#     fig.suptitle("DOF3 Instrument Current DT prediction", fontsize=16)

#     # Plot 1: motor 0 measured vs predicted current
#     motor_index = 0
#     axes[0].plot(
#         t,
#         clean_array(filtered_currents[motor_index]),
#         linewidth=1.2,
#         label="filtered measured current motor 0",
#     )

#     axes[0].plot(
#         t,
#         clean_array(predicted_currents[motor_index]),
#         linestyle="--",
#         linewidth=1.3,
#         label="predicted current motor 0",
#     )

#     axes[0].set_ylabel("Current [mA]")
#     axes[0].set_title("Motor 0: measured vs predicted current")
#     axes[0].grid(True)
#     axes[0].legend(fontsize=8)

#     # Plot 2: motor 3 measured vs predicted current
#     motor_index = 3
#     axes[1].plot(
#         t,
#         clean_array(filtered_currents[motor_index]),
#         linewidth=1.2,
#         label="filtered measured current motor 3",
#     )

#     axes[1].plot(
#         t,
#         clean_array(predicted_currents[motor_index]),
#         linestyle="--",
#         linewidth=1.3,
#         label="predicted current motor 3",
#     )

#     axes[1].set_ylabel("Current [mA]")
#     axes[1].set_title("Motor 3: measured vs predicted current")
#     axes[1].grid(True)
#     axes[1].legend(fontsize=8)

#     # Plot 3: absolute residual only
#     for motor_index in [0, 3]:
#         residual = clean_array(current_deviation[motor_index])

#         axes[2].plot(
#             t,
#             np.abs(residual),
#             linewidth=1.1,
#             label=f"absolute residual motor {motor_index}",
#         )

#     axes[2].set_ylabel("|Residual| [mA]")
#     axes[2].set_xlabel("Time [s]")
#     axes[2].set_title("Absolute current prediction error")
#     axes[2].grid(True)
#     axes[2].legend(fontsize=8)

#     plt.tight_layout(rect=[0, 0, 1, 0.94])
#     fig.savefig(
#         os.path.join(plot_dir, "dof3_instrument_current_prediction.png"),
#         dpi=200,
#     )
#     plt.close(fig)

# DOF4 specific video measurement plotting
def plot_gripper_video_measurements(
    plot_dir,
    video_t,
    jaw_video_angle,
    green_shaft_video_angle, #yellow_shaft_video_angle,
    red_shaft_video_angle,
):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Gripper video measurements", fontsize=16)

    axes[0].plot(
        video_t,
        jaw_video_angle,
        linestyle="-",
        linewidth=2,
        label="measured jaw angle: green vs red [deg]",
    )
    axes[0].set_ylabel("Jaw angle [deg]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        video_t,
        green_shaft_video_angle, #yellow_shaft_video_angle,
        linestyle="-",
        linewidth=2,
        label="measured green marker vs shaft [deg]",
    )
    axes[1].set_ylabel("Green-shaft [deg]")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        video_t,
        red_shaft_video_angle,
        linestyle="-",
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



# General overview plotting functions
# -----------------------------------------
def plot_active_command_axis(
    ax,
    t,
    commanded,
    current_angles,
    active_dof,
    active_dof_name,
):
    if active_dof is None:
        ax.text(
            0.5,
            0.5,
            "No active DOF detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Instrument command")
        ax.grid(True)
        return

    ax.plot(
        t,
        clean_numeric_array(commanded[active_dof]),
        label=f"commanded {active_dof_name}",
        linewidth=1.5,
        color="green",
    )

    measured_angle = clean_numeric_array(current_angles[active_dof])

    if np.isfinite(measured_angle).any():
        ax.plot(
            t,
            measured_angle,
            linestyle="-",
            label=f"controller output {active_dof_name}",
            linewidth=1.2,
            color="black",
        )

    ax.set_title("Instrument command")
    ax.set_ylabel("Angle [rad]")
    ax.grid(True)
    ax.legend(fontsize=8)


def plot_motor_positions_axis(ax, t, motor_pos):
    for motor_index in range(4):
        ax.plot(
            t,
            clean_numeric_array(motor_pos[motor_index]),
            label=f"motor {motor_index}",
            linewidth=1.2,
        )

    ax.set_title("Measured motor positions from encoders")
    ax.set_ylabel("Motor position\nΔ pulses")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=4)

# def get_relevant_current_motors(active_dof):
#     """
#     Return the most relevant motor currents for the compact overview plot.

#     active_dof is zero-based:
#     0 = DOF1
#     1 = DOF2
#     2 = DOF3
#     3 = DOF4
#     """
#     if active_dof == 0:
#         return [1, 2]

#     if active_dof == 1:
#         return [1, 2]

#     if active_dof == 2:
#         return [0]

#     if active_dof == 3:
#         return [0, 3]

#     return [0, 1, 2, 3]

# def plot_motor_currents_axis(
#     ax,
#     t,
#     currents,
#     instrument_current_dt=None,
#     show_prediction=False,
#     active_dof=None,
# ):
#     """
#     Current plot used in the overview figures.

#     Overview 1:
#     - show raw measured current for all motors.

#     Overview 2:
#     - show only filtered measured current and predicted current
#       for the most relevant motors.
#     """
#     if show_prediction and instrument_current_dt is not None:
#         current_dt_t = clean_numeric_array(instrument_current_dt["time"])
#         predicted_currents = instrument_current_dt["predicted_currents"]
#         filtered_currents = instrument_current_dt["filtered_currents"]

#         relevant_motors = get_relevant_current_motors(active_dof)
#         plotted_any_prediction = False

#         for motor_index in relevant_motors:
#             filtered_current = clean_numeric_array(
#                 filtered_currents[motor_index]
#             )

#             predicted_current = clean_numeric_array(
#                 predicted_currents[motor_index]
#             )

#             if np.isfinite(filtered_current).any():
#                 ax.plot(
#                     current_dt_t,
#                     filtered_current,
#                     linewidth=1.2,
#                     label=f"M{motor_index} filtered measured",
#                 )

#             if np.isfinite(predicted_current).any():
#                 ax.plot(
#                     current_dt_t,
#                     predicted_current,
#                     linestyle="--",
#                     linewidth=1.3,
#                     label=f"M{motor_index} predicted",
#                 )
#                 plotted_any_prediction = True

#         if plotted_any_prediction:
#             ax.set_title("Current DT prediction: relevant motors")
#         else:
#             ax.set_title("Measured motor current")
#     else:
#         for motor_index in range(4):
#             ax.plot(
#                 t,
#                 clean_numeric_array(currents[motor_index]),
#                 label=f"M{motor_index} raw measured",
#                 linewidth=1.0,
#             )

#         ax.set_title("Measured motor current")

#     ax.set_ylabel("Motor current\n[mA]")
#     ax.grid(True)
#     ax.legend(fontsize=8, ncol=2)

# def compute_current_prediction_metrics(
#     time_values,
#     filtered_current,
#     predicted_current,
#     evaluation_window=None,
# ):
#     time_values = np.asarray(
#         time_values,
#         dtype=float,
#     )

#     filtered_current = np.asarray(
#         filtered_current,
#         dtype=float,
#     )

#     predicted_current = np.asarray(
#         predicted_current,
#         dtype=float,
#     )

#     valid = (
#         np.isfinite(time_values)
#         & np.isfinite(filtered_current)
#         & np.isfinite(predicted_current)
#     )

#     if evaluation_window is not None:
#         valid &= (
#             time_values
#             >= evaluation_window["start_s"]
#         )

#         valid &= (
#             time_values
#             <= evaluation_window["end_s"]
#         )

#     if np.sum(valid) < 2:
#         return None

#     error = (
#         filtered_current[valid]
#         - predicted_current[valid]
#     )

#     return {
#         "n_samples": int(len(error)),
#         "mae_mA": float(
#             np.mean(np.abs(error))
#         ),
#         "rmse_mA": float(
#             np.sqrt(np.mean(error ** 2))
#         ),
#         "bias_mA": float(
#             np.mean(error)
#         ),
#         "max_abs_error_mA": float(
#             np.max(np.abs(error))
#         ),
#         "evaluation_start_s": float(
#             time_values[valid][0]
#         ),
#         "evaluation_end_s": float(
#             time_values[valid][-1]
#         ),
#     }


# def plot_instrument_current_prediction_per_motor(
#     plot_dir,
#     instrument_current_dt,
#     active_dof_name,
#     evaluation_window=None
# ):
#     if instrument_current_dt is None:
#         return

#     t = clean_numeric_array(instrument_current_dt["time"])
#     predicted_currents = instrument_current_dt["predicted_currents"]
#     filtered_currents = instrument_current_dt["filtered_currents"]
#     current_deviation = instrument_current_dt["current_deviation"]

#     fig, axes = plt.subplots(
#         4,
#         2,
#         figsize=(16, 12),
#         sharex=True,
#         sharey="col",
#     )

#     fig.suptitle(
#         f"Instrument Current DT prediction per motor: {active_dof_name}",
#         fontsize=16,
#     )

#     metrics_by_motor = {}

#     for motor_index in range(4):
#         filtered_current = clean_numeric_array(
#             filtered_currents[motor_index]
#         )

#         predicted_current = clean_numeric_array(
#             predicted_currents[motor_index]
#         )

#         residual = clean_numeric_array(
#             current_deviation[motor_index]
#         )

#         metrics = compute_current_prediction_metrics(
#             time_values=t,
#             filtered_current=filtered_current,
#             predicted_current=predicted_current,
#             evaluation_window=evaluation_window,
#         )

#         metrics_by_motor[motor_index] = metrics

#         # Left column: measured vs predicted current
#         axes[motor_index, 0].plot(
#             t,
#             filtered_current,
#             linewidth=1.2,
#             color = "orange",
#             label=f"M{motor_index} filtered measured",
#         )

#         axes[motor_index, 0].plot(
#             t,
#             predicted_current,
#             linestyle="--",
#             linewidth=1.2,
#             color = "tab:purple",
#             label=f"M{motor_index} predicted",
#         )

#         axes[motor_index, 0].set_ylabel("Current [mA]")
#         axes[motor_index, 0].set_title(
#             f"Motor {motor_index}: filtered measured vs predicted"
#         )
#         axes[motor_index, 0].grid(True)
#         axes[motor_index, 0].legend(fontsize=8)

#         # Right column: residual
#         axes[motor_index, 1].plot(
#             t,
#             residual,
#             linewidth=1.1,
#             color = "#4C78A8",
#             label=f"M{motor_index} residual",
#         )

#         axes[motor_index, 1].axhline(
#             0.0,
#             linewidth=1.0,
#             color="black",
#         )

#         axes[motor_index, 1].set_ylabel("Residual [mA]")
#         axes[motor_index, 1].set_title(
#             f"Motor {motor_index}: residual = filtered measured - predicted"
#         )
#         axes[motor_index, 1].grid(True)
#         axes[motor_index, 1].legend(fontsize=8)

#         if metrics is not None:
#             text = (
#                 f"MAE = {metrics['mae_mA']:.2f} mA\n"
#                 f"RMSE = {metrics['rmse_mA']:.2f} mA"
#                 # f"Bias = {metrics['bias_mA']:.2f} mA"
#             )

#             axes[motor_index, 1].text(
#                 0.01,
#                 0.95,
#                 text,
#                 transform=axes[motor_index, 1].transAxes,
#                 va="top",
#                 ha="left",
#                 fontsize=8,
#                 bbox=dict(boxstyle="round", alpha=0.8),
#             )

#     axes[-1, 0].set_xlabel("Time [s]")
#     axes[-1, 1].set_xlabel("Time [s]")

#     plt.tight_layout(rect=[0, 0, 1, 0.94])

#     plot_path = os.path.join(
#         plot_dir,
#         "motor_current_prediction_residuals.png",
#     )

#     fig.savefig(plot_path, dpi=200)
#     plt.close(fig)

#     metrics_path = os.path.join(
#         plot_dir,
#         "motor_current_prediction_metrics.txt",
#     )

#     with open(metrics_path, "w") as f:
#         f.write("Motor current prediction metrics\n")
#         f.write("========================================\n\n")
#         f.write("Error definition:\n")
#         f.write("residual = filtered_measured_current - predicted_current\n\n")
#         if evaluation_window is not None:
#             f.write("Evaluation window:\n")
#             f.write(
#                 f"{evaluation_window['start_s']:.3f} - "
#                 f"{evaluation_window['end_s']:.3f} s\n"
#             )
#             f.write(
#                 f"Duration: "
#                 f"{evaluation_window['duration_s']:.3f} s\n"
#             )
#             f.write(
#                 "Source: command start + YAML sequence duration\n\n"
#             )

#         for motor_index in range(4):
#             metrics = metrics_by_motor[motor_index]

#             f.write(f"Motor {motor_index}\n")
#             f.write("-" * 20 + "\n")

#             if metrics is None:
#                 f.write("Not available\n\n")
#                 continue

#             f.write(f"MAE:             {metrics['mae_mA']:.3f} mA\n")
#             f.write(f"RMSE:            {metrics['rmse_mA']:.3f} mA\n")
#             # f.write(f"Bias:            {metrics['bias_mA']:.3f} mA\n")
#             f.write(f"Max abs error:   {metrics['max_abs_error_mA']:.3f} mA\n")
#             f.write(f"Samples:         "f"{metrics['n_samples']}\n")
#             f.write(f"Time window:     "f"{metrics['evaluation_start_s']:.3f} - "f"{metrics['evaluation_end_s']:.3f} s\n")

#     print(f"Saved Instrument Current DT plot: {plot_path}")
#     print(f"Saved Instrument Current DT metrics: {metrics_path}")
    
def plot_video_instrument_measurement_axis(
    ax,
    active_dof,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
):
    if len(video_t) == 0:
        ax.text(
            0.5,
            0.5,
            "No video measurement loaded",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Video-measured instrument output")
        ax.set_ylabel("Angle [deg]")
        ax.grid(True)
        return

    if active_dof == 3:
        ax.plot(
            video_t,
            jaw_video_angle,
            linestyle="-",
            markersize=3,
            color="orange",
            label="video measured jaw angle [deg]",
        )
        ax.set_ylabel("Jaw angle [deg]")
    else:
        ax.plot(
            video_t,
            red_shaft_video_angle,
            linestyle="-",
            linewidth=2,
            color="orange",
            label="video measured red marker vs shaft [deg]",
        )
        ax.set_ylabel("Angle [deg]")

    ax.set_title("Video-measured instrument output")
    ax.grid(True)
    ax.legend(fontsize=8)


def plot_relevant_gearbox_axis(ax, t, gearbox, active_dof):
    if active_dof == 3:
        ax.plot(
            t,
            gearbox[2],
            linestyle="--",
            label="Gearbox DT: inner shaft translation [mm]",
            color="blue",
        )
        ax.set_ylabel("Translation [mm]")
        ax.set_title("Gearbox DT prediction: inner shaft translation")

    elif active_dof == 2:
        ax.plot(
            t,
            gearbox[0],
            linestyle="--",
            label="Gearbox DT: inner shaft rotation [deg]",
            color="blue",
        )
        ax.set_ylabel("Rotation [deg]")
        ax.set_title("Gearbox DT prediction: inner shaft rotation")

    else:
        ax.plot(
            t,
            gearbox[3],
            linestyle="--",
            label="Gearbox DT: middle shaft rotation [deg]",
        )
        ax.plot(
            t,
            gearbox[4],
            linestyle="--",
            label="Gearbox DT: outer shaft rotation [deg]",
        )
        ax.plot(
            t,
            gearbox[5],
            linestyle="--",
            label="Gearbox DT: middle-outer relative rotation [deg]",
            color="blue",
        )
        ax.set_ylabel("Rotation [deg]")
        ax.set_title("Gearbox DT prediction: shaft rotations")

    ax.grid(True)
    ax.legend(fontsize=8)

def get_video_prediction_setup(
    active_dof,
    predicted_angles,
    hybrid_predicted_angles,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
):
    """
    Build the video-vs-prediction comparison for the active DOF.

    Returns:
    - video signal
    - video signal name
    - prediction blocks with label, prediction signal, color, linestyle
    """

    if active_dof == 1:
        # DOF2 bend validation: video red marker vs shaft.
        predicted_bend_deg = radians_to_deg_array(predicted_angles[1])
        hybrid_bend_deg = radians_to_deg_array(hybrid_predicted_angles[1])

        prediction_blocks = [
            {
                "name": "Physics-based DT bend",
                "values_deg": predicted_bend_deg,
                "color": "#8064A2",
                "linestyle": "--",
            }
        ]

        if np.isfinite(hybrid_bend_deg).any():
            prediction_blocks.append(
                {
                    "name": "Hybrid DT bend",
                    "values_deg": hybrid_bend_deg,
                    "color": "green",
                    "linestyle": "--",
                }
            )

        return {
            "video_t": video_t,
            "video_values_deg": red_shaft_video_angle,
            "video_label": "median-filtered video red marker vs shaft [deg]",
            "video_signal_name": "median-filtered red marker vs shaft angle [deg]",
            "video_marker_style": "-",
            "prediction_blocks": prediction_blocks,
        }

    if active_dof == 3:
        # DOF4 gripper validation: video jaw angle.
        predicted_jaw_deg = radians_to_deg_array(predicted_angles[3])

        return {
            "video_t": video_t,
            "video_values_deg": jaw_video_angle,
            "video_label": "median-filtered video angle between jaws [deg]",
            "video_signal_name": "median-filtered jaw angle [deg]",
            "video_marker_style": "-",
            "prediction_blocks": [
                {
                    "name": "Instrument DT gripper/articulation",
                    "values_deg": predicted_jaw_deg,
                    "color": "#8064A2",
                    "linestyle": "--",
                }
            ],
        }

    if active_dof is not None:
        # Generic fallback for DOF1/DOF3 if video is available.
        predicted_deg = radians_to_deg_array(predicted_angles[active_dof])

        return {
            "video_t": video_t,
            "video_values_deg": red_shaft_video_angle,
            "video_label": "median-filtered video red marker vs shaft [deg]",
            "video_signal_name": "median-filtered red marker vs shaft angle [deg]",
            "video_marker_style": "-",
            "prediction_blocks": [
                {
                    "name": f"Instrument DT dof{active_dof + 1}",
                    "values_deg": predicted_deg,
                    "color": "#8064A2",
                    "linestyle": "--",
                }
            ],
        }

    return None

def plot_instrument_prediction_axis(
    ax,
    plot_dir,
    t,
    active_dof,
    active_dof_name,
    predicted_angles,
    hybrid_predicted_angles,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    evaluation_window=None,
):
    setup = get_video_prediction_setup(
        active_dof=active_dof,
        predicted_angles=predicted_angles,
        hybrid_predicted_angles=hybrid_predicted_angles,
        video_t=video_t,
        red_shaft_video_angle=red_shaft_video_angle,
        jaw_video_angle=jaw_video_angle,
    )

    if setup is None:
        ax.text(
            0.5,
            0.5,
            "No active DOF detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Instrument DT prediction")
        ax.set_ylabel("Angle [deg]")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        return

    prediction_metric_blocks = []

    for block in setup["prediction_blocks"]:
        prediction_values_deg = block["values_deg"]

        metrics = compute_prediction_metrics(
            video_t=setup["video_t"],
            video_angle_deg=setup["video_values_deg"],
            prediction_t=t,
            prediction_deg=prediction_values_deg,
            evaluation_window=evaluation_window,
        )

        prediction_metric_blocks.append(
            (
                block["name"],
                metrics,
            )
        )

        ax.plot(
            t,
            prediction_values_deg,
            label=f"{block['name']} [deg]",
            linewidth=1.5,
            color=block["color"],
            linestyle=block["linestyle"],
        )

    # Show metrics box for the most relevant prediction.
    # Prefer Hybrid DT when available, otherwise use the last/only block.
    shown_metrics_name, shown_metrics = prediction_metric_blocks[-1]

    add_metrics_text_box(
        ax,
        shown_metrics_name,
        shown_metrics,
    )

    if len(setup["video_t"]) > 0:
        if setup["video_marker_style"] == "-":
            ax.plot(
                setup["video_t"],
                setup["video_values_deg"],
                "-",
                markersize=3,
                color="orange",
                label=setup["video_label"],
            )
        else:
            ax.plot(
                setup["video_t"],
                setup["video_values_deg"],
                "-",
                linewidth=2,
                color="orange",
                label=setup["video_label"],
            )

    if len(setup["video_t"]) > 0:
        write_video_prediction_metrics(
            plot_dir=plot_dir,
            active_dof_name=active_dof_name,
            video_signal_name=setup["video_signal_name"],
            prediction_blocks=prediction_metric_blocks,
        )

    # Optional backwards-compatible DOF2 metrics file.
    # You may remove this later if the generic file is enough.
    if active_dof == 1:
        physics_metrics = None
        hybrid_metrics = None
        used_prediction_name = shown_metrics_name

        for name, metrics in prediction_metric_blocks:
            if name == "Physics-based DT bend":
                physics_metrics = metrics

            if name == "Hybrid DT bend":
                hybrid_metrics = metrics

        # write_dof2_prediction_metrics(
        #     plot_dir,
        #     physics_metrics,
        #     hybrid_metrics,
        #     used_prediction_name,
        # )

    if len(setup["video_t"]) == 0:
        ax.set_title("Instrument DT prediction")
    else:
        ax.set_title("Instrument DT prediction vs video measurement")

    ax.set_ylabel("Angle [deg]")
    ax.set_xlabel("Time [s]")
    ax.grid(True)
    ax.legend(fontsize=8)

def plot_instrument_position_prediction_residual(
    plot_dir,
    t,
    active_dof,
    predicted_angles,
    hybrid_predicted_angles,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    evaluation_window=None
):
    # active_dof uses zero-based indexing:
    # 1 = DOF2 bending, 3 = DOF4 gripper.
    if active_dof not in (1, 3):
        return
        
    setup = get_video_prediction_setup(
        active_dof=active_dof,
        predicted_angles=predicted_angles,
        hybrid_predicted_angles=hybrid_predicted_angles,
        video_t=video_t,
        red_shaft_video_angle=red_shaft_video_angle,
        jaw_video_angle=jaw_video_angle,
    )

    if setup is None or len(setup["video_t"]) < 2:
        return

    prediction_t = clean_numeric_array(t)
    video_time = clean_numeric_array(setup["video_t"])
    video_values = clean_numeric_array(setup["video_values_deg"])

    valid_video = np.isfinite(video_time) & np.isfinite(video_values)

    if np.sum(valid_video) < 2:
        return

    video_time = video_time[valid_video]
    video_values = video_values[valid_video]

    valid_rows = []

    for block in setup["prediction_blocks"]:
        prediction = clean_numeric_array(block["values_deg"])
        valid_prediction = (
            np.isfinite(prediction_t)
            & np.isfinite(prediction)
        )

        if np.sum(valid_prediction) < 2:
            continue

        prediction_time_valid = prediction_t[valid_prediction]
        prediction_valid = prediction[valid_prediction]

        overlap_start = prediction_time_valid[0]
        overlap_end = prediction_time_valid[-1]

        if evaluation_window is not None:
            overlap_start = max(
                overlap_start,
                evaluation_window["start_s"],
            )

            overlap_end = min(
                overlap_end,
                evaluation_window["end_s"],
            )

        overlap = (
            (video_time >= overlap_start)
            & (video_time <= overlap_end)
        )

        if np.sum(overlap) < 2:
            continue

        comparison_time = video_time[overlap]
        measured = video_values[overlap]

        predicted = np.interp(
            comparison_time,
            prediction_time_valid,
            prediction_valid,
        )

        # Positive residual means that the filtered video measurement
        # is larger than the Digital Twin prediction.
        residual = measured - predicted

        valid_rows.append(
            (block, comparison_time, measured, predicted, residual)
        )

    if not valid_rows:
        return

    fig, axes = plt.subplots(
        len(valid_rows),
        2,
        figsize=(16, 4 * len(valid_rows)),
        squeeze=False,
        sharex="col",
        sharey="col"
    )

    for row_index, row in enumerate(valid_rows):
        block, time_values, measured, predicted, residual = row

        axes[row_index, 0].plot(
            time_values,
            measured,
            linestyle="-",
            linewidth=2,
            color="#6BAED6",
            label=setup["video_label"],
        )

        axes[row_index, 0].plot(
            time_values,
            predicted,
            linestyle="--",
            linewidth=1.5,
            color=block["color"],
            label=f"{block['name']} [deg]",
        )

        axes[row_index, 0].set_ylabel("Angle [deg]")
        axes[row_index, 0].set_title(
            f"{block['name']} vs video measurement"
        )
        axes[row_index, 0].grid(True)
        axes[row_index, 0].legend(fontsize=8)

        mae = np.mean(np.abs(residual))
        rmse = np.sqrt(np.mean(residual ** 2))

        axes[row_index, 1].plot(
            time_values,
            residual,
            color="#4C78A8",
            linewidth=1.2,
            label="position residual",
        )

        axes[row_index, 1].axhline(
            0.0,
            color="black",
            linewidth=1.0,
        )

        axes[row_index, 1].set_ylabel("Residual [deg]")
        axes[row_index, 1].set_title(
            "Residual = filtered video measurement - prediction"
        )
        axes[row_index, 1].grid(True)
        axes[row_index, 1].legend(fontsize=8)

        axes[row_index, 1].text(
            0.01,
            0.95,
            f"MAE = {mae:.2f}°\nRMSE = {rmse:.2f}°",
            transform=axes[row_index, 1].transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.8),
        )

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    plt.tight_layout()

    plot_path = os.path.join(
        plot_dir,
        "instrument_position_prediction_residual.png",
    )

    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Saved position prediction residual plot: {plot_path}")

def plot_overview_inputs_outputs(
    plot_dir,
    pattern_name,
    active_dof_name,
    t,
    commanded,
    current_angles,
    motor_pos,
    # currents,
    coupling_mode,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    active_dof,
    params_file=None,
):
    """
    Plot the available input/output signals before adding DT predictions.

    This plot is meant to show:
    - commanded instrument motion
    - measured motor encoder positions
    - measured motor current feedback
    - video-measured instrument output
    """
    show_instrument = coupling_mode == "full_setup"
    nrows = 3 if show_instrument else 2

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(14, 4 * nrows),
        sharex=True,
    )

    fig.suptitle(
        f"Overview inputs and measured outputs: {active_dof_name}",
        fontsize=16,
    )

    plot_active_command_axis(
        axes[0],
        t,
        commanded,
        current_angles,
        active_dof,
        active_dof_name,
    )

    plot_motor_positions_axis(
        axes[1],
        t,
        motor_pos,
    )

    # plot_motor_currents_axis(
    #     axes[2],
    #     t,
    #     currents,
    #     instrument_current_dt=None,
    #     show_prediction=False,
    #     active_dof=active_dof,
    # )

    if show_instrument:
        plot_video_instrument_measurement_axis(
            axes[2],
            active_dof,
            video_t,
            red_shaft_video_angle,
            jaw_video_angle,
        )

    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    safe_pattern_name = safe_text_for_filename(pattern_name)
    safe_active_dof_name = safe_text_for_filename(active_dof_name)

    filename_base = (
        f"overview_inputs_outputs_"
        f"{safe_pattern_name}_{safe_active_dof_name}"
    )

    fig.savefig(os.path.join(plot_dir, f"{filename_base}.png"), dpi=200)

    # Cropped inputs/outputs overview per frequency block.
    frequency_segments = get_frequency_segments_from_params(
        params_file=params_file,
        active_dof=active_dof,
        t=t,
        commanded=commanded,
    )

    original_xlim = axes[0].get_xlim()

    for segment in frequency_segments:
        segment_start = max(float(t[0]), segment["start"] - SEGMENT_MARGIN_BEFORE)
        segment_end = min(float(t[-1]), segment["end"] + SEGMENT_MARGIN_AFTER)

        for ax in axes:
            ax.set_xlim(segment_start, segment_end)

        freq_text = safe_number_for_filename(segment["frequency"])
        freq_factor_text = safe_number_for_filename(segment["frequency_factor"])
        range_text = safe_number_for_filename(segment["range_factor"])

        segment_filename = (
            f"{filename_base}"
            f"_range{range_text}"
            f"_freq{freq_text}Hz"
            f"_freqfactor{freq_factor_text}.png"
        )

        fig.savefig(os.path.join(plot_dir, segment_filename), dpi=200)

    for ax in axes:
        ax.set_xlim(original_xlim)

    plt.close(fig)


def plot_overview_prediction(
    plot_dir,
    pattern_name,
    active_dof_name,
    t,
    commanded,
    current_angles,
    motor_pos,
    # currents,
    gearbox,
    predicted_angles,
    hybrid_predicted_angles,
    # instrument_current_dt,
    coupling_mode,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    active_dof,
    params_file=None,
    evaluation_window=None
):
    """
    Plot the DT prediction overview.

    This plot starts from the same input/output signals as the first overview,
    but adds:
    - Current DT prediction
    - Gearbox DT prediction
    - Instrument DT / Hybrid DT prediction
    """
    show_instrument = coupling_mode == "full_setup"
    nrows = 4 if show_instrument else 3

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(14, 3.2 * nrows),
        sharex=True,
    )

    fig.suptitle(
        f"Overview Digital Twin prediction: {active_dof_name}",
        fontsize=16,
    )

    plot_active_command_axis(
        axes[0],
        t,
        commanded,
        current_angles,
        active_dof,
        active_dof_name,
    )

    plot_motor_positions_axis(
        axes[1],
        t,
        motor_pos,
    )

    # plot_motor_currents_axis(
    #     axes[2],
    #     t,
    #     currents,
    #     instrument_current_dt=instrument_current_dt,
    #     show_prediction=True,
    #     active_dof=active_dof,
    # )

    plot_relevant_gearbox_axis(
        axes[2],
        t,
        gearbox,
        active_dof,
    )

    if show_instrument:
        plot_instrument_prediction_axis(
            axes[3],
            plot_dir,
            t,
            active_dof,
            active_dof_name,
            predicted_angles,
            hybrid_predicted_angles,
            video_t,
            red_shaft_video_angle,
            jaw_video_angle,
            evaluation_window=evaluation_window,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    safe_pattern_name = safe_text_for_filename(pattern_name)
    safe_active_dof_name = safe_text_for_filename(active_dof_name)

    filename_base = (
        f"overview_prediction_"
        f"{safe_pattern_name}_{safe_active_dof_name}"
    )

    fig.savefig(os.path.join(plot_dir, f"{filename_base}.png"), dpi=200)

    # Cropped prediction overview per frequency block.
    frequency_segments = get_frequency_segments_from_params(
        params_file=params_file,
        active_dof=active_dof,
        t=t,
        commanded=commanded,
    )

    original_xlim = axes[0].get_xlim()

    for segment in frequency_segments:
        segment_start = max(float(t[0]), segment["start"] - SEGMENT_MARGIN_BEFORE)
        segment_end = min(float(t[-1]), segment["end"] + SEGMENT_MARGIN_AFTER)

        for ax in axes:
            ax.set_xlim(segment_start, segment_end)

        freq_text = safe_number_for_filename(segment["frequency"])
        freq_factor_text = safe_number_for_filename(segment["frequency_factor"])
        range_text = safe_number_for_filename(segment["range_factor"])

        segment_filename = (
            f"{filename_base}"
            f"_range{range_text}"
            f"_freq{freq_text}Hz"
            f"_freqfactor{freq_factor_text}.png"
        )

        fig.savefig(os.path.join(plot_dir, segment_filename), dpi=200)

    for ax in axes:
        ax.set_xlim(original_xlim)

    plt.close(fig)
    
def plot_continuous_file(
    file_path,
    video_angle_file=None,
    output_dir=None,
    params_file=None,
    coupling_mode="full_setup",
):
    t, commanded, current_angles, predicted_angles, hybrid_predicted_angles, motor_pos, gearbox, motor_ros_t0 = load_continuous_file(file_path)
    basename = os.path.basename(file_path).replace(".jsonl", "")
    pattern_name = os.path.basename(os.path.dirname(file_path))

    if output_dir is None:
        plot_dir = os.path.join(os.path.dirname(file_path), "plots", basename)
    else:
        plot_dir = output_dir

    os.makedirs(plot_dir, exist_ok=True)


    # Creates multiple plots in the output directory, including;
    # 1. commanded vs current instrument angles
    # 2. motor positions
    # 3. motor currents
    # 4. gearbox state
    # 5. overview plot with active DOF, commanded vs current 
    # angles, motor positions, currents, gearbox state
    # and video measurement if available.

    active_dof = detect_active_dof(commanded)
    active_dof_name = f"dof{active_dof + 1}" if active_dof is not None else "unknown_dof"

    evaluation_window = get_sequence_evaluation_window(
        params_file=params_file,
        active_dof=active_dof,
        t=t,
        commanded=commanded,
    )

    if evaluation_window is None:
        print(
            "No valid sequence evaluation window found."
        )
    else:
        print(
            "Sequence evaluation window: "
            f"{evaluation_window['start_s']:.3f} - "
            f"{evaluation_window['end_s']:.3f} s "
            f"({evaluation_window['duration_s']:.3f} s)"
        )

    if video_angle_file is not None:
        (
            video_t,
            red_shaft_video_angle,
            jaw_video_angle,
            # yellow_shaft_video_angle,
            green_shaft_video_angle,
        ) = load_video_angle_file(
            video_angle_file,
            motor_ros_t0,
        )
    else:
        video_t = []
        red_shaft_video_angle = []
        jaw_video_angle = []
        # yellow_shaft_video_angle = []
        green_shaft_video_angle = []


    plot_motor_positions(
        plot_dir,
        t,
        motor_pos,
    )

    # plot_motor_currents(
    #     plot_dir,
    #     t,
    #     currents,
    # )

    plot_gearbox_state(
        plot_dir,
        t,
        gearbox,
    )

    if coupling_mode == "full_setup":
        plot_instrument_angles(
            plot_dir,
            t,
            commanded,
            current_angles,
        )

        plot_instrument_position_prediction_residual(
            plot_dir=plot_dir,
            t=t,
            active_dof=active_dof,
            predicted_angles=predicted_angles,
            hybrid_predicted_angles=hybrid_predicted_angles,
            video_t=video_t,
            red_shaft_video_angle=red_shaft_video_angle,
            jaw_video_angle=jaw_video_angle,
            evaluation_window=evaluation_window,
        )

        if active_dof == 1 and len(video_t) > 0:
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

        if active_dof == 3 and len(video_t) > 0:
            plot_gripper_video_measurements(
                plot_dir,
                video_t,
                jaw_video_angle,
                green_shaft_video_angle,
                red_shaft_video_angle,
            )
    

    # Overview 1: measured inputs and outputs only
    plot_overview_inputs_outputs(
        plot_dir=plot_dir,
        pattern_name=pattern_name,
        active_dof_name=active_dof_name,
        t=t,
        commanded=commanded,
        current_angles=current_angles,
        motor_pos=motor_pos,
        coupling_mode=coupling_mode,
        # currents=currents,
        video_t=video_t,
        red_shaft_video_angle=red_shaft_video_angle,
        jaw_video_angle=jaw_video_angle,
        active_dof=active_dof,
        params_file=params_file,
    )

    # Overview 2: same base signals, with DT predictions added
    plot_overview_prediction(
        plot_dir=plot_dir,
        pattern_name=pattern_name,
        active_dof_name=active_dof_name,
        t=t,
        commanded=commanded,
        current_angles=current_angles,
        motor_pos=motor_pos,
        coupling_mode=coupling_mode,
        # currents=currents,
        gearbox=gearbox,
        predicted_angles=predicted_angles,
        hybrid_predicted_angles=hybrid_predicted_angles,
        # instrument_current_dt=instrument_current_dt,
        video_t=video_t,
        red_shaft_video_angle=red_shaft_video_angle,
        jaw_video_angle=jaw_video_angle,
        active_dof=active_dof,
        params_file=params_file,
        evaluation_window=evaluation_window
    )

    # plot_instrument_current_prediction_per_motor(
    #     plot_dir=plot_dir,
    #     instrument_current_dt=instrument_current_dt,
    #     active_dof_name=active_dof_name,
    #     evaluation_window=evaluation_window,
    # )

    # plot_instrument_position_prediction_residual(
    #     plot_dir=plot_dir,
    #     t=t,
    #     active_dof=active_dof,
    #     predicted_angles=predicted_angles,
    #     hybrid_predicted_angles=hybrid_predicted_angles,
    #     video_t=video_t,
    #     red_shaft_video_angle=red_shaft_video_angle,
    #     jaw_video_angle=jaw_video_angle,
    #     evaluation_window=evaluation_window
    # )


    # if active_dof == 1:
    #     plot_dof2_video_validation(
    #     plot_dir,
    #     t,
    #     commanded,
    #     current_angles,
    #     motor_pos,
    #     video_t,
    #     red_shaft_video_angle,
    #     predicted_angles,
    #     hybrid_predicted_angles,
    # )

    # # if active_dof == 0:
    # #     plot_current_vs_wrapped_angle(
    # #         plot_dir=plot_dir,
    # #         t=t,
    # #         angle_deg=gearbox[4],
    # #         motor_current=currents[2],
    # #         filename="dof1_current_vs_wrapped_angle.png",
    # #         title="DOF1: motor current vs wrapped outer shaft rotation",
    # #         current_label="Motor 2 current [mA]",
    # #         y_limit=(0, 250),
    # #     )
    # # if active_dof == 2:
    # #     plot_current_vs_wrapped_angle(
    # #         plot_dir=plot_dir,
    # #         t=t,
    # #         angle_deg=gearbox[0],
    # #         motor_current=currents[0],
    # #         filename="dof3_current_vs_wrapped_angle.png",
    # #         title="DOF3: motor current vs wrapped inner shaft rotation",
    # #         current_label="Motor 0 current [mA]",
    # #         y_limit=(0, 250),
    # #     )

    #     # plot_dof3_instrument_current_prediction(
    #     #     plot_dir,
    #     #     instrument_current_dt,
    #     # )

    # if active_dof == 3:
    #     plot_gripper_video_measurements(
    #         plot_dir,
    #         video_t,
    #         jaw_video_angle,
    #         green_shaft_video_angle, #yellow_shaft_video_angle,
    #         red_shaft_video_angle,
    #     )    
    print(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--video-angles", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--coupling-mode",
        choices=["gearbox_only", "full_setup"],
        required=True,
    )
   
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
        coupling_mode=args.coupling_mode,
    )

    print("All plots saved.")