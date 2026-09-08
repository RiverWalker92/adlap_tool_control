#!/usr/bin/env python3

"""
Plot Gearbox and Instrument Digital Twin outputs for SATA DOF trials.

The script loads recorded DOF commands, controller-reported instrument angles,
Gearbox DT states, Instrument DT predictions, and optional camera measurements.

For full-setup trials, the plots compare Instrument DT and Hybrid DT predictions
with camera measurements and calculate position-prediction residual metrics.
Motor DT signals are handled separately by the Motor DT plotting script.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


DEFAULT_DOF_PATTERN_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "dof_pattern_params.yaml"
)
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
COMMAND_CHANGE_THRESHOLD = 1e-5
MOTION_MEMORY_S = 1.2

# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
def load_continuous_file(file_path):
    """
    Load Gearbox and Instrument DT signals from a continuous JSONL log file.
    """
    t = []
    commanded = [[], [], [], []]
    controller_angles = [[], [], [], []]
    gearbox = [[], [], [], [], [], []]
    predicted_angles = [[] for _ in range(8)]
    hybrid_predicted_angles = [[] for _ in range(8)]
    ros_timestamps = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            cmd = data.get("commanded_instrument_angles")
            controller_ang = data.get("measured_instrument_angles")
            if controller_ang is None:
                controller_ang = data.get("current_instrument_angles")
            gb = data.get("gearbox_state")
            pred_ang = data.get("predicted_instrument_angles")
            hybrid_pred_ang = data.get(
                "hybrid_predicted_instrument_angles"
            )

            if cmd is None or gb is None:
                continue

            if len(cmd) < 4 or len(gb) < 6:
                continue

            time_value = data.get("time")
            ros_timestamp = data.get("ros_timestamp")

            if time_value is None or ros_timestamp is None:
                continue

            t.append(float(time_value))
            ros_timestamps.append(float(ros_timestamp))

            for i in range(4):
                commanded[i].append(cmd[i])

                controller_angles[i].append(
                    controller_ang[i]
                    if controller_ang is not None and len(controller_ang) > i
                    else None
                )

            for i in range(6):
                gearbox[i].append(gb[i])

            for i in range(8):
                if pred_ang is not None and len(pred_ang) > i:
                    predicted_angles[i].append(pred_ang[i])
                else:
                    predicted_angles[i].append(None)

                if hybrid_pred_ang is not None and len(hybrid_pred_ang) > i:
                    hybrid_predicted_angles[i].append(hybrid_pred_ang[i])
                else:
                    hybrid_predicted_angles[i].append(None)

    if not t:
        raise RuntimeError(
            f"No usable Gearbox/Instrument DT samples found in {file_path}"
        )

    ros_t0 = ros_timestamps[0]
    t0 = t[0]
    t = [value - t0 for value in t]

    for i in range(8):
        first_valid = next(
            (value for value in predicted_angles[i] if value is not None),
            None,
        )

        if first_valid is not None:
            predicted_angles[i] = [
                value - first_valid if value is not None else None
                for value in predicted_angles[i]
            ]

        first_valid = next(
            (
                value
                for value in hybrid_predicted_angles[i]
                if value is not None
            ),
            None,
        )

        if first_valid is not None:
            hybrid_predicted_angles[i] = [
                value - first_valid if value is not None else None
                for value in hybrid_predicted_angles[i]
            ]

    if TRIM_BY_DURATION:
        t_end = TRIM_DURATION + TRIM_MARGIN_AFTER

        keep = [
            index
            for index, time_value in enumerate(t)
            if time_value <= t_end
        ]

        t = [t[index] for index in keep]

        for dof_index in range(4):
            commanded[dof_index] = [
                commanded[dof_index][index]
                for index in keep
            ]

            controller_angles[dof_index] = [
                controller_angles[dof_index][index]
                for index in keep
            ]

        for gearbox_index in range(6):
            gearbox[gearbox_index] = [
                gearbox[gearbox_index][index]
                for index in keep
            ]

        for output_index in range(8):
            predicted_angles[output_index] = [
                predicted_angles[output_index][index]
                for index in keep
            ]

            hybrid_predicted_angles[output_index] = [
                hybrid_predicted_angles[output_index][index]
                for index in keep
            ]

    return (
        t,
        commanded,
        controller_angles,
        predicted_angles,
        hybrid_predicted_angles,
        gearbox,
        ros_t0,
    )

def load_video_angle_file(file_path, ros_t0):
    """
    Load and synchronize filtered camera angle measurements.
    """
    t_video = []
    red_shaft_angle = []
    jaw_angle = []
    secondary_shaft_angle = []

    detected_dof = None
    secondary_marker_color = None

    # Load video angle data
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if secondary_marker_color is None:
                secondary_marker_color = data.get(
                    "secondary_marker_color"
                )

            if detected_dof is None:
                detected_dof = data.get("active_dof")

            ros_time = data.get("ros_time_s")

            # Filtered DOF2 video reference
            red_angle = data.get(
                "measured_angle_red_shaft_zeroed_filtered"
            )

            secondary_angle = None

            if secondary_marker_color is not None:
                secondary_angle = data.get(
                    f"measured_angle_{secondary_marker_color}_shaft_zeroed_filtered"
                )

            # Filtered DOF4 validation reference
            measured_jaw_angle = data.get(
                "measured_angle_between_jaws_filtered"
            )

            if ros_time is None:
                continue

            t_rel = ros_time - ros_t0

            if t_rel < -1.0 or t_rel > TRIM_DURATION + 1.0:
                continue

            t_video.append(t_rel)
            red_shaft_angle.append(red_angle)
            jaw_angle.append(measured_jaw_angle)
            secondary_shaft_angle.append(secondary_angle)

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
        secondary_shaft_angle,
        secondary_marker_color,
    )

# -------------------------------------------------------------------------
# Sequence timing
# -------------------------------------------------------------------------
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
    Determine the primary evaluation window for an instrument sequence.

    The window starts at the first actual deviation from the neutral command value
    and spans the complete configured sequence, including waveform stages and
    inter-stage, inter-frequency, inter-range, and final pauses.
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

    # Include two samples before the detected motion onset.
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

def get_metric_zone_masks(
    sample_t,
    command_t,
    command_values,
    evaluation_window,
):
    """Classify samples as full, dynamic, or stationary."""
    sample_t = np.asarray(sample_t, dtype=float)
    command_t = np.asarray(command_t, dtype=float)
    command_values = clean_numeric_array(command_values)

    full_mask = np.isfinite(sample_t)

    if evaluation_window is not None:
        full_mask &= (
            (sample_t >= evaluation_window["start_s"])
            & (sample_t <= evaluation_window["end_s"])
        )

    valid_command = (
        np.isfinite(command_t)
        & np.isfinite(command_values)
    )
    command_t = command_t[valid_command]
    command_values = command_values[valid_command]

    dynamic_mask = np.zeros(len(sample_t), dtype=bool)

    if len(command_t) >= 2:
        change_indices = np.where(
            np.abs(np.diff(command_values))
            > COMMAND_CHANGE_THRESHOLD
        )[0] + 1

        change_times = command_t[change_indices]

        if len(change_times) > 0:
            previous_change_index = (
                np.searchsorted(
                    change_times,
                    sample_t,
                    side="right",
                )
                - 1
            )

            has_previous_change = previous_change_index >= 0

            time_since_change = np.full(
                len(sample_t),
                np.inf,
                dtype=float,
            )

            time_since_change[has_previous_change] = (
                sample_t[has_previous_change]
                - change_times[
                    previous_change_index[has_previous_change]
                ]
            )

            dynamic_mask = (
                has_previous_change
                & (time_since_change >= 0.0)
                & (time_since_change <= MOTION_MEMORY_S)
            )

    dynamic_mask &= full_mask
    stationary_mask = full_mask & ~dynamic_mask

    return {
        "full": full_mask,
        "dynamic": dynamic_mask,
        "stationary": stationary_mask,
    }

# -------------------------------------------------------------------------
# Prediction metrics
# -------------------------------------------------------------------------
def compute_prediction_metrics(
    video_t,
    video_angle_deg,
    prediction_t,
    prediction_deg,
    evaluation_window=None,
    video_sample_mask=None,
):
    video_t = np.asarray(video_t, dtype=float)
    if video_sample_mask is None:
        video_sample_mask = np.ones(len(video_t), dtype=bool)
    else:
        video_sample_mask = np.asarray(
            video_sample_mask,
            dtype=bool,
        )

        if len(video_sample_mask) != len(video_t):
            raise ValueError(
                "video_sample_mask must match video_t length."
            )

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
        & video_sample_mask
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
        for title, metrics_by_zone in prediction_blocks:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")

            for zone_name in ("full", "dynamic", "stationary"):
                write_block(
                    f,
                    f"Metric zone: {zone_name}",
                    metrics_by_zone.get(zone_name),
                )

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

# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------
def plot_active_command_axis(
    ax,
    t,
    commanded,
    controller_angles,
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

    controller_angle = clean_numeric_array(
        controller_angles[active_dof]
    )

    if np.isfinite(controller_angle).any():
        ax.plot(
            t,
            controller_angle,
            linestyle="-",
            label=f"controller-reported {active_dof_name}",
            linewidth=1.2,
        )

    ax.plot(
        t,
        clean_numeric_array(commanded[active_dof]),
        label=f"commanded {active_dof_name}",
        linewidth=1.5,
    )

    ax.set_title("Instrument command")
    ax.set_ylabel("Angle [rad]")
    ax.grid(True)
    ax.legend(fontsize=8)

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
            color="#E76F8A",
            label="video measured jaw angle [deg]",
        )
        ax.set_ylabel("Jaw angle [deg]")
    else:
        ax.plot(
            video_t,
            red_shaft_video_angle,
            linestyle="-",
            linewidth=2,
            color="#E76F8A",
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
                "color": "#9C89B8",
                "linestyle": "--",
            }
        ]

        if np.isfinite(hybrid_bend_deg).any():
            prediction_blocks.append(
                {
                    "name": "Hybrid DT bend",
                    "values_deg": hybrid_bend_deg,
                    "color": "#8064A2",
                    "linestyle": "--",
                }
            )

        return {
            "video_t": video_t,
            "video_values_deg": red_shaft_video_angle,
            "video_label": "median-filtered video red marker vs shaft [deg]",
            "video_signal_name": "median-filtered red marker vs shaft angle [deg]",
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
    commanded,
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
    metric_zone_masks = get_metric_zone_masks(
        sample_t=setup["video_t"],
        command_t=t,
        command_values=commanded[active_dof],
        evaluation_window=evaluation_window,
    )

    for block in setup["prediction_blocks"]:
        prediction_values_deg = block["values_deg"]

        metrics_by_zone = {}

        for zone_name, zone_mask in metric_zone_masks.items():
            metrics_by_zone[zone_name] = compute_prediction_metrics(
                video_t=setup["video_t"],
                video_angle_deg=setup["video_values_deg"],
                prediction_t=t,
                prediction_deg=prediction_values_deg,
                evaluation_window=evaluation_window,
                video_sample_mask=zone_mask,
            )

        prediction_metric_blocks.append(
            (
                block["name"],
                metrics_by_zone,
            )
        )

        ax.plot(
            t,
            prediction_values_deg,
            label=f"{block['name']} [deg]",
            linewidth=1.5,
            color=block["color"],
            linestyle=block["linestyle"],
            zorder=4 if block["name"] == "Hybrid DT bend" else 3
        )

    # Show metrics box for the most relevant prediction.
    # Prefer Hybrid DT when available, otherwise use the last/only block.
    shown_metrics_name, shown_metrics_by_zone = prediction_metric_blocks[-1]

    shown_metrics = (
        shown_metrics_by_zone.get("dynamic")
        or shown_metrics_by_zone.get("full")
    )
    add_metrics_text_box(
        ax,
        shown_metrics_name,
        shown_metrics,
    )

    if len(setup["video_t"]) > 0:
        ax.plot(
            setup["video_t"],
            setup["video_values_deg"],
            linestyle="-",
            linewidth=2,
            color="#E76F8A",
            zorder=2,
            label=setup["video_label"],
        )

    if len(setup["video_t"]) > 0:
        write_video_prediction_metrics(
            plot_dir=plot_dir,
            active_dof_name=active_dof_name,
            video_signal_name=setup["video_signal_name"],
            prediction_blocks=prediction_metric_blocks,
        )

    if len(setup["video_t"]) == 0:
        ax.set_title("Instrument DT prediction")
    else:
        ax.set_title("Instrument DT prediction vs video measurement")

    ax.set_ylabel("Angle [deg]")
    ax.set_xlabel("Time [s]")
    ax.grid(True)
    ax.legend(fontsize=8)

# -------------------------------------------------------------------------
# General plots
# -------------------------------------------------------------------------
def plot_gearbox_state(plot_dir, t, gearbox):
    """
    Plot all gearbox DT output states for the trial.
    """
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
    """
    Plot camera-vs-DT position predictions and their residuals.
    """
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
            linewidth=1.5,
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

# -------------------------------------------------------------------------
# DOF-specific plots
# -------------------------------------------------------------------------
def plot_dof2_video_validation(
    plot_dir,
    t,
    commanded,
    video_t,
    video_angle,
    predicted_angles,
    hybrid_predicted_angles,
):
    """
    Compare the DOF2 command and Instrument DT predictions with the camera measurement.
    """
    command_deg = radians_to_deg_array(commanded[1])
    physics_bend_deg = radians_to_deg_array(predicted_angles[1])
    hybrid_bend_deg = radians_to_deg_array(
        hybrid_predicted_angles[1]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    fig.suptitle(
        "DOF2 video validation",
        fontsize=16,
    )

    # Command versus physical camera measurement
    axes[0].plot(
        t,
        command_deg,
        label="commanded DOF2",
    )
    axes[0].plot(
        video_t,
        video_angle,
        linewidth=2,
        label="video measurement",
    )
    axes[0].set_ylabel("Angle [deg]")
    axes[0].set_title("Command vs measured instrument output")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # Digital Twin predictions versus camera measurement
    axes[1].plot(
        video_t,
        video_angle,
        linewidth=2,
        label="video measurement",
    )

    if np.isfinite(physics_bend_deg).any():
        axes[1].plot(
            t,
            physics_bend_deg,
            linestyle="--",
            label="Physics-based Instrument DT",
        )

    if np.isfinite(hybrid_bend_deg).any():
        axes[1].plot(
            t,
            hybrid_bend_deg,
            linestyle="--",
            label="Hybrid Instrument DT",
        )

    axes[1].set_ylabel("Angle [deg]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Instrument DT prediction vs video measurement")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(
        os.path.join(
            plot_dir,
            "dof2_video_validation.png",
        ),
        dpi=200,
    )
    plt.close(fig)

def plot_gripper_video_measurements(
    plot_dir,
    video_t,
    jaw_video_angle,
    secondary_shaft_video_angle,
    red_shaft_video_angle,
    secondary_marker_color,
):
    """
    Plot the filtered camera measurements used for DOF4 validation.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Gripper video measurements", fontsize=16)

    axes[0].plot(
        video_t,
        jaw_video_angle,
        linestyle="-",
        linewidth=2,
        label=(
            f"measured jaw angle: "
            f"{secondary_marker_color} vs red [deg]"
        ),
    )
    axes[0].set_ylabel("Jaw angle [deg]")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        video_t,
        secondary_shaft_video_angle, 
        linestyle="-",
        linewidth=2,
        label=(
            f"measured {secondary_marker_color} marker vs shaft [deg]"
        ),
    )
    axes[1].set_ylabel(secondary_marker_color.capitalize() + "-shaft [deg]")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        video_t,
        red_shaft_video_angle,
        linestyle="-",
        linewidth=2,
        label=(
            f"measured red marker vs shaft [deg]"
        ),
    )
    axes[2].set_ylabel("Red-shaft [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(plot_dir, "gripper_video_measurements.png"), dpi=200)
    plt.close(fig)

# -------------------------------------------------------------------------
# Overview plots
# -------------------------------------------------------------------------
def plot_overview_inputs_outputs(
    plot_dir,
    pattern_name,
    active_dof_name,
    t,
    commanded,
    controller_angles,
    coupling_mode,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    active_dof,
    params_file=None,
):
    """
    Plot the instrument command and available measured instrument outputs.

    The overview shows:
    - commanded and controller-reported instrument angles;
    - camera-measured instrument output when available.
    """
    show_instrument = coupling_mode == "full_setup"
    nrows = 2 if show_instrument else 1

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(14, 4 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    fig.suptitle(
        f"Overview inputs and measured outputs: {active_dof_name}",
        fontsize=16,
    )

    plot_active_command_axis(
        axes[0],
        t,
        commanded,
        controller_angles,
        active_dof,
        active_dof_name,
    )

    if show_instrument:
        plot_video_instrument_measurement_axis(
            axes[1],
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
    controller_angles,
    gearbox,
    predicted_angles,
    hybrid_predicted_angles,
    coupling_mode,
    video_t,
    red_shaft_video_angle,
    jaw_video_angle,
    active_dof,
    params_file=None,
    evaluation_window=None
):
    """
    Plot the Gearbox and Instrument Digital Twin prediction overview.

    The overview shows:
    - commanded and controller-reported instrument motion;
    - relevant Gearbox DT states;
    - Instrument DT and Hybrid DT predictions;
    - camera measurements when available.
    """
    show_instrument = coupling_mode == "full_setup"
    nrows = 3 if show_instrument else 2

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
        controller_angles,
        active_dof,
        active_dof_name,
    )

    plot_relevant_gearbox_axis(
        axes[1],
        t,
        gearbox,
        active_dof,
    )

    if show_instrument:
        plot_instrument_prediction_axis(
            axes[2],
            plot_dir,
            t,
            commanded,
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
    
# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------
def plot_continuous_file(
    file_path,
    video_angle_file=None,
    output_dir=None,
    params_file=None,
    coupling_mode="full_setup",
    dof=None,
):
    """
    Generate Gearbox and Instrument DT plots for one recorded DOF trial.

    Full-setup trials additionally include camera validation and instrument
    prediction residuals when video measurements are available.
    """
    t, commanded, controller_angles, predicted_angles, hybrid_predicted_angles, gearbox, ros_t0 = load_continuous_file(file_path)
    basename = os.path.basename(file_path).replace(".jsonl", "")
    pattern_name = os.path.basename(os.path.dirname(file_path))

    if output_dir is None:
        plot_dir = os.path.join(os.path.dirname(file_path), "plots", basename)
    else:
        plot_dir = output_dir

    os.makedirs(plot_dir, exist_ok=True)

    active_dof = (
        dof - 1
        if dof is not None
        else detect_active_dof(commanded)
    )
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
            secondary_shaft_video_angle,
            secondary_marker_color,
        ) = load_video_angle_file(
            video_angle_file,
            ros_t0,
        )
    else:
        video_t = []
        red_shaft_video_angle = []
        jaw_video_angle = []
        secondary_shaft_video_angle = []
        secondary_marker_color = None

    plot_gearbox_state(
        plot_dir,
        t,
        gearbox,
    )

    if coupling_mode == "full_setup":
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
                secondary_shaft_video_angle, 
                red_shaft_video_angle,
                secondary_marker_color
            )
    

    # Overview 1: measured inputs and outputs only
    plot_overview_inputs_outputs(
        plot_dir=plot_dir,
        pattern_name=pattern_name,
        active_dof_name=active_dof_name,
        t=t,
        commanded=commanded,
        controller_angles=controller_angles,
        coupling_mode=coupling_mode,
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
        controller_angles=controller_angles,
        coupling_mode=coupling_mode,
        gearbox=gearbox,
        predicted_angles=predicted_angles,
        hybrid_predicted_angles=hybrid_predicted_angles,
        video_t=video_t,
        red_shaft_video_angle=red_shaft_video_angle,
        jaw_video_angle=jaw_video_angle,
        active_dof=active_dof,
        params_file=params_file,
        evaluation_window=evaluation_window
    )


    print(f"Saved plots in: {plot_dir}")

def main():
    """
    Parse command-line arguments and generate the DT plots for one trial.
    """
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
        default=str(DEFAULT_DOF_PATTERN_CONFIG_PATH),
    )

    parser.add_argument(
        "--dof",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
    )

    args = parser.parse_args()

    plot_continuous_file(
        file_path=args.file,
        video_angle_file=args.video_angles,
        output_dir=args.output_dir,
        params_file=args.params_file,
        coupling_mode=args.coupling_mode,
        dof=args.dof,
    )

    print("All plots saved.")


if __name__ == "__main__":
    main()