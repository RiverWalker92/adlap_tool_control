#!/usr/bin/env python3
"""Offset-insensitive Instrument DT output diagnostics for DOF2 bending.

The detector compares the Hybrid Instrument DT bending prediction with the
filtered video measurement. It is designed to distinguish:

1. a constant prediction/video offset, which is NOT a fault trigger;
2. reduced bending in both directions;
3. repeatable asymmetric bending loss, where one bending direction is
   substantially weaker than the other.

The main diagnostic features are direction-specific bending gains. The gains
are calculated around a local prediction centre, after removing a robust
constant video/prediction offset. Therefore a fixed initialisation offset does
not directly trigger the detector.

Expected input fields
---------------------
ROS log:
    time
    ros_timestamp
    sequence_condition
    predicted_instrument_angles
    hybrid_predicted_instrument_angles

Video angle JSONL:
    active_dof
    ros_time_s
    measured_angle_red_shaft_zeroed_filtered

For DOF2, Instrument DT output index 1 is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DETECTOR_NAME = "instrument_dof2_bending_asymmetry"
DETECTOR_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Development thresholds
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# These values are intended for development/testing only.
# For the final diagnostic framework, replace them with limits obtained from
# repeated HEALTHY full-setup DOF2 trials.
#
# Small prediction amplitudes are excluded because backlash/deadband and video
# noise have a disproportionately large influence on a gain calculation.
MIN_PREDICTED_HALF_RANGE_DEG = 8.0

# Fractions of the prediction half-range used to define diagnostic zones.
CENTER_ZONE_FRACTION = 0.15
EXTREME_ZONE_FRACTION = 0.70

# One-sided impairment requires BOTH:
# - a sufficiently large difference between positive and negative gain;
# - a clearly reduced gain on the weaker side.
GAIN_ASYMMETRY_LIMIT = 0.30
WEAK_SIDE_GAIN_LIMIT = 0.65

# The weak-side loss must repeat in at least this fraction of its extrema.
MIN_WEAK_SIDE_BAD_EVENT_FRACTION = 2.0 / 3.0

# If at least three conditions are evaluable, require two conditions to agree.
MIN_DEVIATING_CONDITIONS_IF_MULTIPLE = 2


# ---------------------------------------------------------------------------
# Loading and synchronisation
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc

    if not rows:
        raise ValueError(f"No rows found in {path}.")

    return rows


def load_dof2_comparison(
    ros_log_path: Path,
    video_angles_path: Path,
    prediction_source: str = "hybrid",
) -> dict[str, np.ndarray]:
    """Load and align DOF2 prediction, video measurement, and condition labels.

    The Instrument DT predictions are zeroed in the same way as the plotting
    workflow: the first finite prediction is subtracted from the complete
    prediction signal.

    Video samples are placed on the ROS-log time axis using ros_time_s.
    Prediction values are linearly interpolated to video timestamps.
    sequence_condition is assigned from the nearest ROS sample.
    """

    if prediction_source not in {"hybrid", "physics"}:
        raise ValueError(
            "prediction_source must be 'hybrid' or 'physics'."
        )

    prediction_key = (
        "hybrid_predicted_instrument_angles"
        if prediction_source == "hybrid"
        else "predicted_instrument_angles"
    )

    ros_rows = _load_jsonl(Path(ros_log_path))

    ros_time = []
    ros_timestamp = []
    prediction_rad = []
    conditions = []

    for row in ros_rows:
        prediction = row.get(prediction_key)

        if prediction is None or len(prediction) <= 1:
            continue

        value = prediction[1]

        if value is None:
            continue

        try:
            value = float(value)
            time_value = float(row["time"])
            ros_stamp = float(row["ros_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (
            np.isfinite(value)
            and np.isfinite(time_value)
            and np.isfinite(ros_stamp)
        ):
            continue

        ros_time.append(time_value)
        ros_timestamp.append(ros_stamp)
        prediction_rad.append(value)

        condition = row.get("sequence_condition")
        if condition is None:
            conditions.append("")
        else:
            conditions.append(str(condition).strip().lower())

    if len(ros_time) < 2:
        raise ValueError(
            f"No usable {prediction_source} Instrument DT prediction found."
        )

    ros_time = np.asarray(ros_time, dtype=float)
    ros_timestamp = np.asarray(ros_timestamp, dtype=float)
    prediction_deg = np.rad2deg(
        np.asarray(prediction_rad, dtype=float)
    )
    conditions = np.asarray(conditions, dtype=object)

    # Same relative time / zeroing principle as the Instrument DT plotter.
    motor_ros_t0 = float(ros_timestamp[0])
    ros_time = ros_time - ros_time[0]
    prediction_deg = prediction_deg - prediction_deg[0]

    video_rows = _load_jsonl(Path(video_angles_path))

    video_time = []
    measured_deg = []

    for row in video_rows:
        active_dof = row.get("active_dof")
        if active_dof is not None and int(active_dof) != 2:
            continue

        value = row.get(
            "measured_angle_red_shaft_zeroed_filtered"
        )
        ros_stamp = row.get("ros_time_s")

        if value is None or ros_stamp is None:
            continue

        try:
            value = float(value)
            ros_stamp = float(ros_stamp)
        except (TypeError, ValueError):
            continue

        if not (
            np.isfinite(value)
            and np.isfinite(ros_stamp)
        ):
            continue

        video_time.append(
            ros_stamp - motor_ros_t0
        )
        measured_deg.append(value)

    if len(video_time) < 2:
        raise ValueError(
            "No usable filtered DOF2 video measurements found."
        )

    video_time = np.asarray(video_time, dtype=float)
    measured_deg = np.asarray(measured_deg, dtype=float)

    overlap = (
        (video_time >= ros_time[0])
        & (video_time <= ros_time[-1])
    )

    video_time = video_time[overlap]
    measured_deg = measured_deg[overlap]

    if len(video_time) < 2:
        raise ValueError(
            "No time overlap between Instrument DT prediction and video."
        )

    predicted_deg = np.interp(
        video_time,
        ros_time,
        prediction_deg,
    )

    # Assign sequence condition using nearest ROS sample.
    right = np.searchsorted(
        ros_time,
        video_time,
        side="left",
    )
    right = np.clip(
        right,
        1,
        len(ros_time) - 1,
    )
    left = right - 1

    choose_left = (
        np.abs(video_time - ros_time[left])
        <= np.abs(ros_time[right] - video_time)
    )

    nearest = np.where(
        choose_left,
        left,
        right,
    )

    video_conditions = conditions[nearest]

    return {
        "time_s": video_time,
        "measured_deg": measured_deg,
        "predicted_deg": predicted_deg,
        "condition": video_conditions,
    }


# ---------------------------------------------------------------------------
# Directional-event analysis
# ---------------------------------------------------------------------------

def _contiguous_events(
    mask: np.ndarray,
    time_s: np.ndarray,
    minimum_samples: int = 2,
) -> list[np.ndarray]:
    """Return index arrays for contiguous True regions.

    A large time gap also starts a new event.
    """

    mask = np.asarray(mask, dtype=bool)
    time_s = np.asarray(time_s, dtype=float)

    if len(mask) == 0:
        return []

    if len(time_s) > 1:
        dt = np.diff(time_s)
        finite_dt = dt[
            np.isfinite(dt) & (dt > 0)
        ]
        median_dt = (
            float(np.median(finite_dt))
            if len(finite_dt)
            else 0.0
        )
    else:
        median_dt = 0.0

    gap_limit = (
        3.0 * median_dt
        if median_dt > 0
        else math.inf
    )

    events: list[np.ndarray] = []
    start: int | None = None

    for index, active in enumerate(mask):
        gap_before = (
            index > 0
            and (
                time_s[index] - time_s[index - 1]
            ) > gap_limit
        )

        if start is not None and (
            not active or gap_before
        ):
            event = np.arange(
                start,
                index,
                dtype=int,
            )

            if len(event) >= minimum_samples:
                events.append(event)

            start = None

        if active and start is None:
            start = index

    if start is not None:
        event = np.arange(
            start,
            len(mask),
            dtype=int,
        )

        if len(event) >= minimum_samples:
            events.append(event)

    return events


def _analyse_direction_events(
    events: list[np.ndarray],
    time_s: np.ndarray,
    predicted_deg: np.ndarray,
    measured_offset_corrected_deg: np.ndarray,
    residual_offset_corrected_deg: np.ndarray,
    prediction_centre_deg: float,
) -> list[dict[str, float]]:
    """Calculate offset-insensitive gain for every directional extremum."""

    results = []

    for event_index, indices in enumerate(events):
        predicted_excursion = float(
            np.median(
                predicted_deg[indices]
                - prediction_centre_deg
            )
        )

        measured_excursion = float(
            np.median(
                measured_offset_corrected_deg[indices]
                - prediction_centre_deg
            )
        )

        if abs(predicted_excursion) < 1e-9:
            continue

        gain = (
            measured_excursion
            / predicted_excursion
        )

        results.append({
            "event_index": int(event_index),
            "start_s": float(time_s[indices[0]]),
            "end_s": float(time_s[indices[-1]]),
            "predicted_excursion_deg":
                predicted_excursion,
            "measured_excursion_deg":
                measured_excursion,
            "gain":
                float(gain),
            "median_centered_residual_deg":
                float(
                    np.median(
                        residual_offset_corrected_deg[
                            indices
                        ]
                    )
                ),
        })

    return results


def analyse_condition(
    time_s: np.ndarray,
    measured_deg: np.ndarray,
    predicted_deg: np.ndarray,
    condition: str,
) -> dict[str, Any]:
    """Analyse one waveform/frequency/range condition."""

    valid = (
        np.isfinite(time_s)
        & np.isfinite(measured_deg)
        & np.isfinite(predicted_deg)
    )

    time_s = time_s[valid]
    measured_deg = measured_deg[valid]
    predicted_deg = predicted_deg[valid]

    if len(time_s) < 10:
        return {
            "condition": condition,
            "status": "not_evaluated",
            "reason": "too_few_samples",
        }

    # Robustly estimate the prediction centre and amplitude.
    lower = float(
        np.percentile(
            predicted_deg,
            5.0,
        )
    )
    upper = float(
        np.percentile(
            predicted_deg,
            95.0,
        )
    )

    prediction_centre = (
        0.5 * (lower + upper)
    )
    prediction_half_range = (
        0.5 * (upper - lower)
    )

    if (
        prediction_half_range
        < MIN_PREDICTED_HALF_RANGE_DEG
    ):
        return {
            "condition": condition,
            "status": "not_evaluated",
            "reason":
                "prediction_amplitude_too_small",
            "prediction_half_range_deg":
                float(prediction_half_range),
            "minimum_required_half_range_deg":
                MIN_PREDICTED_HALF_RANGE_DEG,
        }

    residual = (
        measured_deg
        - predicted_deg
    )

    # ---------------------------------------------------------------
    # Estimate constant offset from centre crossings.
    #
    # The offset is REPORTED and REMOVED, but is not a fault trigger.
    # ---------------------------------------------------------------

    centre_zone = (
        np.abs(
            predicted_deg
            - prediction_centre
        )
        <= (
            CENTER_ZONE_FRACTION
            * prediction_half_range
        )
    )

    if np.sum(centre_zone) >= 3:
        constant_offset_deg = float(
            np.median(
                residual[centre_zone]
            )
        )
    else:
        constant_offset_deg = float(
            np.median(residual)
        )

    measured_corrected = (
        measured_deg
        - constant_offset_deg
    )

    centered_residual = (
        measured_corrected
        - predicted_deg
    )

    # ---------------------------------------------------------------
    # Define positive and negative extreme zones from the PREDICTION.
    #
    # The measured signal is deliberately not used for segmentation:
    # a faulty instrument may fail to reach the expected extremum.
    # ---------------------------------------------------------------

    positive_mask = (
        predicted_deg
        >= (
            prediction_centre
            + EXTREME_ZONE_FRACTION
            * prediction_half_range
        )
    )

    negative_mask = (
        predicted_deg
        <= (
            prediction_centre
            - EXTREME_ZONE_FRACTION
            * prediction_half_range
        )
    )

    positive_events = _analyse_direction_events(
        events=_contiguous_events(
            positive_mask,
            time_s,
        ),
        time_s=time_s,
        predicted_deg=predicted_deg,
        measured_offset_corrected_deg=
            measured_corrected,
        residual_offset_corrected_deg=
            centered_residual,
        prediction_centre_deg=
            prediction_centre,
    )

    negative_events = _analyse_direction_events(
        events=_contiguous_events(
            negative_mask,
            time_s,
        ),
        time_s=time_s,
        predicted_deg=predicted_deg,
        measured_offset_corrected_deg=
            measured_corrected,
        residual_offset_corrected_deg=
            centered_residual,
        prediction_centre_deg=
            prediction_centre,
    )

    if (
        not positive_events
        or not negative_events
    ):
        return {
            "condition": condition,
            "status": "not_evaluated",
            "reason":
                "missing_directional_extrema",
            "prediction_half_range_deg":
                float(prediction_half_range),
            "constant_offset_deg":
                constant_offset_deg,
        }

    positive_gains = np.asarray(
        [
            event["gain"]
            for event in positive_events
        ],
        dtype=float,
    )

    negative_gains = np.asarray(
        [
            event["gain"]
            for event in negative_events
        ],
        dtype=float,
    )

    positive_gain = float(
        np.median(positive_gains)
    )
    negative_gain = float(
        np.median(negative_gains)
    )

    gain_asymmetry = float(
        abs(
            positive_gain
            - negative_gain
        )
    )

    if positive_gain <= negative_gain:
        weak_side = "positive"
        weak_gain = positive_gain
        strong_gain = negative_gain
        weak_event_gains = positive_gains
    else:
        weak_side = "negative"
        weak_gain = negative_gain
        strong_gain = positive_gain
        weak_event_gains = negative_gains

    weak_side_bad_event_fraction = float(
        np.mean(
            weak_event_gains
            < WEAK_SIDE_GAIN_LIMIT
        )
    )

    asymmetry_triggered = (
        gain_asymmetry
        > GAIN_ASYMMETRY_LIMIT
    )

    weak_side_triggered = (
        weak_gain
        < WEAK_SIDE_GAIN_LIMIT
    )

    repeatability_triggered = (
        weak_side_bad_event_fraction
        >= MIN_WEAK_SIDE_BAD_EVENT_FRACTION
    )

    deviating = (
        asymmetry_triggered
        and weak_side_triggered
        and repeatability_triggered
    )

    mean_directional_gain = float(
        0.5
        * (
            positive_gain
            + negative_gain
        )
    )

    return {
        "condition": condition,
        "status":
            "deviating"
            if deviating
            else "healthy",

        "fault_signature":
            (
                "asymmetric_bending_impairment"
                if deviating
                else "none"
            ),

        # Constant bias is not itself a trigger.
        "constant_offset_deg":
            constant_offset_deg,

        "offset_is_fault_trigger":
            False,

        "prediction_centre_deg":
            float(prediction_centre),

        "prediction_half_range_deg":
            float(prediction_half_range),

        "positive_direction": {
            "median_gain":
                positive_gain,
            "event_count":
                int(len(positive_events)),
            "events":
                positive_events,
        },

        "negative_direction": {
            "median_gain":
                negative_gain,
            "event_count":
                int(len(negative_events)),
            "events":
                negative_events,
        },

        "mean_directional_gain":
            mean_directional_gain,

        "gain_asymmetry":
            gain_asymmetry,

        "weak_side":
            weak_side,

        "weak_side_gain":
            float(weak_gain),

        "strong_side_gain":
            float(strong_gain),

        "weak_side_bad_event_fraction":
            weak_side_bad_event_fraction,

        "thresholds": {
            "gain_asymmetry_limit":
                GAIN_ASYMMETRY_LIMIT,
            "weak_side_gain_limit":
                WEAK_SIDE_GAIN_LIMIT,
            "minimum_bad_event_fraction":
                MIN_WEAK_SIDE_BAD_EVENT_FRACTION,
        },

        "triggers": {
            "gain_asymmetry":
                bool(asymmetry_triggered),
            "weak_side_gain":
                bool(weak_side_triggered),
            "repeatability":
                bool(repeatability_triggered),
        },
    }


# ---------------------------------------------------------------------------
# Complete DOF2 diagnosis
# ---------------------------------------------------------------------------

def diagnose_dof2_bending(
    ros_log_path: Path,
    video_angles_path: Path,
    prediction_source: str = "hybrid",
) -> dict[str, Any]:
    """Diagnose repeatable asymmetric DOF2 bending behaviour."""

    comparison = load_dof2_comparison(
        ros_log_path=Path(ros_log_path),
        video_angles_path=Path(video_angles_path),
        prediction_source=prediction_source,
    )

    condition_values = sorted({
        str(value)
        for value in comparison["condition"]
        if str(value)
        not in {
            "",
            "none",
            "pause",
        }
    })

    condition_results = []

    for condition in condition_values:
        mask = (
            comparison["condition"]
            == condition
        )

        result = analyse_condition(
            time_s=comparison["time_s"][mask],
            measured_deg=
                comparison["measured_deg"][mask],
            predicted_deg=
                comparison["predicted_deg"][mask],
            condition=condition,
        )

        condition_results.append(result)

    evaluated = [
        result
        for result in condition_results
        if result["status"]
        in {
            "healthy",
            "deviating",
        }
    ]

    deviating = [
        result
        for result in evaluated
        if result["status"] == "deviating"
    ]

    if not evaluated:
        return {
            "detector": DETECTOR_NAME,
            "detector_version": DETECTOR_VERSION,
            "dof": 2,
            "status": "not_evaluated",
            "reason":
                "no_evaluable_conditions",
            "prediction_source":
                prediction_source,
            "condition_results":
                condition_results,
        }

    required_deviating_conditions = (
        1
        if len(evaluated) < 3
        else MIN_DEVIATING_CONDITIONS_IF_MULTIPLE
    )

    trial_deviating = (
        len(deviating)
        >= required_deviating_conditions
    )

    strongest = max(
        evaluated,
        key=lambda result:
            float(
                result.get(
                    "gain_asymmetry",
                    -math.inf,
                )
            ),
    )

    return {
        "detector":
            DETECTOR_NAME,

        "detector_version":
            DETECTOR_VERSION,

        "dof":
            2,

        "status":
            "deviating"
            if trial_deviating
            else "healthy",

        "fault_signature":
            (
                "asymmetric_bending_impairment"
                if trial_deviating
                else "none"
            ),

        "prediction_source":
            prediction_source,

        "offset_handling":
            (
                "Constant video/prediction offset is estimated "
                "from centre crossings and removed before "
                "directional gain calculation. Offset magnitude "
                "is not a fault trigger."
            ),

        "evaluated_conditions":
            int(len(evaluated)),

        "deviating_conditions":
            int(len(deviating)),

        "required_deviating_conditions":
            int(required_deviating_conditions),

        "deviating_condition_names": [
            result["condition"]
            for result in deviating
        ],

        "strongest_asymmetry": {
            "condition":
                strongest["condition"],
            "gain_asymmetry":
                strongest.get(
                    "gain_asymmetry"
                ),
            "weak_side":
                strongest.get(
                    "weak_side"
                ),
            "weak_side_gain":
                strongest.get(
                    "weak_side_gain"
                ),
            "strong_side_gain":
                strongest.get(
                    "strong_side_gain"
                ),
            "constant_offset_deg":
                strongest.get(
                    "constant_offset_deg"
                ),
        },

        "condition_results":
            condition_results,

        "threshold_note": (
            "Current thresholds are development values. "
            "Replace them with limits derived from repeated "
            "healthy full-setup DOF2 trials before final validation."
        ),
    }


def diagnose_dof2_from_metadata(
    metadata_path: Path,
    prediction_source: str = "hybrid",
) -> dict[str, Any]:
    """Convenience wrapper for the existing automated-trial metadata."""

    metadata_path = Path(
        metadata_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    if str(
        metadata.get(
            "coupling_mode",
            "",
        )
    ) != "full_setup":
        return {
            "detector": DETECTOR_NAME,
            "detector_version": DETECTOR_VERSION,
            "status": "not_evaluated",
            "reason":
                "requires_full_setup",
        }

    if int(
        metadata.get(
            "dof",
            -1,
        )
    ) != 2:
        return {
            "detector": DETECTOR_NAME,
            "detector_version": DETECTOR_VERSION,
            "status": "not_evaluated",
            "reason":
                "currently_only_dof2_supported",
        }

    ros_log_path = metadata.get(
        "ros_log_path"
    )
    video_angles_path = metadata.get(
        "angles_path"
    )

    if not ros_log_path:
        return {
            "detector": DETECTOR_NAME,
            "detector_version": DETECTOR_VERSION,
            "status": "not_evaluated",
            "reason":
                "metadata_has_no_ros_log_path",
        }

    if not video_angles_path:
        return {
            "detector": DETECTOR_NAME,
            "detector_version": DETECTOR_VERSION,
            "status": "not_evaluated",
            "reason":
                "metadata_has_no_video_angle_path",
        }

    return diagnose_dof2_bending(
        ros_log_path=Path(ros_log_path),
        video_angles_path=Path(
            video_angles_path
        ),
        prediction_source=prediction_source,
    )


# ---------------------------------------------------------------------------
# Optional diagnostic plot
# ---------------------------------------------------------------------------

def create_gain_plot(
    diagnosis: dict[str, Any],
    output_path: Path,
) -> None:
    """Create a compact condition-by-condition directional-gain plot."""

    import matplotlib.pyplot as plt

    evaluated = [
        result
        for result in diagnosis.get(
            "condition_results",
            [],
        )
        if result.get("status")
        in {
            "healthy",
            "deviating",
        }
    ]

    if not evaluated:
        return

    labels = [
        result["condition"]
        .replace("sinusoid_", "sin_")
        .replace("triangle_", "tri_")
        for result in evaluated
    ]

    positive = [
        result["positive_direction"][
            "median_gain"
        ]
        for result in evaluated
    ]

    negative = [
        result["negative_direction"][
            "median_gain"
        ]
        for result in evaluated
    ]

    x = np.arange(
        len(evaluated)
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        x,
        positive,
        marker="o",
        label="Positive-direction gain",
    )

    ax.plot(
        x,
        negative,
        marker="o",
        label="Negative-direction gain",
    )

    ax.axhline(
        WEAK_SIDE_GAIN_LIMIT,
        linestyle="--",
        linewidth=1.0,
        label="Development weak-side limit",
    )

    ax.set_ylabel(
        "Measured / predicted bending gain [-]"
    )

    ax.set_xlabel(
        "DOF2 condition"
    )

    ax.set_title(
        "Offset-insensitive DOF2 directional bending response"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--ros-log",
        help="Full-setup ROS log JSONL.",
    )

    parser.add_argument(
        "--video-angles",
        help="Filtered webcam angle JSONL.",
    )

    parser.add_argument(
        "--metadata-file",
        help=(
            "Automated-trial metadata JSON. "
            "When supplied, ros_log_path and angles_path "
            "are read from metadata."
        ),
    )

    parser.add_argument(
        "--prediction-source",
        choices=[
            "hybrid",
            "physics",
        ],
        default="hybrid",
    )

    parser.add_argument(
        "--output-json",
        help="Optional diagnostics JSON output path.",
    )

    parser.add_argument(
        "--output-plot",
        help="Optional directional-gain plot output path.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.metadata_file:
        diagnosis = diagnose_dof2_from_metadata(
            Path(args.metadata_file),
            prediction_source=
                args.prediction_source,
        )
    else:
        if (
            not args.ros_log
            or not args.video_angles
        ):
            raise SystemExit(
                "Use --metadata-file, or provide both "
                "--ros-log and --video-angles."
            )

        diagnosis = diagnose_dof2_bending(
            ros_log_path=
                Path(args.ros_log),
            video_angles_path=
                Path(args.video_angles),
            prediction_source=
                args.prediction_source,
        )

    print(
        json.dumps(
            diagnosis,
            indent=2,
            allow_nan=False,
        )
    )

    if args.output_json:
        output_json = Path(
            args.output_json
        )
        output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_json.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                diagnosis,
                handle,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")

    if args.output_plot:
        create_gain_plot(
            diagnosis,
            Path(args.output_plot),
        )


if __name__ == "__main__":
    main()
