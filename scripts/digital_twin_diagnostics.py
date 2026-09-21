#!/usr/bin/env python3
"""Digital Twin residual-band diagnostics for motor, gearbox, and full-setup tests.

Diagnosis is based on residuals between measured and Digital Twin predicted
motor position and current.

Supported coupling modes:
- motor_only
- gearbox_only
- full_setup

For DOF-controlled trials, diagnosis is evaluated per waveform condition
(sequence_condition), with separate handling of pause intervals when defined.
"""
# TODO: add motor-only m0 dictiorary in dof3
# TODO: add localisation past
# TODO: add instrument output deviation 

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import yaml

from ament_index_python.packages import (
    get_package_share_directory,
)

try:
    from adlap_tool_control.instrument_output_diagnostics import (
        create_gain_plot,
        diagnose_dof2_from_metadata,
    )
except ImportError:
    from instrument_output_diagnostics import (
        create_gain_plot,
        diagnose_dof2_from_metadata,
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DETECTOR_NAME = "digital_twin_residual_band_diagnostics"
DETECTOR_VERSION = "2.2.1"

DEFAULT_DIAGNOSTIC_CONFIG_PATH = (
    Path(
        get_package_share_directory(
            "adlap_tool_control"
        )
    )
    / "config"
    / "diagnostic_params.yaml"
)


def load_diagnostic_params(
    config_path: Path = DEFAULT_DIAGNOSTIC_CONFIG_PATH,
) -> dict[str, Any]:

    config_path = Path(
        config_path
    ).expanduser()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(
            handle
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid diagnostics config: {config_path}"
        )

    try:
        params = config[
            "digital_twin_diagnostics"
        ]
    except KeyError as exc:
        raise ValueError(
            "Missing 'digital_twin_diagnostics' "
            f"in {config_path}"
        ) from exc

    return params


DIAGNOSTIC_PARAMS = load_diagnostic_params()

GENERAL_PARAMS = DIAGNOSTIC_PARAMS[
    "general"
]

MOTOR_ONLY_PARAMS = DIAGNOSTIC_PARAMS[
    "motor_only"
]

GEARBOX_ONLY_PARAMS = DIAGNOSTIC_PARAMS[
    "gearbox_only"
]

FULL_SETUP_PARAMS = DIAGNOSTIC_PARAMS[
    "full_setup"
]

SCISSOR_DOF4_PARAMS = FULL_SETUP_PARAMS[
    "scissor_dof4"
]

# ---------------------------------------------------------------------------
# Diagnostic parameter aliases
# ---------------------------------------------------------------------------

MAX_OUTSIDE_FRACTION = float(
    GENERAL_PARAMS[
        "max_outside_fraction"
    ]
)

DOF_RELEVANT_MOTORS = (
    GENERAL_PARAMS[
        "dof_relevant_motors"
    ]
)

SETUP_FAULT_LOCATION = (
    GENERAL_PARAMS[
        "setup_fault_location"
    ]
)

MOTOR_ONLY_BASELINE_NAME = (
    GENERAL_PARAMS[
        "baseline_names"
    ]["motor_only"]
)

GEARBOX_BASELINE_NAME = (
    GENERAL_PARAMS[
        "baseline_names"
    ]["gearbox_only"]
)

FULL_SETUP_BASELINE_NAME = (
    GENERAL_PARAMS[
        "baseline_names"
    ]["full_setup"]
)


# ---------------------------------------------------------------------------
# Motor-only
# ---------------------------------------------------------------------------

MOTOR_ONLY_POSITION_BANDS = (
    MOTOR_ONLY_PARAMS[
        "position_bands"
    ]
)

MOTOR_ONLY_CURRENT_BANDS = (
    MOTOR_ONLY_PARAMS[
        "current_bands"
    ]
)

MOTOR_ONLY_DOF_POSITION_BANDS = (
    MOTOR_ONLY_PARAMS[
        "dof_position_bands"
    ]
)

MOTOR_ONLY_DOF_CURRENT_BANDS = (
    MOTOR_ONLY_PARAMS[
        "dof_current_bands"
    ]
)

MOTOR_ONLY_DOF_CURRENT_DURATION_LIMITS = (
    MOTOR_ONLY_PARAMS[
        "dof_current_duration_limits_s"
    ]
)


# ---------------------------------------------------------------------------
# Gearbox-only
# ---------------------------------------------------------------------------

GEARBOX_POSITION_BANDS = (
    GEARBOX_ONLY_PARAMS[
        "position_bands"
    ]
)

GEARBOX_CURRENT_BANDS = (
    GEARBOX_ONLY_PARAMS[
        "current_bands"
    ]
)

# Optional configuration. The one-sided upper-band fraction rule uses the
# existing healthy current bands. A condition-specific mean rule is only
# evaluated after limits have been calibrated from healthy gearbox runs.
GEARBOX_FRICTION_DOF3_PARAMS = GEARBOX_ONLY_PARAMS.get(
    "friction_dof3", {}
)


# ---------------------------------------------------------------------------
# Full setup
# ---------------------------------------------------------------------------

FULL_SETUP_DOF_POSITION_BANDS = (
    FULL_SETUP_PARAMS[
        "dof_position_bands"
    ]
)

FULL_SETUP_DOF_CURRENT_BANDS = (
    FULL_SETUP_PARAMS[
        "dof_current_bands"
    ]
)

FULL_SETUP_DOF_CURRENT_DURATION_LIMITS = (
    FULL_SETUP_PARAMS[
        "dof_current_duration_limits_s"
    ]
)

FULL_SETUP_DOF_CURRENT_PEAK_LIMITS = (
    FULL_SETUP_PARAMS[
        "dof_current_peak_limits"
    ]
)

FULL_SETUP_DOF_PAUSE_POSITION_BANDS = (
    FULL_SETUP_PARAMS[
        "pause_position_bands"
    ]
)

FULL_SETUP_DOF_PAUSE_CURRENT_BANDS = (
    FULL_SETUP_PARAMS[
        "pause_current_bands"
    ]
)


# ---------------------------------------------------------------------------
# Full-setup DOF4 scissor
# ---------------------------------------------------------------------------

SCISSOR_DOF4_CURRENT_MOTOR = int(
    SCISSOR_DOF4_PARAMS[
        "current_motor"
    ]
)

SCISSOR_CLOSING_DIRECTION_WINDOW_S = float(
    SCISSOR_DOF4_PARAMS[
        "closing_direction_window_s"
    ]
)

SCISSOR_CLOSING_MIN_TARGET_CHANGE = float(
    SCISSOR_DOF4_PARAMS[
        "closing_min_target_change_pulses"
    ]
)

FULL_SETUP_SCISSOR_DOF4_MEAN_CLOSING_RESIDUAL_LIMITS = (
    SCISSOR_DOF4_PARAMS[
        "mean_closing_residual_limits_ma"
    ]
)
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    if not rows:
        raise ValueError("Replay contains no rows.")

    return rows


def validate_replay(rows: list[dict[str, Any]]) -> None:
    required = {
        "test_type",
        "task_label",
        "coupling_mode",
        "segment_time",

        "offline_commanded_target",

        "offline_predicted_positions",
        "measured_positions",
        "encoder_deviation",

        "offline_predicted_currents",
        "filtered_measured_currents",
        "filtered_current_deviation",
    }

    missing = sorted(required - rows[0].keys())

    if missing:
        raise ValueError(
            "Replay misses required fields: "
            + ", ".join(missing)
        )

    coupling_modes = {
        str(row["coupling_mode"])
        for row in rows
    }

    supported_modes = {
        "motor_only",
        "gearbox_only",
        "full_setup",
    }

    if (
        len(coupling_modes) != 1
        or not coupling_modes.issubset(supported_modes)
    ):
        raise ValueError(
            "This detector supports coupling_mode=motor_only, "
            "gearbox_only and full_setup. "
            f"Found: {sorted(coupling_modes)}"
        )

def detect_dof(
    replay_path: Path,
    rows: list[dict[str, Any]] | None = None,
) -> int:
    """Identify DOF from replay metadata, filename, or parent directory."""

    # Resistance logs need not contain `dof3` in their filename.
    if rows:
        for row in rows:
            value = row.get("active_dof")
            try:
                dof = int(value)
            except (ValueError, TypeError):
                continue
            if dof in (1, 2, 3, 4):
                return dof

    # Fallback: inspect the full path (e.g. /dof3_resistance/...).
    import re
    candidates = {
        int(match.group(1))
        for match in re.finditer(
            r"dof([1-4])(?![0-9])", str(replay_path).lower()
        )
    }
    if len(candidates) == 1:
        return next(iter(candidates))

    raise ValueError(
        "Could not determine unique DOF from active_dof or replay path."
    )

def is_gearbox_resistance_trial(
    replay_path: Path,
    rows: list[dict[str, Any]],
    coupling_mode: str,
    dof: int | None,
) -> bool:
    """Limit the special friction rule to explicitly named DOF3 resistance runs."""
    if coupling_mode != "gearbox_only" or dof != 3 or not rows:
        return False
    identifiers = (
        str(replay_path),
        str(rows[0].get("motor_name", "")),
        str(rows[0].get("test_type", "")),
        str(rows[0].get("task_label", "")),
    )
    return any("resistance" in value.lower() for value in identifiers)


def detect_dof_condition(
    row: dict[str, Any],
) -> str:

    condition = str(
        row.get("sequence_condition", "")
    ).strip().lower()

    if not condition or condition == "none":
        raise ValueError(
            "Missing sequence_condition in replay."
        )

    if condition == "pause":
        raise ValueError(
            "Sequence pause is not a diagnostic motion condition."
        )

    return condition

def analyse_pause_rows(
    rows: list[dict[str, Any]],
    motors: list[int],
    coupling_mode: str,
    dof: int | None,
) -> dict[str, Any]:

    pause_rows = [
        row
        for row in rows
        if str(
            row.get("sequence_condition", "")
        ).strip().lower() == "pause"
    ]

    result = {
        "status": "not_evaluated",
        "samples": len(pause_rows),
        "affected_motors": [],
        "motor_results": {},
    }

    if not pause_rows:
        return result

    # At the moment healthy pause bands are only
    # defined for full_setup.
    evaluate_pause = (
        coupling_mode == "full_setup"
        and dof is not None
    )

    any_evaluated = False

    for motor in motors:

        position_values = []
        current_values = []

        for row in pause_rows:

            try:
                position = (
                    row["encoder_deviation"][motor]
                )

                current = (
                    row["filtered_current_deviation"][motor]
                )

                if (
                    position is None
                    or current is None
                ):
                    continue

                position = float(position)
                current = float(current)

                if (
                    not np.isfinite(position)
                    or not np.isfinite(current)
                ):
                    continue

                position_values.append(
                    position
                )

                current_values.append(
                    current
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

        if not position_values:
            continue

        position_values = np.asarray(
            position_values,
            dtype=float,
        )

        current_values = np.asarray(
            current_values,
            dtype=float,
        )

        # -----------------------------------------------------------
        # If no pause baseline exists for this configuration,
        # keep descriptive statistics only.
        # -----------------------------------------------------------

        if not evaluate_pause:

            result["motor_results"][
                f"motor_{motor}"
            ] = {
                "status": "not_evaluated",

                "position": {
                    "mean_residual":
                        float(np.mean(position_values)),
                    "minimum_residual":
                        float(np.min(position_values)),
                    "maximum_residual":
                        float(np.max(position_values)),
                },

                "current": {
                    "mean_residual":
                        float(np.mean(current_values)),
                    "minimum_residual":
                        float(np.min(current_values)),
                    "maximum_residual":
                        float(np.max(current_values)),
                },
            }

            continue

        # -----------------------------------------------------------
        # Full-setup pause healthy bands
        # -----------------------------------------------------------

        try:
            position_lower, position_upper = (
                FULL_SETUP_DOF_PAUSE_POSITION_BANDS[
                    dof
                ][motor]
            )

            current_lower, current_upper = (
                FULL_SETUP_DOF_PAUSE_CURRENT_BANDS[
                    dof
                ][motor]
            )

        except KeyError:

            result["motor_results"][
                f"motor_{motor}"
            ] = {
                "status": "not_evaluated",
            }

            continue

        any_evaluated = True

        # -----------------------------------------------------------
        # Position pause baseline
        # -----------------------------------------------------------

        position_outside = (
            (position_values < position_lower)
            |
            (position_values > position_upper)
        )

        position_outside_count = int(
            np.sum(position_outside)
        )

        position_fraction = (
            position_outside_count
            / len(position_values)
        )

        position_deviating = (
            position_fraction
            > MAX_OUTSIDE_FRACTION
        )

        # -----------------------------------------------------------
        # Current pause baseline
        # -----------------------------------------------------------

        current_outside = (
            (current_values < current_lower)
            |
            (current_values > current_upper)
        )

        current_outside_count = int(
            np.sum(current_outside)
        )

        current_fraction = (
            current_outside_count
            / len(current_values)
        )

        current_deviating = (
            current_fraction
            > MAX_OUTSIDE_FRACTION
        )

        motor_deviating = (
            position_deviating
            or current_deviating
        )

        if motor_deviating:
            result["affected_motors"].append(
                motor
            )

        result["motor_results"][
            f"motor_{motor}"
        ] = {
            "status":
                "deviating"
                if motor_deviating
                else "healthy",

            "position": {
                "status":
                    "deviating"
                    if position_deviating
                    else "healthy",

                "healthy_band": {
                    "lower": position_lower,
                    "upper": position_upper,
                    "unit": "pulses",
                },

                "samples":
                    len(position_values),

                "samples_outside_band":
                    position_outside_count,

                "fraction_outside_band":
                    position_fraction,

                "percentage_outside_band":
                    100.0 * position_fraction,

                "mean_residual":
                    float(np.mean(position_values)),

                "minimum_residual":
                    float(np.min(position_values)),

                "maximum_residual":
                    float(np.max(position_values)),

                "fraction_rule_triggered":
                    position_deviating,
            },

            "current": {
                "status":
                    "deviating"
                    if current_deviating
                    else "healthy",

                "healthy_band": {
                    "lower": current_lower,
                    "upper": current_upper,
                    "unit": "mA",
                },

                "samples":
                    len(current_values),

                "samples_outside_band":
                    current_outside_count,

                "fraction_outside_band":
                    current_fraction,

                "percentage_outside_band":
                    100.0 * current_fraction,

                "mean_residual":
                    float(np.mean(current_values)),

                "minimum_residual":
                    float(np.min(current_values)),

                "maximum_residual":
                    float(np.max(current_values)),

                "fraction_rule_triggered":
                    current_deviating,
            },
        }

    if any_evaluated:

        result["status"] = (
            "deviating"
            if result["affected_motors"]
            else "healthy"
        )

    return result

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def longest_true_duration(
    mask: np.ndarray,
    times: np.ndarray,
) -> float:
    """Return longest continuous True period in seconds."""

    if len(mask) == 0:
        return 0.0

    if len(times) > 1:
        sample_period = float(
            np.median(np.diff(times))
        )
    else:
        sample_period = 0.0

    longest = 0.0
    start_index = None

    for i, value in enumerate(mask):

        if value and start_index is None:
            start_index = i

        if not value and start_index is not None:
            end_index = i - 1

            duration = (
                float(
                    times[end_index]
                    - times[start_index]
                )
                + sample_period
            )

            longest = max(
                longest,
                duration,
            )

            start_index = None

    if start_index is not None:
        end_index = len(mask) - 1

        duration = (
            float(
                times[end_index]
                - times[start_index]
            )
            + sample_period
        )

        longest = max(
            longest,
            duration,
        )

    return longest

def detect_scissor_closing_mask(
    commanded_target: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Detect DOF4 scissor closing from the motor-3 commanded target."""

    closing = np.zeros(
        len(commanded_target),
        dtype=bool,
    )

    if len(commanded_target) < 3:
        return closing

    time_diff = np.diff(times)

    valid_dt = time_diff[
        np.isfinite(time_diff)
        & (time_diff > 0.0)
    ]

    if len(valid_dt) == 0:
        return closing

    sample_period = float(
        np.median(valid_dt)
    )

    half_window = max(
        1,
        int(
            round(
                SCISSOR_CLOSING_DIRECTION_WINDOW_S
                / (2.0 * sample_period)
            )
        ),
    )

    if 2 * half_window >= len(commanded_target):
        return closing

    target_change = (
        commanded_target[2 * half_window:]
        - commanded_target[:-2 * half_window]
    )

    closing[
        half_window:-half_window
    ] = (
        target_change
        < -SCISSOR_CLOSING_MIN_TARGET_CHANGE
    )

    return closing

def analyse_scissor_closing_current(
    residual: np.ndarray,
    times: np.ndarray,
    commanded_target: np.ndarray,
    lower: float,
    upper: float,
    max_continuous_outside_s: float | None,
    mean_residual_limit: float | None,
) -> dict[str, Any]:
    """Analyse motor-3 current during DOF4 scissor closing."""

    closing = detect_scissor_closing_mask(
        commanded_target=commanded_target,
        times=times,
    )

    closing_count = int(
        np.sum(closing)
    )

    if closing_count == 0:
        raise ValueError(
            "No DOF4 scissor closing samples detected."
        )

    closing_residual = residual[
        closing
    ]

    outside = np.zeros(
        len(residual),
        dtype=bool,
    )

    outside[
        closing
    ] = (
        (closing_residual < lower)
        | (closing_residual > upper)
    )

    outside_count = int(
        np.sum(outside)
    )

    outside_fraction = (
        outside_count
        / closing_count
    )

    longest_outside_s = longest_true_duration(
        outside,
        times,
    )
    # ---------------------------------------------------------------
    # Method 1: residual-band diagnosis during closing
    # ---------------------------------------------------------------
    # Fraction and duration rules are temporarily disabled for
    # DOF4 scissors. Keep their statistics for inspection.

    fraction_triggered = False
    duration_triggered = False

    # ---------------------------------------------------------------
    # Method 2: mean residual during closing
    # ---------------------------------------------------------------

    mean_residual = float(
        np.mean(closing_residual)
    )

    mean_rule_evaluated = (
        mean_residual_limit is not None
    )

    # One-sided detection: measured current exceeds prediction.
    mean_rule_triggered = (
        mean_rule_evaluated
        and mean_residual > mean_residual_limit
    )

    # Only the mean rule determines the scissor current diagnosis.
    deviating = mean_rule_triggered

    return {
        "status":
            "deviating"
            if deviating
            else "healthy",

        "analysis_phase":
            "closing",

        "healthy_band": {
            "lower": lower,
            "upper": upper,
            "unit": "mA",
        },

        "samples":
            closing_count,

        "samples_outside_band":
            outside_count,

        "fraction_outside_band":
            outside_fraction,

        "percentage_outside_band":
            100.0 * outside_fraction,

        "maximum_continuous_outside_s":
            max_continuous_outside_s,

        "longest_continuous_outside_s":
            longest_outside_s,

        "minimum_residual":
            float(np.min(closing_residual)),

        "maximum_residual":
            float(np.max(closing_residual)),

        "mean_residual":
            mean_residual,

        "mean_residual_limit":
            mean_residual_limit,

        "mean_rule_evaluated":
            mean_rule_evaluated,

        "mean_rule_triggered":
            mean_rule_triggered,

        "fraction_rule_triggered":
            fraction_triggered,

        "duration_rule_triggered":
            duration_triggered,

        # Keep these fields for compatibility with the existing popup.
        "maximum_absolute_residual":
            float(
                np.max(
                    np.abs(closing_residual)
                )
            ),

        "very_large_residual_limit":
            None,

        "large_transient_rule_triggered":
            False,
    }

def get_gearbox_motion_mask(
    times: np.ndarray,
    task_labels: np.ndarray,
    commanded_target: np.ndarray,
) -> np.ndarray:
    """Select commanded motor motion, keeping separate tasks independent."""
    times = np.asarray(times, dtype=float)
    task_labels = np.asarray(task_labels, dtype=str)
    commanded_target = np.asarray(commanded_target, dtype=float)
    if len(times) != len(task_labels) or len(times) != len(commanded_target):
        raise ValueError("Gearbox motion input lengths do not match.")

    min_target_change = float(
        GEARBOX_FRICTION_DOF3_PARAMS.get("min_target_change_pulses", 1.0)
    )
    motion = np.zeros(len(times), dtype=bool)
    for label in np.unique(task_labels):
        indices = np.where(task_labels == label)[0]
        indices = indices[np.argsort(times[indices])]
        if len(indices) < 2:
            continue

        a = commanded_target[indices[:-1]]
        b = commanded_target[indices[1:]]
        dt = times[indices[1:]] - times[indices[:-1]]
        valid_pair = (
            np.isfinite(a) & np.isfinite(b)
            & np.isfinite(dt) & (dt > 0)
        )
        moving_pair = valid_pair & (np.abs(b - a) > min_target_change)
        motion[indices[1:]] = moving_pair
        if moving_pair[0]:
            motion[indices[0]] = True

    return motion & np.isfinite(times)


def analyse_gearbox_friction_current(
    residual: np.ndarray,
    times: np.ndarray,
    task_labels: np.ndarray,
    commanded_target: np.ndarray,
    lower: float,
    upper: float,
    mean_residual_limit: float | None,
) -> dict[str, Any]:
    """DOF3 gearbox-friction diagnosis from positive current excess in motion.

    Motion selection is performed separately for each task label to prevent
    commanded target differences across separate trials from being counted.
    The currently configured *upper* healthy current band provides the
    initial decision rule; an optional positive-mean threshold is evaluated
    only after calibration from healthy data.
    """
    residual = np.asarray(residual, dtype=float)
    times = np.asarray(times, dtype=float)
    task_labels = np.asarray(task_labels, dtype=str)
    commanded_target = np.asarray(commanded_target, dtype=float)

    if not (
        len(residual) == len(times) == len(task_labels)
        == len(commanded_target)
    ):
        raise ValueError("Gearbox friction input lengths do not match.")

    min_motion_samples = int(
        GEARBOX_FRICTION_DOF3_PARAMS.get("min_motion_samples", 20)
    )
    motion = get_gearbox_motion_mask(
        times=times,
        task_labels=task_labels,
        commanded_target=commanded_target,
    ) & np.isfinite(residual)
    count = int(motion.sum())
    if count < min_motion_samples:
        raise ValueError(
            f"Not enough valid commanded-motion samples for gearbox "
            f"friction diagnosis: {count} < {min_motion_samples}."
        )

    moving_residual = residual[motion]
    outside = motion & (residual > upper)  # only higher measured current
    outside_count = int(outside.sum())
    outside_fraction = outside_count / count
    mean_residual = float(np.mean(moving_residual))

    if mean_residual_limit is not None:
        mean_residual_limit = float(mean_residual_limit)
        if not np.isfinite(mean_residual_limit) or mean_residual_limit < 0:
            raise ValueError("Invalid gearbox mean residual limit.")

    mean_rule_evaluated = mean_residual_limit is not None
    mean_rule_triggered = bool(
        mean_rule_evaluated and mean_residual > mean_residual_limit
    )
    fraction_triggered = bool(outside_fraction > MAX_OUTSIDE_FRACTION)
    deviating = mean_rule_triggered or fraction_triggered

    # Descriptive longest excursion only; duration does not trigger diagnosis.
    longest_outside = 0.0
    for label in np.unique(task_labels):
        idx = np.where(task_labels == label)[0]
        idx = idx[np.argsort(times[idx])]
        if len(idx) == 0:
            continue
        longest_outside = max(
            longest_outside,
            longest_true_duration(outside[idx], times[idx]),
        )

    return {
        "status": "deviating" if deviating else "healthy",
        "analysis_phase": "commanded_motion",
        "diagnosis_method": "gearbox_friction_positive_current",
        "healthy_band": {"lower": lower, "upper": upper, "unit": "mA"},
        "samples": count,
        "samples_outside_band": outside_count,
        "fraction_outside_band": outside_fraction,
        "percentage_outside_band": 100.0 * outside_fraction,
        "maximum_continuous_outside_s": None,
        "longest_continuous_outside_s": longest_outside,
        "minimum_residual": float(np.min(moving_residual)),
        "maximum_residual": float(np.max(moving_residual)),
        "mean_residual": mean_residual,
        "mean_residual_limit": mean_residual_limit,
        "mean_rule_evaluated": mean_rule_evaluated,
        "mean_rule_triggered": mean_rule_triggered,
        "fraction_rule_triggered": fraction_triggered,
        "duration_rule_triggered": False,
        "maximum_absolute_residual": float(np.max(np.abs(moving_residual))),
        "very_large_residual_limit": None,
        "large_transient_rule_triggered": False,
    }


def analyse_signal(
    residual: np.ndarray,
    times: np.ndarray,
    task_labels: np.ndarray,
    lower: float,
    upper: float,
    unit: str,
    max_continuous_outside_s: float | None,
    very_large_residual_limit: float | None = None,
) -> dict[str, Any]:
    """Analyse one residual signal against its healthy band."""

    outside = (
        (residual < lower)
        | (residual > upper)
    )

    sample_count = len(residual)
    outside_count = int(np.sum(outside))

    outside_fraction = (
        outside_count / sample_count
        if sample_count
        else 0.0
    )

    # ---------------------------------------------------------------
    # Longest continuous excursion:
    # calculate separately for every task_label
    # ---------------------------------------------------------------

    longest_outside_s = 0.0

    for task_label in np.unique(task_labels):

        task_mask = (
            task_labels == task_label
        )

        task_outside = outside[
            task_mask
        ]

        task_times = times[
            task_mask
        ]

        if len(task_times) == 0:
            continue

        order = np.argsort(
            task_times
        )

        task_outside = task_outside[
            order
        ]

        task_times = task_times[
            order
        ]

        task_duration = longest_true_duration(
            task_outside,
            task_times,
        )

        longest_outside_s = max(
            longest_outside_s,
            task_duration,
        )

    exceeds_fraction = (
        outside_fraction > MAX_OUTSIDE_FRACTION
    )

    exceeds_duration = (
        max_continuous_outside_s is not None
        and longest_outside_s >= max_continuous_outside_s
    )
    
    maximum_absolute_residual = float(
        np.max(np.abs(residual))
    )

    large_transient_triggered = (
        very_large_residual_limit is not None
        and maximum_absolute_residual
            >= very_large_residual_limit
    )

    deviating = (
        exceeds_fraction
        or exceeds_duration
        or large_transient_triggered
    )

    return {
        "status":
            "deviating"
            if deviating
            else "healthy",

        "healthy_band": {
            "lower": lower,
            "upper": upper,
            "unit": unit,
        },

        "samples":
            sample_count,

        "samples_outside_band":
            outside_count,

        "fraction_outside_band":
            outside_fraction,

        "percentage_outside_band":
            100.0 * outside_fraction,

        "maximum_continuous_outside_s":
            max_continuous_outside_s,

        "longest_continuous_outside_s":
            longest_outside_s,

        "minimum_residual":
            float(np.min(residual)),

        "maximum_residual":
            float(np.max(residual)),

        "fraction_rule_triggered":
            exceeds_fraction,

        "duration_rule_triggered":
            exceeds_duration,

        "maximum_absolute_residual":
            maximum_absolute_residual,

        "very_large_residual_limit":
            very_large_residual_limit,

        "large_transient_rule_triggered":
            large_transient_triggered,

    }

def get_healthy_bands(
    coupling_mode: str,
    motor: int,
    test_type: str,
    dof: int | None,
    condition: str | None = None,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:

    # ---------------------------------------------------------------
    # Motor-only baseline
    # ---------------------------------------------------------------

    if coupling_mode == "motor_only":

        if dof is not None:

            if condition is None:
                raise ValueError(
                    "DOF condition is required for motor-only DOF diagnosis."
                )

            try:
                position_band = (
                    MOTOR_ONLY_DOF_POSITION_BANDS[
                        dof
                    ][condition][motor]
                )

                current_band = (
                    MOTOR_ONLY_DOF_CURRENT_BANDS[
                        dof
                    ][condition][motor]
                )

            except KeyError as exc:
                raise ValueError(
                    f"No healthy bands defined for "
                    f"DOF{dof}, condition '{condition}', "
                    f"motor {motor}."
                ) from exc

            return (
                position_band,
                current_band,
            )

        position_band = (
            MOTOR_ONLY_POSITION_BANDS[
                test_type
            ][motor]
        )

        current_band = (
            MOTOR_ONLY_CURRENT_BANDS[
                test_type
            ][motor]
        )

        return (
            position_band,
            current_band,
        )
    # ---------------------------------------------------------------
    # Gearbox-only baseline
    # ---------------------------------------------------------------

    if coupling_mode == "gearbox_only":

        if dof is None:
            raise ValueError(
                "DOF is required for gearbox_only diagnosis."
            )

        position_band = (
            GEARBOX_POSITION_BANDS[
                dof
            ][motor]
        )

        current_band = (
            GEARBOX_CURRENT_BANDS[
                dof
            ][motor]
        )

        if (
            position_band is None
            or current_band is None
        ):
            raise ValueError(
                f"No gearbox healthy band defined "
                f"for DOF{dof}, motor {motor}."
            )

        return (
            position_band,
            current_band,
        )
    # ---------------------------------------------------------------
    # Full-setup baseline
    # ---------------------------------------------------------------

    if coupling_mode == "full_setup":

        if dof is None:
            raise ValueError(
                "DOF is required for full_setup diagnosis."
            )

        if condition is None:
            raise ValueError(
                "DOF condition is required for full_setup diagnosis."
            )

        try:
            position_band = (
                FULL_SETUP_DOF_POSITION_BANDS[
                    dof
                ][condition][motor]
            )

            current_band = (
                FULL_SETUP_DOF_CURRENT_BANDS[
                    dof
                ][condition][motor]
            )

        except KeyError as exc:
            raise ValueError(
                f"No full-setup healthy bands defined for "
                f"DOF{dof}, condition '{condition}', "
                f"motor {motor}."
            ) from exc

        return (
            position_band,
            current_band,
        )

    raise ValueError(
        f"Unsupported coupling mode: {coupling_mode}"
    )

def get_current_duration_limit(
    coupling_mode: str,
    motor: int,
    dof: int | None,
    condition: str | None,
) -> float | None:

    # Enable condition-specific current duration detection
    # for motor-only and full-setup DOF tests.
    if (
        coupling_mode in (
            "motor_only",
            "full_setup",
        )
        and dof is not None
        and condition is not None
    ):

        try:

            if coupling_mode == "full_setup":
                return (
                    FULL_SETUP_DOF_CURRENT_DURATION_LIMITS[
                        dof
                    ][condition][motor]
                )

            return (
                MOTOR_ONLY_DOF_CURRENT_DURATION_LIMITS[
                    dof
                ][condition][motor]
            )

        except KeyError as exc:
            raise ValueError(
                f"No current duration limit defined for "
                f"{coupling_mode}, DOF{dof}, "
                f"condition '{condition}', motor {motor}."
            ) from exc

    return None

def get_current_peak_limit(
    coupling_mode: str,
    motor: int,
    dof: int | None,
    condition: str | None,
) -> float | None:

    if (
        coupling_mode == "full_setup"
        and dof is not None
        and condition is not None
    ):
        return (
            FULL_SETUP_DOF_CURRENT_PEAK_LIMITS
            .get(dof, {})
            .get(condition, {})
            .get(motor)
        )

    return None

def analyse_motor_test(
    rows: list[dict[str, Any]],
    motor: int,
    test_type: str,
    coupling_mode: str,
    dof: int | None = None,
    condition: str | None = None,
    gearbox_friction_trial: bool = False,
) -> dict[str, Any]:
    
    valid_rows = []

    for row in rows:
        try:
            position_residual = row["encoder_deviation"][motor]
            current_residual = row["filtered_current_deviation"][motor]

            if position_residual is None or current_residual is None:
                continue

            if not np.isfinite(float(position_residual)):
                continue

            if not np.isfinite(float(current_residual)):
                continue

            valid_rows.append(row)

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            continue

    if not valid_rows:
        raise ValueError(
            f"No valid residual samples available for "
            f"motor {motor}, test '{test_type}'."
        )

    rows = sorted(
        valid_rows,
        key=lambda row: float(row["segment_time"]),
    )

    times = np.asarray(
        [
            float(row["segment_time"])
            for row in rows
        ],
        dtype=float,
    )
    
    task_labels = np.asarray(
        [
            str(row["task_label"])
            for row in rows
        ],
        dtype=str,
    )


    position_residual = np.asarray(
        [
            float(row["encoder_deviation"][motor])
            for row in rows
        ],
        dtype=float,
    )

    current_residual = np.asarray(
        [
            float(row["filtered_current_deviation"][motor])
            for row in rows
        ],
        dtype=float,
    )

    (
        position_band,
        current_band,
    ) = get_healthy_bands(
        coupling_mode=coupling_mode,
        motor=motor,
        test_type=test_type,
        dof=dof,
        condition=condition,
    )

    position_lower, position_upper = (
        position_band
    )

    current_lower, current_upper = (
        current_band
    )

    position_result = analyse_signal(
        residual=position_residual,
        times=times,
        task_labels=task_labels,
        lower=position_lower,
        upper=position_upper,
        unit="pulses",
        max_continuous_outside_s= None,
    )

    current_duration_limit = (
        get_current_duration_limit(
            coupling_mode=coupling_mode,
            motor=motor,
            dof=dof,
            condition=condition,
        )
    )

    instrument_config = str(
        rows[0].get(
            "instrument_config",
            "",
        )
    ).strip().lower()

    is_scissor_dof4_current = (
        coupling_mode == "full_setup"
        and dof == 4
        and motor == SCISSOR_DOF4_CURRENT_MOTOR
        and "scissor" in instrument_config
    )

    is_gearbox_friction_dof3 = gearbox_friction_trial

    if is_gearbox_friction_dof3:
        commanded_target = np.asarray(
            [
                float(row["offline_commanded_target"][motor])
                if row["offline_commanded_target"][motor] is not None
                else np.nan
                for row in rows
            ],
            dtype=float,
        )
        # Apply position diagnosis only to the moving samples as well.
        # Keep the full sequence available for the current-result statistics.
        motion_mask = get_gearbox_motion_mask(
            times=times,
            task_labels=task_labels,
            commanded_target=commanded_target,
        )
        min_motion_samples = int(
            GEARBOX_FRICTION_DOF3_PARAMS.get("min_motion_samples", 20)
        )
        if int(motion_mask.sum()) < min_motion_samples:
            raise ValueError(
                "Insufficient valid motion samples for gearbox diagnosis."
            )
        position_result = analyse_signal(
            residual=position_residual[motion_mask],
            times=times[motion_mask],
            task_labels=task_labels[motion_mask],
            lower=position_lower,
            upper=position_upper,
            unit="pulses",
            max_continuous_outside_s=None,
        )

        gearbox_variant = str(rows[0].get("gearbox_variant", "")).strip()
        if gearbox_variant not in ("gearbox_1", "gearbox_2"):
            raise ValueError(
                f"Unknown gearbox variant for friction diagnosis: "
                f"'{gearbox_variant}'."
            )
        variant_limits = (
            GEARBOX_FRICTION_DOF3_PARAMS
            .get("mean_positive_residual_limits_ma", {})
            .get(gearbox_variant, {})
        )
        mean_limit = variant_limits.get(motor, variant_limits.get(str(motor)))
        current_result = analyse_gearbox_friction_current(
            residual=current_residual,
            times=times,
            task_labels=task_labels,
            commanded_target=commanded_target,
            lower=current_lower,
            upper=current_upper,
            mean_residual_limit=mean_limit,
        )
        current_result["gearbox_variant"] = gearbox_variant

    elif is_scissor_dof4_current:

        commanded_target = np.asarray(
            [
                (
                    float(
                        row[
                            "offline_commanded_target"
                        ][motor]
                    )
                    if row[
                        "offline_commanded_target"
                    ][motor] is not None
                    else np.nan
                )
                for row in rows
            ],
            dtype=float,
        )

        mean_residual_limit = (
            FULL_SETUP_SCISSOR_DOF4_MEAN_CLOSING_RESIDUAL_LIMITS
            .get(condition)
        )
        if mean_residual_limit is None:
            raise ValueError(
                f"Missing DOF4 scissor mean closing residual limit for "
                f"condition '{condition}'."
            )

        current_result = (
            analyse_scissor_closing_current(
                residual=current_residual,
                times=times,
                commanded_target=commanded_target,
                lower=current_lower,
                upper=current_upper,
                max_continuous_outside_s=current_duration_limit,
                mean_residual_limit=mean_residual_limit,
            )
        )

    else:

        current_peak_limit = (
            get_current_peak_limit(
                coupling_mode=coupling_mode,
                motor=motor,
                dof=dof,
                condition=condition,
            )
        )

        current_result = analyse_signal(
            residual=current_residual,
            times=times,
            task_labels=task_labels,
            lower=current_lower,
            upper=current_upper,
            unit="mA",
            max_continuous_outside_s=current_duration_limit,
            very_large_residual_limit=current_peak_limit,
        )
        
    deviating = (
        position_result["status"] == "deviating"
        or
        current_result["status"] == "deviating"
    )

    triggered_signals = []

    if position_result["status"] == "deviating":
        triggered_signals.append("position")

    if current_result["status"] == "deviating":
        triggered_signals.append("current")

    return {
        "status":
            "deviating"
            if deviating
            else "healthy",

        "motor":
            motor,

        "test_type":
            test_type,

        "condition":
            condition,

        "triggered_signals":
            triggered_signals,

        "position":
            position_result,

        "current":
            current_result,
    }


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def determine_fault_location(
    coupling_mode: str,
    deviating: bool,
) -> str:
    """Localise a detected deviation based on the tested setup."""

    if not deviating:
        return "none"

    return SETUP_FAULT_LOCATION.get(
        coupling_mode,
        "undetermined",
    )

def invalid_diagnosis(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "detector": DETECTOR_NAME,
        "detector_version": DETECTOR_VERSION,
        "evaluated_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "coupling_mode": "unknown",
        "status": "invalid",

        "probable_fault_location":
            "undetermined",

        "affected_motors": [],

        "invalid_reasons": [reason],
    }


def diagnose_replay(
    replay_path: Path,
) -> dict[str, Any]:

    try:
        rows = load_jsonl(replay_path)
        validate_replay(rows)

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
    ) as exc:
        return invalid_diagnosis(str(exc))

    coupling_mode = str(
        rows[0]["coupling_mode"]
    )
    
    if coupling_mode == "motor_only":

        try:
            dof = detect_dof(
                replay_path,
                rows,
            )

            motors_to_evaluate = (
                DOF_RELEVANT_MOTORS[dof]
            )

        except ValueError:

            dof = None

            motors_to_evaluate = [
                0,
                1,
                2,
                3,
            ]

    elif coupling_mode == "gearbox_only":

        try:
            dof = detect_dof(
                replay_path,
                rows,
            )

        except ValueError as exc:
            return invalid_diagnosis(
                str(exc)
            )

        motors_to_evaluate = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    elif coupling_mode == "full_setup":

        try:
            dof = detect_dof(
                replay_path,
                rows,
            )

        except ValueError as exc:
            return invalid_diagnosis(
                str(exc)
            )

        motors_to_evaluate = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    else:

        return invalid_diagnosis(
            f"Unsupported coupling mode: "
            f"{coupling_mode}"
        )
    
    gearbox_friction_trial = is_gearbox_resistance_trial(
        replay_path, rows, coupling_mode, dof
    )

    grouped = defaultdict(list)

    for row in rows:

        test_type = str(
            row["test_type"]
        )

        if test_type == "idle_baseline":
            continue

        # ---------------------------------------------------------------
        # DOF-controlled pattern:
        # group per waveform + frequency + range
        # for motor_only and full_setup
        # ---------------------------------------------------------------

        if (
            coupling_mode in (
                "motor_only",
                "full_setup",
            )
            and dof is not None
        ):

            sequence_condition_raw = row.get(
                "sequence_condition"
            )

            # Startup / unclassified samples are expected
            # and are not diagnostic motion conditions.
            if sequence_condition_raw is None:
                continue

            sequence_condition = str(
                sequence_condition_raw
            ).strip().lower()

            # Empty values and pauses are not motion conditions.
            # Pause samples are evaluated separately by analyse_pause_rows().
            if sequence_condition in ("", "none", "pause"):
                continue

            try:
                condition = detect_dof_condition(
                    row
                )

            except ValueError as exc:
                print(
                    f"WARNING: invalid sequence condition "
                    f"'{sequence_condition}': {exc}",
                    file=sys.stderr,
                )
                continue

            group_key = (
                test_type,
                condition,
            )

        # ---------------------------------------------------------------
        # Normal motor-only pattern / gearbox
        # ---------------------------------------------------------------

        else:

            if coupling_mode == "motor_only":

                if (
                    test_type not in MOTOR_ONLY_POSITION_BANDS
                    or test_type not in MOTOR_ONLY_CURRENT_BANDS
                ):
                    continue

            group_key = (
                test_type,
                None,
            )

        grouped[
            group_key
        ].append(
            row
        )
    skipped_conditions = set()
    results = []

    for (test_type, condition,), test_rows in grouped.items():

        for motor in motors_to_evaluate:

            try:
                result = analyse_motor_test(
                    rows=test_rows,
                    motor=motor,
                    test_type=test_type,
                    coupling_mode=coupling_mode,
                    dof=dof,
                    condition=condition,
                    gearbox_friction_trial=gearbox_friction_trial,
                )

            except ValueError as exc:

                skipped_conditions.add((condition, motor))

                print(
                    f"WARNING: skipping diagnostics for "
                    f"DOF{dof if dof is not None else '-'}, "
                    f"motor {motor}, test '{test_type}', "
                    f"condition '{condition}': {exc}",
                    file=sys.stderr,
                )
                continue

            results.append(result)

    if not results:
        return invalid_diagnosis(
            "No valid motor/test combinations were evaluated."
        )
    pause_analysis = None

    if (
        coupling_mode in (
            "motor_only",
            "full_setup",
        )
        and dof is not None
    ):
        pause_analysis = analyse_pause_rows(
            rows=rows,
            motors=motors_to_evaluate,
            coupling_mode=coupling_mode,
            dof=dof,
        )

    evaluation_coverage = (
        "partial"
        if skipped_conditions
        else "complete"
    )
    # A missing gearbox-motor result cannot establish a healthy gearbox.
    gearbox_incomplete = gearbox_friction_trial and bool(skipped_conditions)

    deviating_results = [
        result
        for result in results
        if result["status"] == "deviating"
    ]

    pause_deviating = (
        pause_analysis is not None
        and pause_analysis.get("status")
            == "deviating"
    )

    motion_affected_motors = {
        result["motor"]
        for result in deviating_results
    }

    pause_affected_motors = set(
        pause_analysis.get(
            "affected_motors",
            [],
        )
        if pause_analysis
        else []
    )

    affected_motors = sorted(
        motion_affected_motors
        | pause_affected_motors
    )

    deviating = (
        bool(deviating_results)
        or pause_deviating
    )

    motor_results = {}

    for motor in motors_to_evaluate:

        motor_tests = [
            result
            for result in results
            if result["motor"] == motor
        ]

        motor_deviations = [
            result
            for result in motor_tests
            if result["status"] == "deviating"
        ]

        pause_motor_result = (
            pause_analysis
            .get(
                "motor_results",
                {},
            )
            .get(
                f"motor_{motor}"
            )
            if pause_analysis
            else None
        )

        pause_motor_deviating = (
            pause_motor_result is not None
            and pause_motor_result.get("status")
                == "deviating"
        )

        motor_is_deviating = (
            bool(motor_deviations)
            or pause_motor_deviating
        )

        motor_results[f"motor_{motor}"] = {
            "status":
                "deviating"
                if motor_is_deviating
                else ("not_evaluated" if not motor_tests else "healthy"),

            "tests":
                motor_tests,

            "pause_baseline":
                pause_motor_result,
        }

    position_deviations = sum(
        result["position"]["status"] == "deviating"
        for result in results
    )

    current_deviations = sum(
        result["current"]["status"] == "deviating"
        for result in results
    )

    return {
        "schema_version": 3,

        "detector":
            DETECTOR_NAME,

        "detector_version":
            DETECTOR_VERSION,

        "evaluated_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "coupling_mode":
            coupling_mode,

        "dof":
            dof,

        "evaluation_coverage":
            evaluation_coverage,

        "skipped_conditions": [
            {
                "condition": condition,
                "motor": motor,
            }
            for condition, motor
            in sorted(skipped_conditions, key=lambda item: (str(item[0]), item[1]))
        ],

        "pause_analysis":
            pause_analysis,

        "status":
            "deviating"
            if deviating
            else ("invalid" if gearbox_incomplete else "healthy"),

        "probable_fault_location": (
            "undetermined"
            if gearbox_incomplete and not deviating
            else determine_fault_location(
                coupling_mode=coupling_mode,
                deviating=deviating,
            )
        ),

        "affected_motors":
            affected_motors,

        "baseline_reference": {
            "name":
                (
                    MOTOR_ONLY_BASELINE_NAME
                    if coupling_mode == "motor_only"
                    else (
                        FULL_SETUP_BASELINE_NAME
                        if coupling_mode == "full_setup"
                        else GEARBOX_BASELINE_NAME
                    )
                ),

            "method": (
                "gearbox DOF3: positive current excess during motion "
                "(upper-band fraction; optional calibrated mean); "
                "position residual bands"
                if gearbox_friction_trial
                else "fixed healthy position and current residual bands "
                     "with DOF-specific rules"
            ),

            "maximum_fraction_outside":
                MAX_OUTSIDE_FRACTION,

            "position_maximum_continuous_outside_s":
                None, #POSITION_MAX_CONTINUOUS_OUTSIDE_S,

            "current_maximum_continuous_outside_s":
                None, #CURRENT_MAX_CONTINUOUS_OUTSIDE_S,

            "current_signal":
                "filtered_current_deviation",
        },

        "summary": {
            "evaluated_motor_test_combinations":
                len(results),

            "deviating_motor_test_combinations":
                len(deviating_results),

            "position_deviations":
                int(position_deviations),

            "current_deviations":
                int(current_deviations),

            "pause_baseline_status":
                pause_analysis.get("status", "not_evaluated")
                if pause_analysis
                else "not_evaluated",
            
        },

        "motor_results":
            motor_results,
    }

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def create_diagnostic_plots(
    replay_path: Path,
    output_dir: Path,
) -> list[str]:
    """Create one diagnostic plot per motor using only that motor's active tasks."""

    rows = load_jsonl(replay_path)
    validate_replay(rows)

    coupling_mode = str(
        rows[0]["coupling_mode"]
    )

    if coupling_mode == "gearbox_only":

        dof = detect_dof(
            replay_path,
            rows,
        )

        motors_to_plot = (
            DOF_RELEVANT_MOTORS[dof]
        )

    elif coupling_mode == "motor_only":

        try:
            dof = detect_dof(
                replay_path,
                rows,
            )

            motors_to_plot = (
                DOF_RELEVANT_MOTORS[dof]
            )

        except ValueError:

            dof = None

            motors_to_plot = [
                0,
                1,
                2,
                3,
            ]

    elif coupling_mode == "full_setup":

        dof = detect_dof(
            replay_path,
            rows,
        )

        motors_to_plot = (
            DOF_RELEVANT_MOTORS[
                dof
            ]
        )

    gearbox_friction_trial = is_gearbox_resistance_trial(
        replay_path, rows, coupling_mode, dof
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_paths = []

    # Group complete replay by task_label.
    task_segments = defaultdict(list)

    for row in rows:
        task_segments[
            str(row["task_label"])
        ].append(row)

    for motor in motors_to_plot:

        # ---------------------------------------------------------------
        # Select only tasks in which this motor is actually commanded
        # ---------------------------------------------------------------

        motor_rows = []

        for task_label, segment in task_segments.items():

            segment = sorted(
                segment,
                key=lambda row: float(row["segment_time"]),
            )

            if not segment:
                continue

            test_type = str(
                segment[0]["test_type"]
            )

            # Skip idle baseline.
            if test_type == "idle_baseline":
                continue

            # Skip tests for which no diagnostic band exists.
            if coupling_mode == "motor_only" and dof is None:

                if (
                    test_type not in MOTOR_ONLY_POSITION_BANDS
                    or test_type not in MOTOR_ONLY_CURRENT_BANDS
                ):
                    continue

            if coupling_mode == "motor_only" and dof is None:

                commanded_target = np.asarray(
                    [
                        float(
                            row["offline_commanded_target"][motor]
                        )
                        for row in segment
                    ],
                    dtype=float,
                )

                if len(commanded_target) < 2:
                    continue

                target_change = np.diff(
                    commanded_target
                )

                motor_is_active = np.any(
                    np.abs(target_change) > 1e-12
                )

                if not motor_is_active:
                    continue

            # For gearbox_only, the relevant motors are already selected
            # through DOF_RELEVANT_MOTORS[dof].

            motor_rows.extend(
                segment
            )

        if not motor_rows:
            print(
                f"WARNING: no active diagnostic rows found "
                f"for motor {motor}.",
                file=sys.stderr,
            )
            continue

        # ---------------------------------------------------------------
        # Remove rows with missing / invalid values required for plotting
        # ---------------------------------------------------------------

        valid_motor_rows = []

        for row in motor_rows:

            try:

                values = [
                    row["offline_predicted_positions"][motor],
                    row["measured_positions"][motor],
                    row["encoder_deviation"][motor],

                    row["offline_predicted_currents"][motor],
                    row["filtered_measured_currents"][motor],
                    row["filtered_current_deviation"][motor],
                ]

                if any(
                    value is None
                    for value in values
                ):
                    continue

                if not all(
                    np.isfinite(float(value))
                    for value in values
                ):
                    continue

                valid_motor_rows.append(
                    row
                )

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

        motor_rows = valid_motor_rows

        if not motor_rows:

            print(
                f"WARNING: no valid plot rows found "
                f"for motor {motor}.",
                file=sys.stderr,
            )

            continue

        # ---------------------------------------------------------------
        # Continuous plot x-axis for only this motor's own tasks
        # ---------------------------------------------------------------

        # Keep only DOF motion conditions for which healthy bands exist.
        if (
            coupling_mode in (
                "motor_only",
                "full_setup",
            )
            and dof is not None
        ):

            filtered_rows = []

            for row in motor_rows:
                condition = str(
                    row.get("sequence_condition", "")
                ).strip().lower()

                if condition in ("", "none", "pause"):
                    continue

                try:
                    get_healthy_bands(
                        coupling_mode=coupling_mode,
                        motor=motor,
                        test_type=str(row["test_type"]),
                        dof=dof,
                        condition=condition,
                    )
                except ValueError:
                    continue

                filtered_rows.append(row)

            motor_rows = filtered_rows

            if not motor_rows:
                print(
                    f"WARNING: no DOF rows with defined healthy bands "
                    f"for motor {motor}.",
                    file=sys.stderr,
                )
                continue


        x = np.arange(
            len(motor_rows)
        )

        test_types = [
            str(row["test_type"])
            for row in motor_rows
        ]

        conditions = []

        for row in motor_rows:

            if (
                coupling_mode in (
                    "motor_only",
                    "full_setup",
                )
                and dof is not None
            ):
                condition = detect_dof_condition(
                    row
                )
            else:
                condition = None

            conditions.append(
                condition
            )
        # ---------------------------------------------------------------
        # Position signals
        # ---------------------------------------------------------------

        predicted_position = np.asarray(
            [
                float(
                    row["offline_predicted_positions"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        measured_position = np.asarray(
            [
                float(
                    row["measured_positions"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        position_residual = np.asarray(
            [
                float(
                    row["encoder_deviation"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        # ---------------------------------------------------------------
        # Current signals
        # ---------------------------------------------------------------

        predicted_current = np.asarray(
            [
                float(
                    row["offline_predicted_currents"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        measured_current = np.asarray(
            [
                float(
                    row["filtered_measured_currents"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        current_residual = np.asarray(
            [
                float(
                    row["filtered_current_deviation"][motor]
                )
                for row in motor_rows
            ],
            dtype=float,
        )

        # ---------------------------------------------------------------
        # Healthy bands for every selected sample
        # ---------------------------------------------------------------

        position_lower = np.full(
            len(motor_rows),
            np.nan,
        )

        position_upper = np.full(
            len(motor_rows),
            np.nan,
        )

        current_lower = np.full(
            len(motor_rows),
            np.nan,
        )

        current_upper = np.full(
            len(motor_rows),
            np.nan,
        )

        for i, (
            test_type,
            condition,
        ) in enumerate(
            zip(
                test_types,
                conditions,
            )
        ):

            (
                position_band,
                current_band,
            ) = get_healthy_bands(
                coupling_mode=coupling_mode,
                motor=motor,
                test_type=test_type,
                condition=condition,
                dof=dof,
            )

            position_lower[i] = (
                position_band[0]
            )

            position_upper[i] = (
                position_band[1]
            )

            current_lower[i] = (
                current_band[0]
            )

            current_upper[i] = (
                current_band[1]
            )

        # ---------------------------------------------------------------
        # Healthy envelopes around DT prediction
        # ---------------------------------------------------------------

        position_prediction_lower = (
            predicted_position
            + position_lower
        )

        position_prediction_upper = (
            predicted_position
            + position_upper
        )

        current_prediction_lower = (
            predicted_current
            + current_lower
        )

        current_prediction_upper = (
            predicted_current
            + current_upper
        )

        # ---------------------------------------------------------------
        # Samples outside the healthy residual bands
        # ---------------------------------------------------------------

        position_outside = (
            (position_residual < position_lower)
            |
            (position_residual > position_upper)
        )

        if gearbox_friction_trial:
            # Use the same motion selection as the gearbox friction diagnosis.
            motion = get_gearbox_motion_mask(
                times=np.asarray(
                    [float(row["segment_time"]) for row in motor_rows],
                    dtype=float,
                ),
                task_labels=np.asarray(
                    [str(row["task_label"]) for row in motor_rows],
                    dtype=str,
                ),
                commanded_target=np.asarray(
                    [
                        float(row["offline_commanded_target"][motor])
                        if row["offline_commanded_target"][motor] is not None
                        else np.nan
                        for row in motor_rows
                    ],
                    dtype=float,
                ),
            )
            # Position deviations are evaluated during commanded motion only.
            position_outside &= motion
            current_outside = (current_residual > current_upper) & motion
        else:
            current_outside = (
                (current_residual < current_lower)
                | (current_residual > current_upper)
            )

        # ---------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 14),
            sharex=True,
        )

        # ---------------------------------------------------------------
        # Position prediction
        # ---------------------------------------------------------------

        # axes[0].fill_between(
        #     x,
        #     position_prediction_lower,
        #     position_prediction_upper,
        #     alpha=0.20,
        #     label="Healthy prediction envelope",
        # )

        axes[0].plot(
            x,
            measured_position,
            linewidth=1.0,
            label="Measured position",
        )

        axes[0].plot(
            x,
            predicted_position,
            linestyle="--",
            linewidth=1.0,
            label="DT predicted position",
        )

        axes[0].set_ylabel(
            "Position [pulses]"
        )

        axes[0].set_title(
            f"Motor {motor} - Position prediction"
        )

        axes[0].grid(
            True,
            alpha=0.25,
        )

        axes[0].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Position residual
        # ---------------------------------------------------------------

        axes[1].fill_between(
            x,
            position_lower,
            position_upper,
            alpha=0.20,
            label="Healthy residual band",
        )

        axes[1].plot(
            x,
            position_residual,
            linewidth=0.9,
            label="Position residual",
        )

        if np.any(
            position_outside
        ):
            axes[1].scatter(
                x[position_outside],
                position_residual[
                    position_outside
                ],
                s=10,
                label="Outside healthy band",
            )

        axes[1].axhline(
            0.0,
            linewidth=0.8,
        )

        axes[1].set_ylabel(
            "Residual [pulses]"
        )

        axes[1].set_title(
            "Position residual diagnosis"
        )

        axes[1].grid(
            True,
            alpha=0.25,
        )

        axes[1].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Current prediction
        # ---------------------------------------------------------------

        # axes[2].fill_between(
        #     x,
        #     current_prediction_lower,
        #     current_prediction_upper,
        #     alpha=0.20,
        #     label="Healthy prediction envelope",
        # )

        axes[2].plot(
            x,
            measured_current,
            linewidth=0.9,
            label="Filtered measured current",
        )

        axes[2].plot(
            x,
            predicted_current,
            linestyle="--",
            linewidth=0.9,
            label="DT predicted current",
        )

        axes[2].set_ylabel(
            "Current [mA]"
        )

        axes[2].set_title(
            f"Motor {motor} - Current prediction"
        )

        axes[2].grid(
            True,
            alpha=0.25,
        )

        axes[2].legend(
            loc="upper right"
        )

        # ---------------------------------------------------------------
        # Current residual
        # ---------------------------------------------------------------

        axes[3].fill_between(
            x,
            current_lower,
            current_upper,
            alpha=0.20,
            label=(
                "Reference residual band (only upper bound triggers)"
                if gearbox_friction_trial
                else "Healthy residual band"
            ),
        )

        axes[3].plot(
            x,
            current_residual,
            linewidth=0.9,
            label="Current residual",
        )

        if np.any(
            current_outside
        ):
            axes[3].scatter(
                x[current_outside],
                current_residual[
                    current_outside
                ],
                s=10,
                label="Outside healthy band",
            )

        axes[3].axhline(
            0.0,
            linewidth=0.8,
        )

        axes[3].set_xlabel(
            "Sample"
        )

        axes[3].set_ylabel(
            "Residual [mA]"
        )

        axes[3].set_title(
            "Current residual (positive upper bound; motion-only diagnosis)"
            if gearbox_friction_trial
            else "Current residual diagnosis"
        )

        axes[3].grid(
            True,
            alpha=0.25,
        )

        axes[3].legend(
            loc="upper right"
        )

        fig.suptitle(
            f"Digital Twin diagnosis - Motor {motor}",
            fontsize=16,
        )

        fig.tight_layout()

        plot_path = (
            output_dir
            / f"digital_twin_diagnosis_motor_{motor}.png"
        )

        fig.savefig(
            plot_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        plot_paths.append(
            str(plot_path)
        )

    return plot_paths

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def update_metadata(
    metadata_path: Path,
    diagnosis: dict[str, Any],
) -> None:

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    metadata["diagnostics"] = diagnosis

    temporary_name = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:

            temporary_name = handle.name

            json.dump(
                metadata,
                handle,
                indent=2,
                allow_nan=False,
            )

            handle.write("\n")

        os.replace(
            temporary_name,
            metadata_path,
        )

    finally:

        if (
            temporary_name
            and os.path.exists(temporary_name)
        ):
            os.unlink(temporary_name)

def write_diagnostics_file(
    diagnostics_path: Path,
    diagnosis: dict[str, Any],
) -> None:

    diagnostics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with diagnostics_path.open(
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

# ---------------------------------------------------------------------------
# Popup
# ---------------------------------------------------------------------------

def show_diagnosis_popup(
    diagnosis: dict[str, Any],
) -> bool:

    try:
        import tkinter as tk
        from tkinter import ttk

    except ImportError:
        print(
            "WARNING: diagnosis popup unavailable; "
            "install python3-tk, "
            "for Ubuntu use sudo apt install python3-tk.",
            file=sys.stderr,
        )
        return False

    status = str(
        diagnosis.get(
            "status",
            "invalid",
        )
    )

    colors = {
        "healthy":
            ("#176B35", "#F1FFF5"),

        "deviating":
            ("#8B1E1E", "#FFF4F4"),

        "invalid":
            ("#9A5B00", "#FFF8E8"),
    }

    accent, background = colors.get(
        status,
        colors["invalid"],
    )

    try:
        root = tk.Tk()

    except tk.TclError as exc:
        print(
            f"WARNING: diagnosis popup unavailable: {exc}",
            file=sys.stderr,
        )
        return False

    root.title(
        f"Digital Twin diagnosis - {status.upper()}"
    )

    root.geometry(
        "1450x750"
    )

    root.minsize(
        1050,
        600,
    )

    root.configure(
        bg=background,
    )

    root.attributes(
        "-topmost",
        True,
    )

    heading = {
        "healthy":
            "TEST HEALTHY",

        "deviating":
            "DEVIATION DETECTED",

        "invalid":
            "TEST INVALID",
    }.get(
        status,
        "DIAGNOSIS",
    )

    tk.Label(
        root,
        text=heading,
        font=(
            "DejaVu Sans",
            20,
            "bold",
        ),
        fg="white",
        bg=accent,
        pady=12,
    ).pack(
        fill="x",
    )

    notebook = ttk.Notebook(
        root
    )

    notebook.pack(
        fill="both",
        expand=True,
        padx=15,
        pady=15,
    )

    # ------------------------------------------------------------------
    # SUMMARY TAB
    # ------------------------------------------------------------------

    summary_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        summary_tab,
        text="Summary",
    )

    affected = diagnosis.get(
        "affected_motors",
        [],
    )

    affected_text = (
        ", ".join(
            f"M{motor}"
            for motor in affected
        )
        if affected
        else "none"
    )

    summary_data = diagnosis.get(
        "summary",
        {},
    )

    summary_lines = [
        f"Overall result: {status.upper()}",
        "",
        (
            "Configuration: "
            f'{diagnosis.get("coupling_mode", "unknown")}'
        ),
        (
            f'DOF: {diagnosis.get("dof")}'
            if diagnosis.get("dof") is not None
            else ""
        ),
        (
            "Evaluation coverage: "
                f'{diagnosis.get("evaluation_coverage", "complete").upper()}'
        ),

        f"Affected motor(s): {affected_text}",
        "",
        (
            "Motor DT probable location: "
            f'{diagnosis.get("probable_fault_location", "none")}'
        ),
        (
            "Instrument output: "
            f'{summary_data.get(
                "instrument_output_status",
                "not_evaluated",
            ).upper()}'
        ),
        (
            "Instrument output possible location: "
            f'{diagnosis.get(
                "instrument_output_fault_location",
                "none",
            )}'
        ),
        (
            "Position deviations: "
            f'{summary_data.get("position_deviations", 0)}'
        ),
        (
            "Current deviations: "
            f'{summary_data.get("current_deviations", 0)}'
        ),
        (
            "Pause baseline: "
            f'{summary_data.get(
                "pause_baseline_status",
                "not_evaluated",
            ).upper()}'
        ),
        "",
        (
            "DOF4 scissors M3: mean positive current excess during closing; "
            "band and duration triggers disabled."
            if diagnosis.get("coupling_mode") == "full_setup"
            and diagnosis.get("dof") == 4
            else (
                "Gearbox DOF3 resistance: positive upper-band fraction "
                f"> {100.0 * MAX_OUTSIDE_FRACTION:.0f}% during motion; "
                "optional calibrated mean rule."
                if diagnosis.get("coupling_mode") == "gearbox_only"
                and diagnosis.get("dof") == 3
                else (
                    "Detection rule: more than "
                    f"{100.0 * MAX_OUTSIDE_FRACTION:.0f}% of samples "
                    "outside the healthy band (with configured duration/peak rules)."
                )
            )
        ),
    ]

    tk.Label(
        summary_tab,
        text="\n".join(
            summary_lines
        ),
        justify="left",
        anchor="nw",
        font=(
            "DejaVu Sans",
            13,
        ),
        bg=background,
        padx=25,
        pady=25,
    ).pack(
        fill="both",
        expand=True,
    )

    # ------------------------------------------------------------------
    # DETAILS TAB
    # ------------------------------------------------------------------

    details_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        details_tab,
        text="Details",
    )

    columns = (
        "motor",
        "test",
        "condition",
        "signal",
        "band",
        "outside",
        "duration",
        "trigger",
        "range",
        "status",
    )

    tree = ttk.Treeview(
        details_tab,
        columns=columns,
        show="headings",
    )

    labels = {
        "motor":
            "Motor",

        "test":
            "Test type",

        "condition":
            "Test condition",

        "signal":
            "Signal",

        "band":
            "Healthy band",

        "outside":
            "Samples outside band",

        "duration":
            "Longest excursion",

        "trigger":
            "Trigger",

        "range":
            "Observed residual",

        "status":
            "Result",
    }

    widths = {
        "motor": 60,
        "test": 210,
        "condition": 220,
        "signal": 90,
        "band": 180,
        "outside": 200,
        "duration": 160,
        "trigger": 110,
        "range": 210,
        "status": 110,
    }

    for column in columns:

        tree.heading(
            column,
            text=labels[column],
        )

        tree.column(
            column,
            width=widths[column],
            anchor="w",
        )

    scrollbar = ttk.Scrollbar(
        details_tab,
        orient="vertical",
        command=tree.yview,
    )

    tree.configure(
        yscrollcommand=scrollbar.set,
    )

    tree.pack(
        side="left",
        fill="both",
        expand=True,
        padx=10,
        pady=10,
    )

    scrollbar.pack(
        side="right",
        fill="y",
        pady=10,
    )

    for motor_name, motor_result in (
        diagnosis
        .get("motor_results", {})
        .items()
    ):

        motor_label = motor_name.replace(
            "motor_",
            "M",
        )

        for test in motor_result.get(
            "tests",
            [],
        ):

            for signal_name in (
                "position",
                "current",
            ):

                signal = test[
                    signal_name
                ]

                band = signal[
                    "healthy_band"
                ]

                unit = band["unit"]

                band_text = (
                    f'{band["lower"]:.0f} '
                    f'to '
                    f'{band["upper"]:.0f} '
                    f'{unit}'
                )

                outside_text = (
                    f'{signal["samples_outside_band"]}'
                    f'/{signal["samples"]} '
                    f'('
                    f'{signal["percentage_outside_band"]:.1f}%'
                    f')'
                )

                duration_limit = signal.get(
                    "maximum_continuous_outside_s"
                )

                if duration_limit is not None:

                    duration_text = (
                        f'{1000.0 * signal["longest_continuous_outside_s"]:.0f} '
                        f'/ '
                        f'{1000.0 * duration_limit:.0f} ms'
                    )

                else:

                    duration_text = (
                        f'{1000.0 * signal["longest_continuous_outside_s"]:.0f} ms'
                    )

                if signal.get("analysis_phase") in {"closing", "commanded_motion"}:

                    mean_residual = signal.get(
                        "mean_residual"
                    )

                    mean_limit = signal.get(
                        "mean_residual_limit"
                    )

                    if mean_limit is not None:
                        range_text = (
                            f'mean {mean_residual:.1f} / '
                            f'limit {mean_limit:.1f} {unit}'
                        )
                    else:
                        range_text = (
                            f'mean {mean_residual:.1f} {unit} '
                            f'(mean rule disabled; upper band active)'
                        )

                else:
                    range_text = (
                        f'{signal["minimum_residual"]:.1f} '
                        f'to '
                        f'{signal["maximum_residual"]:.1f} '
                        f'{unit}'
                    )

                fraction_triggered = signal.get(
                    "fraction_rule_triggered",
                    False,
                )

                duration_triggered = signal.get(
                    "duration_rule_triggered",
                    False,
                )

                large_transient_triggered = signal.get(
                    "large_transient_rule_triggered",
                    False,
                )

                mean_triggered = signal.get(
                    "mean_rule_triggered",
                    False,
                )

                trigger_parts = []

                if fraction_triggered:
                    trigger_parts.append("fraction")

                if duration_triggered:
                    trigger_parts.append("duration")

                if large_transient_triggered:
                    trigger_parts.append("large transient")

                if mean_triggered:
                    trigger_parts.append("mean_closing")

                trigger_text = (
                    " + ".join(trigger_parts)
                    if trigger_parts
                    else "-"
                )

                tree.insert(
                    "",
                    "end",
                    values=(
                        motor_label,
                        test["test_type"],
                        test.get("condition") or "-",
                        signal_name,
                        band_text,
                        outside_text,
                        duration_text,
                        trigger_text,
                        range_text,
                        signal["status"],
                    ),
                )

    # ------------------------------------------------------------------
    # OUTPUT FILES TAB
    # ------------------------------------------------------------------

    files_tab = tk.Frame(
        notebook,
        bg=background,
    )

    notebook.add(
        files_tab,
        text="Output files",
    )

    plot_paths = diagnosis.get(
        "plot_files",
        [],
    )

    diagnostics_file = diagnosis.get(
        "diagnostics_file",
        "not written",
    )

    file_lines = [
        "Diagnostic plots:",
        "",
    ]

    file_lines.extend(
        plot_paths
    )

    file_lines.extend([
        "",
        "Detailed diagnostics JSON:",
        str(diagnostics_file),
    ])

    tk.Label(
        files_tab,
        text="\n".join(
            file_lines
        ),
        justify="left",
        anchor="nw",
        font=(
            "DejaVu Sans Mono",
            10,
        ),
        bg=background,
        padx=20,
        pady=20,
    ).pack(
        fill="both",
        expand=True,
    )

    def close_popup():
        root.quit()
        root.destroy()


    tk.Button(
        root,
        text="Close and continue",
        command=close_popup,
        font=(
            "DejaVu Sans",
            12,
            "bold",
        ),
        bg=accent,
        fg="white",
        padx=22,
        pady=8,
    ).pack(
        pady=12,
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        close_popup,
    )

    root.mainloop()

    return True

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--replay-file",
        "--file",
        dest="replay_file",
        help="Motor Digital Twin replay JSONL",
    )

    parser.add_argument(
        "--popup-file",
        help="Show popup from an existing diagnostics JSON file",
    )

    parser.add_argument(
        "--metadata-file",
        help="Existing run metadata JSON to update",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diagnosis without modifying metadata",
    )

    parser.add_argument(
        "--no-popup",
        action="store_true",
        help="Do not show diagnosis popup",
    )

    return parser


def main() -> None:

    args = build_parser().parse_args()

    # Standalone popup mode.
    if args.popup_file:

        popup_path = (
            Path(args.popup_file)
            .expanduser()
            .resolve()
        )

        with popup_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            diagnosis = json.load(handle)

        show_diagnosis_popup(
            diagnosis
        )

        return

    if not args.replay_file or not args.metadata_file:
        raise SystemExit(
            "--replay-file and --metadata-file are required "
            "unless --popup-file is used."
        )

    replay_path = (
        Path(args.replay_file)
        .expanduser()
        .resolve()
    )

    metadata_path = (
        Path(args.metadata_file)
        .expanduser()
        .resolve()
    )

    # ------------------------------------------------------------------
    # Run diagnosis
    # ------------------------------------------------------------------

    diagnosis = diagnose_replay(
        replay_path,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    diagnostics_dir = (
        metadata_path.parent
        / "diagnostics"
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Instrument output diagnosis
    # ------------------------------------------------------------------
    #
    # Currently implemented for full_setup DOF2 when video is available.
    # A constant video/prediction offset is removed before directional
    # bending gains are evaluated, so initialisation offset is not itself
    # a fault trigger.

    instrument_output_analysis = None

    if diagnosis.get("status") != "invalid":

        try:
            instrument_output_analysis = (
                diagnose_dof2_from_metadata(
                    metadata_path,
                    prediction_source="hybrid",
                )
            )

        except Exception as exc:
            instrument_output_analysis = {
                "status": "not_evaluated",
                "reason": str(exc),
            }

            print(
                "WARNING: instrument output diagnostics "
                f"could not be evaluated: {exc}",
                file=sys.stderr,
            )

    diagnosis["instrument_output_analysis"] = (
        instrument_output_analysis
    )

    if instrument_output_analysis is not None:

        instrument_status = (
            instrument_output_analysis.get(
                "status",
                "not_evaluated",
            )
        )

        diagnosis.setdefault(
            "summary",
            {},
        )["instrument_output_status"] = (
            instrument_status
        )
        # for now instrument output faults are gearbox or instrument, later deviation between those
        if instrument_status == "deviating":
            diagnosis["status"] = "deviating"

            diagnosis["instrument_output_fault_location"] = (
                "gearbox_or_instrument"
            )
            # Keep the overall location consistent if camera detects a
            # deviation not already indicated by the Motor DT.
            diagnosis["probable_fault_location"] = (
                "gearbox_or_instrument"
            )
    # ------------------------------------------------------------------
    # Create diagnostic plots
    # ------------------------------------------------------------------

    plot_files = []

    if diagnosis.get("status") != "invalid":

        try:

            plot_files = create_diagnostic_plots(
                replay_path,
                diagnostics_dir,
            )

        except Exception as exc:

            print(
                f"WARNING: could not create "
                f"diagnostic plots: {exc}",
                file=sys.stderr,
            )

    if (
        instrument_output_analysis is not None
        and instrument_output_analysis.get("status")
            in {"healthy", "deviating"}
    ):

        try:
            instrument_plot_path = (
                diagnostics_dir
                / "instrument_dof2_directional_gain.png"
            )

            create_gain_plot(
                instrument_output_analysis,
                instrument_plot_path,
            )

            if instrument_plot_path.exists():
                plot_files.append(
                    str(instrument_plot_path)
                )

        except Exception as exc:

            print(
                "WARNING: could not create instrument "
                f"diagnostic plot: {exc}",
                file=sys.stderr,
            )

    diagnosis["plot_files"] = (
        plot_files
    )

    # ------------------------------------------------------------------
    # Separate diagnostics JSON
    # ------------------------------------------------------------------

    diagnostics_path = (
        diagnostics_dir
        / "digital_twin_diagnosis.json"
    )

    diagnosis["diagnostics_file"] = (
        str(diagnostics_path)
    )

    if not args.dry_run:

        try:

            write_diagnostics_file(
                diagnostics_path,
                diagnosis,
            )

        except Exception as exc:

            print(
                f"WARNING: could not write "
                f"diagnostics JSON: {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    if not args.dry_run:

        try:

            update_metadata(
                metadata_path,
                diagnosis,
            )

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:

            raise SystemExit(
                f"Could not update metadata: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------------

    print(
        json.dumps(
            diagnosis,
            indent=2,
            allow_nan=False,
        )
    )

    if args.dry_run:

        print(
            "Dry run: metadata was not changed."
        )

    else:

        print(
            f"Updated metadata: {metadata_path}"
        )

        print(
            f"Diagnostics directory: "
            f"{diagnostics_dir}"
        )

    # ------------------------------------------------------------------
    # Start popup as separate process
    # ------------------------------------------------------------------

    if (
        not args.dry_run
        and not args.no_popup
    ):

        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--popup-file",
                str(diagnostics_path),
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()