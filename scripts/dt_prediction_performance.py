#!/usr/bin/env python3
"""Thesis-ready Digital Twin prediction-performance evaluation.

This version is tailored to the existing SATA training-data layout:

    trainings data V6/
      motors_only_dof_pattern/DOF 1..4/*.jsonl
      motors_gearbox/gearbox_1/DOF 1..4/*.jsonl
      motors_gearbox/gearbox_2/DOF 1..4/*.jsonl
      full_setup/gripper/DOF 1..4/*.jsonl
      full_setup/gripper/DOF 2/bend_dt_results/bend_training_dataset.csv

Motor DT
--------
Per mechanical dataset, all nominal ROS logs are discovered automatically.
True leave-one-run-out (LORO) validation is then performed at file/run level.
For every fold, the selected model family is FIXED (default: random_forest),
so the held-out run is not used for model-family selection.

The feature-building and preprocessing functions are imported from the existing
train_motor_digital_twin.py. This keeps the evaluation consistent with the
normal training pipeline without modifying the production models.

Hybrid DOF2 Instrument DT
-------------------------
The already-built synchronized bend_training_dataset.csv is used. One complete
trial_id is held out per fold. The residual correction model is trained on all
remaining trials and evaluated on the held-out trial. Both the physics-based
and hybrid predictions are reported against the same video reference.

Physics-based DOF4 Instrument DT
--------------------------------
ROS logs and matching *_webcam_angles.jsonl files are discovered automatically
and compared directly; no cross-validation is required because the physics
model is not refitted to the validation runs.

Error convention
----------------
    error = measurement - prediction

This matches the thesis Methods definition. Positive bias therefore means the
model under-predicts the measured value on average.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor


SCRIPT_VERSION = "2.0.0"

CONFIG_LABELS = {
    "motor_only": "Motor only",
    "gearbox_only": "Motor--gearbox",
    "full_setup": "Full system",
}


# ============================================================================
# Generic helpers
# ============================================================================


def expand_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def calculate_metrics(
    measurement: Iterable[float],
    prediction: Iterable[float],
) -> dict[str, float]:
    y = np.asarray(list(measurement), dtype=float)
    y_hat = np.asarray(list(prediction), dtype=float)

    if y.shape != y_hat.shape:
        raise ValueError(f"Shape mismatch: {y.shape} != {y_hat.shape}")

    valid = np.isfinite(y) & np.isfinite(y_hat)
    if not np.any(valid):
        raise ValueError("No finite measurement/prediction pairs available.")

    y = y[valid]
    y_hat = y_hat[valid]

    # Thesis convention: measured - predicted.
    error = y - y_hat

    return {
        "n_samples": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def sample_mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan
    return mean, sd, int(len(values))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def latex_mean_sd(mean: float, sd: float, decimals: int) -> str:
    if not math.isfinite(float(mean)):
        return "-- $\\pm$ --"
    if not math.isfinite(float(sd)):
        return f"{mean:.{decimals}f} $\\pm$ --"
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return rows


def vector_value(row: dict[str, Any], key: str, index: int) -> float | None:
    value = row.get(key)
    if not isinstance(value, (list, tuple)) or len(value) <= index:
        return None
    return safe_float(value[index])


# ============================================================================
# Discovery
# ============================================================================


def discover_ros_logs(root: Path) -> list[Path]:
    """Discover raw ROS logs, excluding result/output folders."""
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    paths = []
    for path in root.rglob("*_ros_log.jsonl"):
        lowered_parts = [part.lower() for part in path.parts]
        if any(
            part.endswith("_results")
            or part in {"prediction_plots", "models", "bend_dt_results"}
            for part in lowered_parts
        ):
            continue
        paths.append(path.resolve())

    return sorted(paths)


def infer_protocol_from_path(path: Path) -> str:
    for parent in path.parents:
        name = parent.name.strip().lower().replace("_", " ")
        if name.startswith("dof "):
            return parent.name
    return "unknown"


def paired_video_file(ros_log: Path) -> Path | None:
    name = ros_log.name
    if not name.endswith("_ros_log.jsonl"):
        return None
    video = ros_log.with_name(
        name[: -len("_ros_log.jsonl")] + "_webcam_angles.jsonl"
    )
    return video if video.exists() else None


# ============================================================================
# Import existing Motor-DT training implementation
# ============================================================================


def import_training_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            "Motor training script not found: "
            f"{script_path}\n"
            "Set motor.training_script in dt_prediction_performance.yaml."
        )

    spec = importlib.util.spec_from_file_location(
        "sata_train_motor_digital_twin",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "load_segments_from_files",
        "build_dataset",
        "stack_samples",
        "make_encoder_models",
        "make_current_models",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"{script_path} is missing required function(s): {missing}"
        )
    return module


# ============================================================================
# Motor Digital Twin: true leave-one-run-out
# ============================================================================


def prepare_motor_samples(
    training_module,
    file_paths: list[Path],
    coupling_mode: str,
    input_source: str,
) -> dict[int, list[Any]]:
    """Load and feature-build the dataset once; split folds later by source_file."""
    if len(file_paths) < 2:
        raise RuntimeError(
            f"Need at least two runs for LORO, found {len(file_paths)}."
        )

    segments = training_module.load_segments_from_files(
        file_paths=file_paths,
        coupling_mode=coupling_mode,
        input_source=input_source,
        conversion_parameters=None,
        controller_starting_positions=None,
    )

    # build_dataset expects these metadata fields. They do not affect the
    # numerical features; LORO splitting below uses metadata['source_file'].
    for segment in segments:
        segment["dataset_split"] = "train"
        if hasattr(training_module, "get_split_group"):
            segment["split_group"] = training_module.get_split_group(
                segment,
                coupling_mode,
                input_source,
            )
        else:
            segment["split_group"] = infer_protocol_from_path(
                Path(segment["source_file"])
            )

    return training_module.build_dataset(
        segments=segments,
        coupling_mode=coupling_mode,
        input_source=input_source,
    )


def split_samples_by_source(
    samples: list[Any],
    held_out_file: Path,
) -> tuple[list[Any], list[Any]]:
    held_out_resolved = held_out_file.resolve()
    train_samples = []
    test_samples = []

    for sample in samples:
        metadata = sample[4]
        source = Path(metadata["source_file"]).resolve()
        if source == held_out_resolved:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples


def evaluate_motor_loro_dataset(
    training_module,
    configuration: str,
    dataset_name: str,
    file_paths: list[Path],
    coupling_mode: str,
    input_source: str,
    model_family: str,
    variant: str = "",
    instrument_config: str = "",
) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print(f"Motor LORO: {dataset_name}")
    print(f"Runs: {len(file_paths)}")
    print(f"Fixed model family: {model_family}")
    print("=" * 80)

    samples_by_motor = prepare_motor_samples(
        training_module,
        file_paths,
        coupling_mode,
        input_source,
    )

    results: list[dict[str, Any]] = []

    for fold_index, held_out_file in enumerate(file_paths, 1):
        print(
            f"[{dataset_name}] fold {fold_index}/{len(file_paths)}: "
            f"{held_out_file.name}"
        )

        for motor_index, samples in sorted(samples_by_motor.items()):
            train_samples, test_samples = split_samples_by_source(
                samples,
                held_out_file,
            )
            if not test_samples:
                continue
            if not train_samples:
                raise RuntimeError(
                    f"No training samples for M{motor_index} in fold "
                    f"{held_out_file.name}"
                )

            (
                X_encoder_train,
                X_current_train,
                y_encoder_train,
                y_current_train,
            ) = training_module.stack_samples(train_samples)

            (
                X_encoder_test,
                X_current_test,
                y_encoder_test,
                y_current_test,
            ) = training_module.stack_samples(test_samples)

            encoder_models = training_module.make_encoder_models()
            current_models = training_module.make_current_models()

            if model_family not in encoder_models or model_family not in current_models:
                raise RuntimeError(
                    f"Model family '{model_family}' not available in current "
                    "train_motor_digital_twin.py"
                )

            encoder_model = encoder_models[model_family]
            current_model = current_models[model_family]

            encoder_model.fit(X_encoder_train, y_encoder_train)
            current_model.fit(X_current_train, y_current_train)

            encoder_pred = encoder_model.predict(X_encoder_test)
            current_pred = current_model.predict(X_current_test)

            common = {
                "configuration": configuration,
                "dataset": dataset_name,
                "variant": variant,
                "instrument_config": instrument_config,
                "run_id": held_out_file.stem.replace("_ros_log", ""),
                "protocol": infer_protocol_from_path(held_out_file),
                "motor": int(motor_index),
                "model_family": model_family,
                "held_out_file": str(held_out_file),
                "n_training_runs": len(file_paths) - 1,
            }

            results.append({
                **common,
                "output": "encoder_position",
                "unit": "pulses",
                **calculate_metrics(y_encoder_test, encoder_pred),
            })

            results.append({
                **common,
                "output": "motor_current",
                "unit": "mA",
                **calculate_metrics(y_current_test, current_pred),
            })

    return results


def aggregate_motor_per_run(per_motor: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in per_motor:
        key = (
            row["configuration"],
            row["dataset"],
            row["variant"],
            row["instrument_config"],
            row["run_id"],
            row["protocol"],
            row["output"],
            row["unit"],
        )
        grouped[key].append(row)

    output = []
    for key, rows in sorted(grouped.items()):
        (
            configuration,
            dataset,
            variant,
            instrument_config,
            run_id,
            protocol,
            response,
            unit,
        ) = key

        result = {
            "configuration": configuration,
            "dataset": dataset,
            "variant": variant,
            "instrument_config": instrument_config,
            "run_id": run_id,
            "protocol": protocol,
            "output": response,
            "unit": unit,
            "n_motors": len(rows),
            "n_samples_total": int(sum(row["n_samples"] for row in rows)),
        }
        for metric in ("mae", "rmse", "bias", "max_abs_error"):
            result[metric] = float(np.mean([row[metric] for row in rows]))
        output.append(result)

    return output


def summarize_motor(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate all held-out runs into the three thesis configuration rows.

    Gearbox variants are trained separately but combined only at this final
    run-metric aggregation stage.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["configuration"], row["output"], row["unit"])].append(row)

    summaries = []
    for (configuration, response, unit), rows in sorted(grouped.items()):
        result = {
            "configuration": configuration,
            "output": response,
            "unit": unit,
            "n_runs": len(rows),
        }
        for metric in ("mae", "rmse", "bias", "max_abs_error"):
            mean, sd, _ = sample_mean_sd(row[metric] for row in rows)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        summaries.append(result)
    return summaries


# ============================================================================
# Instrument Digital Twin: DOF2 LORO from synchronized dataset
# ============================================================================


def evaluate_dof2_loro(
    dataset_path: Path,
    model_family: str = "random_forest",
) -> list[dict[str, Any]]:
    if model_family != "random_forest":
        raise ValueError(
            "This thesis evaluator currently fixes DOF2 hybrid validation to "
            "random_forest."
        )

    if not dataset_path.exists():
        raise FileNotFoundError(f"DOF2 dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required = {"trial_id", "prediction_error", "dt_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"DOF2 dataset is missing required columns: {sorted(missing)}"
        )

    reference_col = (
        "video_angle_filtered"
        if "video_angle_filtered" in df.columns
        else "video_angle"
    )

    feature_cols = [
        f"gearbox_{index}"
        for index in range(6)
        if f"gearbox_{index}" in df.columns
    ]
    if len(feature_cols) != 6:
        raise RuntimeError(
            f"Expected gearbox_0..gearbox_5, found {feature_cols}"
        )

    work = df[
        ["trial_id", reference_col, "dt_prediction", "prediction_error", *feature_cols]
    ].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    trial_ids = sorted(work["trial_id"].astype(str).unique())
    if len(trial_ids) < 2:
        raise RuntimeError("At least two DOF2 trials are required for LORO.")

    results = []
    print("\n" + "=" * 80)
    print(f"Instrument DOF2 LORO: {len(trial_ids)} trials")
    print("Fixed hybrid model family: random_forest")
    print("=" * 80)

    for fold_index, trial_id in enumerate(trial_ids, 1):
        train = work[work["trial_id"].astype(str) != trial_id]
        test = work[work["trial_id"].astype(str) == trial_id]

        X_train = train[feature_cols]
        y_train = train["prediction_error"].to_numpy(dtype=float)
        X_test = test[feature_cols]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=5,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        residual_pred_deg = model.predict(X_test)

        measurement_deg = test[reference_col].to_numpy(dtype=float)
        physics_deg = test["dt_prediction"].to_numpy(dtype=float)
        hybrid_deg = physics_deg + residual_pred_deg

        # Same relative-zero convention as Instrument DT plotting/diagnostics
        hybrid_deg = hybrid_deg - hybrid_deg[0]

        measurement_rad = np.deg2rad(measurement_deg)
        physics_rad = np.deg2rad(physics_deg)
        hybrid_rad = np.deg2rad(hybrid_deg)

        print(
            f"[DOF2] fold {fold_index}/{len(trial_ids)}: {trial_id} "
            f"({len(test)} samples)"
        )

        base = {
            "dof": 2,
            "run_id": trial_id,
            "unit": "rad",
            "n_training_runs": len(trial_ids) - 1,
            "source": str(dataset_path),
        }

        results.append({
            **base,
            "model": "physics",
            **calculate_metrics(measurement_rad, physics_rad),
        })
        results.append({
            **base,
            "model": "hybrid",
            **calculate_metrics(measurement_rad, hybrid_rad),
        })

    return results


# ============================================================================
# Instrument Digital Twin: DOF4 direct physics-based validation
# ============================================================================


def load_dof4_prediction(ros_log: Path, prediction_index: int) -> tuple[np.ndarray, np.ndarray, float]:
    rows = load_jsonl(ros_log)
    times = []
    ros_times = []
    values = []

    for row in rows:
        t = safe_float(row.get("time"))
        ros_t = safe_float(row.get("ros_timestamp"))
        prediction = vector_value(row, "predicted_instrument_angles", prediction_index)
        if t is None or ros_t is None or prediction is None:
            continue
        times.append(t)
        ros_times.append(ros_t)
        values.append(prediction)

    if len(times) < 2:
        raise RuntimeError(f"No usable DOF4 prediction in {ros_log}")

    t = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    ros_t0 = float(ros_times[0])

    t = t - t[0]
    values = values - values[0]

    # Remove duplicate timestamps for interpolation.
    unique_t, unique_idx = np.unique(t, return_index=True)
    values = values[unique_idx]
    order = np.argsort(unique_t)
    return unique_t[order], values[order], ros_t0


def load_dof4_video(video_path: Path, ros_t0: float) -> tuple[np.ndarray, np.ndarray]:
    rows = load_jsonl(video_path)
    times = []
    values_deg = []

    for row in rows:
        active_dof = row.get("active_dof")
        if active_dof is not None:
            try:
                if int(active_dof) != 4:
                    continue
            except (TypeError, ValueError):
                continue

        ros_time = safe_float(row.get("ros_time_s"))
        value = safe_float(row.get("measured_angle_between_jaws_filtered"))
        if ros_time is None or value is None:
            continue
        times.append(ros_time - ros_t0)
        values_deg.append(value)

    if len(times) < 2:
        raise RuntimeError(
            "No filtered 'measured_angle_between_jaws_filtered' samples in "
            f"{video_path}. Re-run the current video detector for this run if needed."
        )

    t = np.asarray(times, dtype=float)
    values_rad = np.deg2rad(np.asarray(values_deg, dtype=float))
    order = np.argsort(t)
    return t[order], values_rad[order]


def evaluate_dof4_runs(data_dir: Path, prediction_index: int) -> list[dict[str, Any]]:
    ros_logs = discover_ros_logs(data_dir)
    pairs = [(log, paired_video_file(log)) for log in ros_logs]
    pairs = [(log, video) for log, video in pairs if video is not None]

    if not pairs:
        raise RuntimeError(f"No DOF4 ROS/video pairs found in {data_dir}")

    results = []
    print("\n" + "=" * 80)
    print(f"Instrument DOF4 direct validation: {len(pairs)} runs")
    print("=" * 80)

    for index, (ros_log, video_file) in enumerate(pairs, 1):
        try:
            pred_t, pred_rad, ros_t0 = load_dof4_prediction(
                ros_log,
                prediction_index,
            )
            video_t, measurement_rad = load_dof4_video(video_file, ros_t0)

            valid = (
                np.isfinite(video_t)
                & np.isfinite(measurement_rad)
                & (video_t >= pred_t[0])
                & (video_t <= pred_t[-1])
            )
            video_t = video_t[valid]
            measurement_rad = measurement_rad[valid]
            if len(video_t) < 2:
                raise RuntimeError("No overlapping synchronized samples.")

            predicted_rad = np.interp(video_t, pred_t, pred_rad)
            metrics = calculate_metrics(measurement_rad, predicted_rad)

            run_id = ros_log.stem.replace("_ros_log", "")
            print(f"[DOF4] run {index}/{len(pairs)}: {run_id}")

            results.append({
                "dof": 4,
                "run_id": run_id,
                "model": "physics",
                "unit": "rad",
                "n_training_runs": 0,
                **metrics,
                "source": str(ros_log),
                "video_file": str(video_file),
            })
        except RuntimeError as exc:
            print(f"WARNING: skipping {ros_log.name}: {exc}")

    return results


def summarize_instrument(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dof"]), row["model"], row["unit"])].append(row)

    summaries = []
    for (dof, model, unit), group in sorted(grouped.items()):
        result = {
            "dof": dof,
            "model": model,
            "unit": unit,
            "n_runs": len(group),
        }
        for metric in ("mae", "rmse", "bias", "max_abs_error"):
            mean, sd, _ = sample_mean_sd(row[metric] for row in group)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        summaries.append(result)
    return summaries


# ============================================================================
# LaTeX
# ============================================================================


def generate_latex(
    motor_summary: list[dict[str, Any]],
    instrument_summary: list[dict[str, Any]],
    motor_decimals: int,
    instrument_decimals: int,
) -> str:
    motor_lookup = {
        (row["configuration"], row["output"]): row
        for row in motor_summary
    }
    instrument_lookup = {
        (int(row["dof"]), row["model"]): row
        for row in instrument_summary
    }

    lines = [
        "% Automatically generated by dt_prediction_performance.py",
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Motor Digital Twin prediction performance obtained using leave-one-run-out cross-validation. Values are reported as mean $\\pm$ standard deviation across held-out runs.}",
        "    \\label{tab:motor_dt_performance}",
        "    \\small",
        "    \\setlength{\\tabcolsep}{6pt}",
        "    \\begin{tabular}{llccc}",
        "        \\toprule",
        "        \\textbf{Configuration} & \\textbf{Output} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{Max. absolute error} \\\\",
        "        \\midrule",
    ]

    order = [
        ("motor_only", "Motor only"),
        ("gearbox_only", "Motor--gearbox"),
        ("full_setup", "Full system"),
    ]
    for config_index, (key, label) in enumerate(order):
        for output, output_label in [
            ("encoder_position", "Encoder position [pulses]"),
            ("motor_current", "Motor current [mA]"),
        ]:
            row = motor_lookup.get((key, output))
            if row is None:
                mae = rmse = maximum = "-- $\\pm$ --"
            else:
                mae = latex_mean_sd(row["mae_mean"], row["mae_sd"], motor_decimals)
                rmse = latex_mean_sd(row["rmse_mean"], row["rmse_sd"], motor_decimals)
                maximum = latex_mean_sd(
                    row["max_abs_error_mean"],
                    row["max_abs_error_sd"],
                    motor_decimals,
                )
            lines.append(
                f"        {label} & {output_label} & {mae} & {rmse} & {maximum} \\\\"
            )
        if config_index < len(order) - 1:
            lines.append("        \\midrule")

    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Instrument Digital Twin prediction performance relative to the video-based reference measurements. DOF2 hybrid results were obtained using leave-one-run-out cross-validation. Values are reported as mean $\\pm$ standard deviation across validation runs.}",
        "    \\label{tab:instrument_dt_performance}",
        "    \\small",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{tabular}{llcccc}",
        "        \\toprule",
        "        \\textbf{DOF} & \\textbf{Model} & \\textbf{MAE [rad]} & \\textbf{RMSE [rad]} & \\textbf{Bias [rad]} & \\textbf{Max. abs. error [rad]} \\\\",
        "        \\midrule",
    ])

    for dof, model, dof_label, model_label in [
        (2, "physics", "DOF2", "Physics-based"),
        (2, "hybrid", "DOF2", "Hybrid"),
        (4, "physics", "DOF4", "Physics-based"),
    ]:
        row = instrument_lookup.get((dof, model))
        if row is None:
            mae = rmse = bias = maximum = "-- $\\pm$ --"
        else:
            mae = latex_mean_sd(row["mae_mean"], row["mae_sd"], instrument_decimals)
            rmse = latex_mean_sd(row["rmse_mean"], row["rmse_sd"], instrument_decimals)
            bias = latex_mean_sd(row["bias_mean"], row["bias_sd"], instrument_decimals)
            maximum = latex_mean_sd(
                row["max_abs_error_mean"],
                row["max_abs_error_sd"],
                instrument_decimals,
            )
        lines.append(
            f"        {dof_label} & {model_label} & {mae} & {rmse} & {bias} & {maximum} \\\\"
        )

    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


# ============================================================================
# Pipeline
# ============================================================================


def run_pipeline(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config_dir = config_path.parent
    training_root = expand_path(config["training_data_root"], config_dir)
    output_dir = expand_path(
        config.get("output_dir", "~/ros2_ws/test_data/dt_prediction_performance_results"),
        config_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    motor_per_motor: list[dict[str, Any]] = []
    motor_cfg = config.get("motor", {}) or {}

    if motor_cfg.get("enabled", True):
        training_script = expand_path(
            motor_cfg.get(
                "training_script",
                "~/ros2_ws/src/adlap_tool_control/scripts/train_motor_digital_twin.py",
            ),
            config_dir,
        )
        training_module = import_training_module(training_script)
        model_family = str(motor_cfg.get("model_family", "random_forest"))
        configurations = motor_cfg.get("configurations", {}) or {}

        # Motor only -----------------------------------------------------------
        entry = configurations.get("motor_only")
        if entry and entry.get("enabled", True):
            data_dir = expand_path(entry["data_dir"], training_root)
            logs = discover_ros_logs(data_dir)
            motor_per_motor.extend(evaluate_motor_loro_dataset(
                training_module=training_module,
                configuration="motor_only",
                dataset_name="motor_only",
                file_paths=logs,
                coupling_mode="motor_only",
                input_source=str(entry.get("input_source", "targets")),
                model_family=model_family,
            ))

        # Motor + gearbox: train/evaluate each hardware variant separately ----
        entry = configurations.get("gearbox_only")
        if entry and entry.get("enabled", True):
            base_dir = expand_path(entry["data_dir"], training_root)
            variants = entry.get("variants", ["gearbox_1", "gearbox_2"])
            for variant in variants:
                variant_dir = base_dir / str(variant)
                logs = discover_ros_logs(variant_dir)
                motor_per_motor.extend(evaluate_motor_loro_dataset(
                    training_module=training_module,
                    configuration="gearbox_only",
                    dataset_name=f"gearbox_only/{variant}",
                    file_paths=logs,
                    coupling_mode="gearbox_only",
                    input_source=str(entry.get("input_source", "targets")),
                    model_family=model_family,
                    variant=str(variant),
                ))

        # Full setup -----------------------------------------------------------
        entry = configurations.get("full_setup")
        if entry and entry.get("enabled", True):
            data_dir = expand_path(entry["data_dir"], training_root)
            logs = discover_ros_logs(data_dir)
            instrument_config = str(entry.get("instrument_config", "gripper"))
            motor_per_motor.extend(evaluate_motor_loro_dataset(
                training_module=training_module,
                configuration="full_setup",
                dataset_name=f"full_setup/{instrument_config}",
                file_paths=logs,
                coupling_mode="full_setup",
                input_source=str(entry.get("input_source", "targets")),
                model_family=model_family,
                instrument_config=instrument_config,
            ))

    motor_per_run = aggregate_motor_per_run(motor_per_motor) if motor_per_motor else []
    motor_summary = summarize_motor(motor_per_run) if motor_per_run else []

    instrument_rows: list[dict[str, Any]] = []
    instrument_cfg = config.get("instrument", {}) or {}
    if instrument_cfg.get("enabled", True):
        dof2_cfg = instrument_cfg.get("dof2", {}) or {}
        if dof2_cfg.get("enabled", True):
            dataset_path = expand_path(dof2_cfg["dataset"], training_root)
            instrument_rows.extend(evaluate_dof2_loro(
                dataset_path,
                model_family=str(dof2_cfg.get("model_family", "random_forest")),
            ))

        dof4_cfg = instrument_cfg.get("dof4", {}) or {}
        if dof4_cfg.get("enabled", True):
            data_dir = expand_path(dof4_cfg["data_dir"], training_root)
            instrument_rows.extend(evaluate_dof4_runs(
                data_dir,
                prediction_index=int(dof4_cfg.get("prediction_index", 3)),
            ))

    instrument_summary = summarize_instrument(instrument_rows) if instrument_rows else []

    # Write results -----------------------------------------------------------
    if motor_per_motor:
        write_csv(output_dir / "motor_per_motor_metrics.csv", motor_per_motor, [
            "configuration", "dataset", "variant", "instrument_config",
            "run_id", "protocol", "motor", "model_family", "output", "unit",
            "n_training_runs", "n_samples", "mae", "rmse", "bias",
            "max_abs_error", "held_out_file",
        ])

    if motor_per_run:
        write_csv(output_dir / "motor_per_run_metrics.csv", motor_per_run, [
            "configuration", "dataset", "variant", "instrument_config",
            "run_id", "protocol", "output", "unit", "n_motors",
            "n_samples_total", "mae", "rmse", "bias", "max_abs_error",
        ])

    if motor_summary:
        write_csv(output_dir / "motor_summary.csv", motor_summary, [
            "configuration", "output", "unit", "n_runs",
            "mae_mean", "mae_sd", "rmse_mean", "rmse_sd",
            "bias_mean", "bias_sd", "max_abs_error_mean", "max_abs_error_sd",
        ])

    if instrument_rows:
        fields = [
            "dof", "run_id", "model", "unit", "n_training_runs", "n_samples",
            "mae", "rmse", "bias", "max_abs_error", "source", "video_file",
        ]
        write_csv(output_dir / "instrument_per_run_metrics.csv", instrument_rows, fields)

    if instrument_summary:
        write_csv(output_dir / "instrument_summary.csv", instrument_summary, [
            "dof", "model", "unit", "n_runs",
            "mae_mean", "mae_sd", "rmse_mean", "rmse_sd",
            "bias_mean", "bias_sd", "max_abs_error_mean", "max_abs_error_sd",
        ])

    latex_cfg = config.get("latex", {}) or {}
    latex = generate_latex(
        motor_summary,
        instrument_summary,
        int(latex_cfg.get("motor_decimals", 2)),
        int(latex_cfg.get("instrument_decimals", 3)),
    )
    (output_dir / "dt_performance_tables.tex").write_text(latex, encoding="utf-8")

    metadata = {
        "script_version": SCRIPT_VERSION,
        "error_definition": "measurement - prediction",
        "motor_cross_validation": "leave-one-run-out at complete ROS-log level",
        "motor_model_family_fixed_before_evaluation": str(
            motor_cfg.get("model_family", "random_forest")
        ),
        "dof2_cross_validation": "leave-one-trial-out using trial_id",
        "dof4_cross_validation": "not applicable; physics-based direct validation",
        "training_data_root": str(training_root),
        "output_dir": str(output_dir),
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Digital Twin prediction-performance evaluation complete")
    print(f"Output directory: {output_dir}")
    print("Generated:")
    for name in [
        "motor_per_motor_metrics.csv",
        "motor_per_run_metrics.csv",
        "motor_summary.csv",
        "instrument_per_run_metrics.csv",
        "instrument_summary.csv",
        "dt_performance_tables.tex",
        "evaluation_metadata.json",
    ]:
        path = output_dir / name
        if path.exists():
            print(f"  - {path}")


def write_example_config(path: Path) -> None:
    text = '''# dt_prediction_performance.yaml
training_data_root: "~/ros2_ws/test_data/automated_trials/trainings data V6"
output_dir: "~/ros2_ws/test_data/dt_prediction_performance_results"

motor:
  enabled: true
  training_script: "~/ros2_ws/src/adlap_tool_control/scripts/train_motor_digital_twin.py"
  model_family: random_forest

  configurations:
    motor_only:
      enabled: true
      data_dir: "motors_only_dof_pattern"
      input_source: targets

    gearbox_only:
      enabled: true
      data_dir: "motors_gearbox"
      input_source: targets
      variants:
        - gearbox_1
        - gearbox_2

    full_setup:
      enabled: true
      data_dir: "full_setup/gripper"
      input_source: targets
      instrument_config: gripper

instrument:
  enabled: true

  dof2:
    enabled: true
    dataset: "full_setup/gripper/DOF 2/bend_dt_results/bend_training_dataset.csv"
    model_family: random_forest

  dof4:
    enabled: true
    data_dir: "full_setup/gripper/DOF 4"
    prediction_index: 3

latex:
  motor_decimals: 2
  instrument_decimals: 3
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dt_prediction_performance.yaml",
        help="Evaluation YAML configuration.",
    )
    parser.add_argument(
        "--write-example-config",
        default=None,
        help="Write the recommended YAML and exit.",
    )
    args = parser.parse_args()

    if args.write_example_config:
        path = Path(args.write_example_config).expanduser().resolve()
        write_example_config(path)
        print(f"Wrote example configuration: {path}")
        return

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    run_pipeline(config_path)


if __name__ == "__main__":
    main()