#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT_TRAINING_DATA_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "automated_trials"
    / "trainings data"
)

DOF_TRAINING_FOLDERS = {
    1: "DOF 1",
    2: "DOF 2",
    3: "DOF 3",
    4: "DOF 4",
}

MOTOR_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}


def make_current_models():
    return {
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "polynomial_ridge": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0),
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


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


def safe_gradient(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return np.zeros_like(y)

    dt = np.diff(t)
    positive_dt = dt[dt > 0.0]
    median_dt = np.nanmedian(positive_dt) if len(positive_dt) > 0 else 0.01

    t_safe = t.copy()
    for i in range(1, len(t_safe)):
        if t_safe[i] <= t_safe[i - 1]:
            t_safe[i] = t_safe[i - 1] + median_dt

    return np.gradient(y, t_safe)


def time_since_signal_change(t, signal, threshold=1e-5):
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


def wrap_to_pi(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def get_time_value(data):
    if data.get("time") is not None:
        return data.get("time")
    if data.get("timestamp") is not None:
        return data.get("timestamp")
    if data.get("ros_timestamp") is not None:
        return data.get("ros_timestamp")
    return None


def as_float_array(value, expected_length):
    if value is None or len(value) < expected_length:
        return None

    array = np.array(value[:expected_length], dtype=float)
    if not np.all(np.isfinite(array)):
        return None

    return array


def find_tool_ros_logs(input_dir):
    input_dir = Path(input_dir).expanduser()

    if not input_dir.exists():
        raise RuntimeError(f"Input folder does not exist: {input_dir}")

    candidates = []

    for path in input_dir.rglob("*.jsonl"):
        name = path.name.lower()

        if path.stat().st_size == 0:
            continue

        if "webcam" in name or "angle" in name or "debug" in name:
            continue

        if "motor_dt_replay" in name:
            continue

        if "instrument_current_dt_replay" in name:
            continue

        if "metadata" in name:
            continue

        # Prefer automated trial ROS logs, but keep the script permissive.
        if "ros_log" not in name and "trial" not in name and "auto" not in name:
            continue

        candidates.append(path)

    candidates = sorted(candidates)

    if not candidates:
        raise RuntimeError(
            f"No usable tool ROS .jsonl logs found in:\n{input_dir}\n\n"
            "Expected files such as auto_dof3_..._ros_log.jsonl."
        )

    print("\nAutomatically selected training files:")
    for path in candidates:
        print(f"  - {path}")

    return candidates


def load_tool_log(file_path, active_dof_index, current_window=5, require_gearbox=False):
    file_path = Path(file_path).expanduser()

    times = []
    commanded = []
    positions = []
    currents = []
    gearbox_states = []
    predicted_angles = []
    task_labels = []

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            t_value = get_time_value(data)
            cmd = as_float_array(data.get("commanded_instrument_angles"), 4)
            pos = as_float_array(data.get("measured_motor_positions"), 4)
            cur = as_float_array(data.get("measured_currents"), 4)

            if t_value is None or cmd is None or pos is None or cur is None:
                continue

            gb = as_float_array(data.get("gearbox_state"), 6)
            pred = as_float_array(data.get("predicted_instrument_angles"), 8)

            if require_gearbox and gb is None:
                continue

            if gb is None:
                gb = np.zeros(6, dtype=float)
                gearbox_available = 0.0
            else:
                gearbox_available = 1.0

            if pred is None:
                pred = np.zeros(8, dtype=float)
                prediction_available = 0.0
            else:
                prediction_available = 1.0

            times.append(float(t_value))
            commanded.append(cmd)
            positions.append(pos)
            currents.append(cur)
            gearbox_states.append(np.r_[gb, gearbox_available])
            predicted_angles.append(np.r_[pred, prediction_available])
            task_labels.append(data.get("task_label") or data.get("task") or "unlabeled")

    if len(times) < 20:
        raise RuntimeError(f"Not enough usable samples in {file_path}")

    t = np.array(times, dtype=float)
    t = t - t[0]

    commanded = np.vstack(commanded)
    positions = np.vstack(positions)
    currents = np.vstack(currents)
    gearbox_states = np.vstack(gearbox_states)
    predicted_angles = np.vstack(predicted_angles)

    # Relative motor positions per run: absolute encoder offsets are not useful for this model.
    positions = positions - positions[0, :]
    # Use only the active commanded instrument DOF as model input.
    # For example, with --dof 3 this uses commanded_instrument_angles[2].
    target = commanded[:, active_dof_index]

    abs_target = np.abs(target)

    # Change in commanded target between consecutive samples.
    instant_target_delta = np.zeros_like(target)
    instant_target_delta[1:] = target[1:] - target[:-1]

    # Keep the last non-zero target change.
    # This gives the model information about the most recent movement direction/step.
    target_delta = np.zeros_like(target)
    last_delta = 0.0
    command_change_threshold = 1e-5

    for sample_index in range(len(target)):
        if abs(instant_target_delta[sample_index]) > command_change_threshold:
            last_delta = instant_target_delta[sample_index]

        target_delta[sample_index] = last_delta

        abs_target_delta = np.abs(target_delta)
        target_direction = np.sign(target_delta)

        time_after_change = time_since_signal_change(
            t,
            target,
            threshold=command_change_threshold,
        )

        # Moving-state based only on the command.
        movement_memory_s = 1.2
        target_based_is_moving = (
            (np.abs(target_delta) > command_change_threshold)
            & (time_after_change < movement_memory_s)
        ).astype(float)

        # Use the same moving signal for the zone metrics.
        is_moving = target_based_is_moving

        feature_columns = []
        feature_names = []

        def add_feature(name, values):
            feature_names.append(name)
            feature_columns.append(np.asarray(values, dtype=float))

        add_feature("target", target)
        add_feature("abs_target", abs_target)
        add_feature("target_delta", target_delta)
        add_feature("abs_target_delta", abs_target_delta)
        add_feature("target_direction", target_direction)
        add_feature("time_after_change", time_after_change)
        add_feature("target_based_is_moving", target_based_is_moving)
    
    X = np.column_stack(feature_columns)

    y_by_motor = np.column_stack([
        causal_moving_average(currents[:, motor_index], window_size=current_window)
        for motor_index in range(4)
    ])

    valid = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y_by_motor), axis=1)

    X = X[valid]
    y_by_motor = y_by_motor[valid]
    t = t[valid]
    is_moving = is_moving[valid]
    currents = currents[valid]
    commanded = commanded[valid]

    if len(t) < 20:
        raise RuntimeError(f"Not enough finite samples in {file_path}")

    return {
        "source_file": str(file_path),
        "time": t,
        "X": X,
        "y_by_motor": y_by_motor,
        "raw_currents": currents,
        "commanded": commanded,
        "is_moving": is_moving,
        "feature_names": feature_names,
        "n_samples": int(len(t)),
        "task_label_first": task_labels[0] if task_labels else "unknown",
    }


def load_dataset(file_paths, active_dof_index, current_window=5, require_gearbox=False):
    runs = []

    for file_path in file_paths:
        print(f"\nLoading tool log: {file_path}")
        run = load_tool_log(
            file_path=file_path,
            active_dof_index=active_dof_index,
            current_window=current_window,
            require_gearbox=require_gearbox,
        )
        print(f"  samples: {run['n_samples']}")
        print(f"  task:    {run['task_label_first']}")
        runs.append(run)

    if not runs:
        raise RuntimeError("No usable runs loaded.")

    feature_names = runs[0]["feature_names"]
    for run in runs[1:]:
        if run["feature_names"] != feature_names:
            raise RuntimeError("Feature-name mismatch between runs.")

    return runs, feature_names


def split_dataset(runs, test_fraction=0.25, random_state=42):
    X_all = np.vstack([run["X"] for run in runs])
    y_all = np.vstack([run["y_by_motor"] for run in runs])
    moving_all = np.concatenate([run["is_moving"] for run in runs])
    source_all = np.concatenate([
        np.full(run["n_samples"], run_index, dtype=int)
        for run_index, run in enumerate(runs)
    ])
    time_all = np.concatenate([run["time"] for run in runs])

    n_samples = len(X_all)
    indices = np.arange(n_samples)

    if len(runs) >= 2:
        # Keep the newest/last selected file as test run.
        test_source = len(runs) - 1
        test_mask = source_all == test_source
        train_mask = ~test_mask

        train_indices = indices[train_mask]
        test_indices = indices[test_mask]
    else:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_state,
            shuffle=True,
        )

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError("Train/test split failed: empty train or test set.")

    return {
        "X_train": X_all[train_indices],
        "X_test": X_all[test_indices],
        "y_train": y_all[train_indices],
        "y_test": y_all[test_indices],
        "moving_train": moving_all[train_indices],
        "moving_test": moving_all[test_indices],
        "time_test": time_all[test_indices],
        "test_indices": test_indices,
        "source_test": source_all[test_indices],
    }


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
    }


def compute_current_zone_metrics(y_true, y_pred, moving_mask):
    metrics = {}
    moving_mask = moving_mask.astype(bool)
    baseline_mask = ~moving_mask

    if np.any(baseline_mask):
        metrics["baseline_current_mA"] = compute_metrics(
            y_true[baseline_mask],
            y_pred[baseline_mask],
        )

    if np.any(moving_mask):
        metrics["moving_current_mA"] = compute_metrics(
            y_true[moving_mask],
            y_pred[moving_mask],
        )

    peak_threshold = np.nanpercentile(y_true, 95)
    peak_mask = y_true >= peak_threshold

    if np.any(peak_mask):
        metrics["peak_current_mA"] = compute_metrics(
            y_true[peak_mask],
            y_pred[peak_mask],
        )

    metrics["peak_threshold_mA"] = float(peak_threshold)

    return metrics


def model_selection_score(model_result):
    zone_metrics = model_result["zone_metrics"]

    if "moving_current_mA" in zone_metrics:
        return zone_metrics["moving_current_mA"]["mae"]

    return model_result["overall_metrics"]["mae"]


def flatten_metrics_row(motor_index, model_name, overall_metrics, zone_metrics):
    row = {
        "motor_index": int(motor_index),
        "model": model_name,
        "overall_mae_mA": safe_float(overall_metrics.get("mae")),
        "overall_rmse_mA": safe_float(overall_metrics.get("rmse")),
        "overall_max_abs_error_mA": safe_float(overall_metrics.get("max_abs_error")),
        "peak_threshold_mA": safe_float(zone_metrics.get("peak_threshold_mA")),
    }

    zones = {
        "baseline": "baseline_current_mA",
        "moving": "moving_current_mA",
        "peak": "peak_current_mA",
    }

    for prefix, zone_name in zones.items():
        zone = zone_metrics.get(zone_name, {})
        row[f"{prefix}_mae_mA"] = safe_float(zone.get("mae"))
        row[f"{prefix}_rmse_mA"] = safe_float(zone.get("rmse"))
        row[f"{prefix}_max_abs_error_mA"] = safe_float(zone.get("max_abs_error"))

    return row


def write_model_comparison(rows, output_dir):
    output_dir = Path(output_dir)
    json_path = output_dir / "instrument_current_model_comparison_metrics.json"
    csv_path = output_dir / "instrument_current_model_comparison_metrics.csv"
    txt_path = output_dir / "instrument_current_model_comparison_metrics.txt"

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "motor_index",
        "model",
        "overall_mae_mA",
        "overall_rmse_mA",
        "overall_max_abs_error_mA",
        "baseline_mae_mA",
        "baseline_rmse_mA",
        "baseline_max_abs_error_mA",
        "moving_mae_mA",
        "moving_rmse_mA",
        "moving_max_abs_error_mA",
        "peak_mae_mA",
        "peak_rmse_mA",
        "peak_max_abs_error_mA",
        "peak_threshold_mA",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(txt_path, "w") as f:
        f.write("Instrument current model comparison\n")
        f.write("===================================\n\n")
        f.write("Models are ranked per motor by moving-current MAE.\n\n")

        for motor_index in sorted(set(row["motor_index"] for row in rows)):
            f.write(f"Motor {motor_index}\n")
            f.write("-" * 20 + "\n")

            motor_rows = [row for row in rows if row["motor_index"] == motor_index]
            motor_rows = sorted(
                motor_rows,
                key=lambda row: (
                    row["moving_mae_mA"] is None,
                    row["moving_mae_mA"] if row["moving_mae_mA"] is not None else float("inf"),
                ),
            )

            for rank, row in enumerate(motor_rows, start=1):
                f.write(f"{rank}. {row['model']}\n")
                f.write(f"   Overall MAE:       {row['overall_mae_mA']:.3f} mA\n")
                f.write(f"   Overall RMSE:      {row['overall_rmse_mA']:.3f} mA\n")
                f.write(f"   Moving MAE:        {row['moving_mae_mA']:.3f} mA\n")
                f.write(f"   Moving RMSE:       {row['moving_rmse_mA']:.3f} mA\n")
                f.write(f"   Peak MAE:          {row['peak_mae_mA']:.3f} mA\n\n")

            f.write("\n")

    print(f"Saved comparison: {json_path}")
    print(f"Saved comparison: {csv_path}")
    print(f"Saved comparison: {txt_path}")


def train_current_models(split, feature_names, active_dof, output_dir, training_files, current_window):
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train_all = split["y_train"]
    y_test_all = split["y_test"]
    moving_test = split["moving_test"]

    metrics = {
        "active_dof": int(active_dof),
        "feature_names": feature_names,
        "training_files": [str(path) for path in training_files],
        "current_filter_window_samples": int(current_window),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "motors": {},
    }

    all_model_package = {
        "active_dof": int(active_dof),
        "feature_names": feature_names,
        "current_filter_window_samples": int(current_window),
        "models_by_motor": {},
        "selected_model_names_by_motor": {},
    }

    comparison_rows = []
    trained_models = {}

    for motor_index in range(4):
        y_train = y_train_all[:, motor_index]
        y_test = y_test_all[:, motor_index]

        candidate_results = {}

        for model_name, model in make_current_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            overall_metrics = compute_metrics(y_test, y_pred)
            zone_metrics = compute_current_zone_metrics(
                y_true=y_test,
                y_pred=y_pred,
                moving_mask=moving_test,
            )

            candidate_results[model_name] = {
                "model": model,
                "prediction": y_pred,
                "overall_metrics": overall_metrics,
                "zone_metrics": zone_metrics,
            }

            comparison_rows.append(
                flatten_metrics_row(
                    motor_index=motor_index,
                    model_name=model_name,
                    overall_metrics=overall_metrics,
                    zone_metrics=zone_metrics,
                )
            )

        best_model_name = min(
            candidate_results,
            key=lambda name: model_selection_score(candidate_results[name]),
        )

        best_result = candidate_results[best_model_name]
        best_model = best_result["model"]

        motor_package = {
            "motor_index": int(motor_index),
            "active_dof": int(active_dof),
            "model": best_model,
            "selected_model": best_model_name,
            "feature_names": feature_names,
            "current_filter_window_samples": int(current_window),
        }

        model_path = models_dir / f"instrument_current_dt_dof{active_dof}_m{motor_index}.pkl"
        joblib.dump(motor_package, model_path)

        trained_models[motor_index] = best_model
        all_model_package["models_by_motor"][motor_index] = best_model
        all_model_package["selected_model_names_by_motor"][motor_index] = best_model_name

        metrics["motors"][f"motor_{motor_index}"] = {
            "selected_model": best_model_name,
            "model_path": str(model_path),
            "overall_metrics_mA": best_result["overall_metrics"],
            "zone_metrics_mA": best_result["zone_metrics"],
            "model_comparison": {
                name: {
                    "overall_metrics_mA": result["overall_metrics"],
                    "zone_metrics_mA": result["zone_metrics"],
                }
                for name, result in candidate_results.items()
            },
        }

        print(f"\nMotor {motor_index}")
        print(f"  selected model: {best_model_name}")
        print(f"  overall MAE:    {best_result['overall_metrics']['mae']:.3f} mA")
        if "moving_current_mA" in best_result["zone_metrics"]:
            print(f"  moving MAE:     {best_result['zone_metrics']['moving_current_mA']['mae']:.3f} mA")
        print(f"  saved model:    {model_path}")

    all_model_path = models_dir / f"instrument_current_dt_dof{active_dof}_all_motors.pkl"
    joblib.dump(all_model_package, all_model_path)
    metrics["all_motors_model_path"] = str(all_model_path)

    metrics_path = output_dir / "instrument_current_dt_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved all-motor model package: {all_model_path}")
    print(f"Saved metrics: {metrics_path}")

    write_model_comparison(comparison_rows, output_dir)

    return trained_models, metrics


def plot_predictions(split, trained_models, output_dir):
    plot_dir = Path(output_dir) / "prediction_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    X_test = split["X_test"]
    y_test_all = split["y_test"]

    # Use sample index to avoid strange lines when random split samples are not time-contiguous.
    x_axis = np.arange(len(X_test))

    for motor_index, model in trained_models.items():
        y_test = y_test_all[:, motor_index]
        y_pred = model.predict(X_test)
        residual = y_test - y_pred

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"Instrument current prediction - motor {motor_index}", fontsize=14)

        axes[0].plot(
            x_axis,
            y_test,
            linewidth=1.3,
            color=MOTOR_COLORS[motor_index],
            label="filtered measured current",
        )
        axes[0].plot(
            x_axis,
            y_pred,
            linewidth=1.3,
            color="tab:purple",
            label="predicted current",
        )
        axes[0].set_ylabel("Current [mA]")
        axes[0].grid(True)
        axes[0].legend(fontsize=8)

        axes[1].plot(
            x_axis,
            residual,
            linewidth=1.1,
            color="black",
            label="residual = measured - predicted",
        )
        axes[1].axhline(0.0, linewidth=1.0, color="gray")
        axes[1].set_ylabel("Residual [mA]")
        axes[1].set_xlabel("Test sample index")
        axes[1].grid(True)
        axes[1].legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plot_path = plot_dir / f"instrument_current_prediction_motor_{motor_index}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        print(f"Saved prediction plot: {plot_path}")

def train_all_dofs_from_training_root(
    training_root,
    output_root,
    current_window,
    test_fraction,
    require_gearbox,
):
    training_root = Path(training_root).expanduser()

    if output_root is None:
        output_root = training_root / "instrument_current_dt_results"
    else:
        output_root = Path(output_root).expanduser()

    output_root.mkdir(parents=True, exist_ok=True)

    trained_dofs = []
    skipped_dofs = []

    for dof, folder_name in DOF_TRAINING_FOLDERS.items():
        dof_input_dir = training_root / folder_name
        dof_output_dir = output_root / f"dof{dof}"

        print("")
        print("=" * 80)
        print(f"Instrument Current DT training: DOF{dof}")
        print(f"Input folder:  {dof_input_dir}")
        print(f"Output folder: {dof_output_dir}")
        print("=" * 80)

        if not dof_input_dir.exists():
            print(f"Skipping DOF{dof}: folder does not exist.")
            skipped_dofs.append(dof)
            continue

        try:
            file_paths = find_tool_ros_logs(dof_input_dir)
        except RuntimeError as exc:
            print(f"Skipping DOF{dof}: {exc}")
            skipped_dofs.append(dof)
            continue

        train_instrument_current_model(
            file_paths=file_paths,
            output_dir=dof_output_dir,
            active_dof=dof,
            current_window=current_window,
            test_fraction=test_fraction,
            require_gearbox=require_gearbox,
        )

        trained_dofs.append(dof)

    print("")
    print("=" * 80)
    print("Instrument Current DT multi-DOF training finished")
    print(f"Trained DOFs: {trained_dofs}")
    print(f"Skipped DOFs: {skipped_dofs}")
    print(f"Output root:  {output_root}")
    print("=" * 80)

def train_instrument_current_model(file_paths, output_dir, active_dof, current_window, test_fraction, require_gearbox):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    active_dof_index = active_dof - 1

    runs, feature_names = load_dataset(
        file_paths=file_paths,
        active_dof_index=active_dof_index,
        current_window=current_window,
        require_gearbox=require_gearbox,
    )

    total_samples = sum(run["n_samples"] for run in runs)
    print(f"\nTotal runs:    {len(runs)}")
    print(f"Total samples: {total_samples}")
    print(f"Features:      {len(feature_names)}")

    split = split_dataset(
        runs=runs,
        test_fraction=test_fraction,
    )

    print(f"Train samples: {len(split['X_train'])}")
    print(f"Test samples:  {len(split['X_test'])}")

    trained_models, _ = train_current_models(
        split=split,
        feature_names=feature_names,
        active_dof=active_dof,
        output_dir=output_dir,
        training_files=file_paths,
        current_window=current_window,
    )

    plot_predictions(
        split=split,
        trained_models=trained_models,
        output_dir=output_dir,
    )

    print("\nInstrument current DT training completed.")
    print(f"Output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train an instrument current Digital Twin from tool/gearbox/instrument JSONL logs."
    )

    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Path to a specific tool ROS log. Can be used multiple times.",
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_TRAINING_DATA_DIR),
        help="Folder where tool ROS logs are searched recursively when --file is not used.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder where models, metrics and plots are saved. If omitted, a results folder is created inside input-dir.",
    )

    parser.add_argument(
        "--dof",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help=(
            "Optional: train only one DOF. "
            "If omitted, all DOF folders in the training root are trained."
        ),
    )

    parser.add_argument(
        "--current-window",
        type=int,
        default=5,
        help="Causal moving average window for measured current target.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Test fraction used only when training on one file.",
    )

    parser.add_argument(
        "--require-gearbox",
        action="store_true",
        help="Skip samples without gearbox_state. Use only after gearbox/instrument DT logging is fixed.",
    )
    
    args = parser.parse_args()

    training_root = Path(args.input_dir).expanduser()

    # Case 1: specific files manually provided.
    # Then the DOF must be known, otherwise the script cannot know which command
    # column should be used as target.
    if args.file is not None:
        if args.dof is None:
            raise RuntimeError("--dof is required when using --file.")

        file_paths = [Path(path).expanduser() for path in args.file]

        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser()
        else:
            output_dir = training_root / "instrument_current_dt_results" / f"dof{args.dof}"

        train_instrument_current_model(
            file_paths=file_paths,
            output_dir=output_dir,
            active_dof=args.dof,
            current_window=args.current_window,
            test_fraction=args.test_fraction,
            require_gearbox=args.require_gearbox,
        )

        return

    # Case 2: one specific DOF requested.
    if args.dof is not None:
        dof_input_dir = training_root / DOF_TRAINING_FOLDERS[args.dof]

        file_paths = find_tool_ros_logs(dof_input_dir)

        if args.output_dir is not None:
            output_dir = Path(args.output_dir).expanduser()
        else:
            output_dir = training_root / "instrument_current_dt_results" / f"dof{args.dof}"

        train_instrument_current_model(
            file_paths=file_paths,
            output_dir=output_dir,
            active_dof=args.dof,
            current_window=args.current_window,
            test_fraction=args.test_fraction,
            require_gearbox=args.require_gearbox,
        )

        return

    # Case 3: default behaviour.
    # Train all DOF folders and ignore Motors only.
    train_all_dofs_from_training_root(
        training_root=training_root,
        output_root=args.output_dir,
        current_window=args.current_window,
        test_fraction=args.test_fraction,
        require_gearbox=args.require_gearbox,
    )


if __name__ == "__main__":
    main()
