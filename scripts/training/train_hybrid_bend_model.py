#!/usr/bin/env python3

"""
Train and evaluate the Hybrid Bend Digital Twin for SATA DOF2.

The script:
- loads synchronized ROS and camera measurements,
- combines the physical Digital Twin prediction with measured gearbox states,
- trains regression models to predict the remaining bend-angle error,
- evaluates models using a trial-based hold-out split,
- saves the trained models, metrics, datasets, and diagnostic plots.

The Hybrid Bend model corrects the original Digital Twin prediction according
to:

    hybrid prediction = DT prediction + ML-predicted residual
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ament_index_python.packages import get_package_share_directory
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT_TRAINING_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "training_config.yaml"
)
DEFAULT_DOF_PATTERN_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "dof_pattern_params.yaml"
)

# -------------------------------------------------------------------------
# Configuration and file loading
# -------------------------------------------------------------------------
def load_training_config(config_path: Path):
    """
    Load the training configuration and resolve the training-data root directory.
    """
    config_path = Path(config_path).expanduser()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_data_root = Path(
        config["test_data_root"]
    ).expanduser()

    training_data_root = Path(
        config["training_data_root"]
    ).expanduser()

    if not training_data_root.is_absolute():
        training_data_root = (
            test_data_root / training_data_root
        )

    return config, training_data_root


def resolve_training_path(training_data_root: Path, path_value: str) -> Path:
    """
    Resolve a configured training path relative to the training-data root.
    """
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return training_data_root / path
    

def load_dof2_pattern_config(config_path: Path):
    """
    Load the DOF2 pattern configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["dof_pattern_runner_node"]["ros__parameters"]["dof2"]


def load_jsonl(path: Path):
    """
    Load a JSONL file into a list of dictionaries.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# -------------------------------------------------------------------------
# Dataset construction
# -------------------------------------------------------------------------
def expand_list_column(df, column, prefix, expected_len):
    if column not in df.columns:
        for i in range(expected_len):
            df[f"{prefix}{i}"] = np.nan
        return df

    values = df[column].apply(lambda x: x if isinstance(x, list) else None)

    for i in range(expected_len):
        df[f"{prefix}{i}"] = values.apply(
            lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan
        )

    return df

def build_ros_dataframe(ros_log: Path):
    """
    Convert a ROS trial log into a time-sorted dataframe with expanded signals.
    """
    df = pd.DataFrame(load_jsonl(ros_log))

    df = expand_list_column(df, "measured_motor_positions", "motor_pos_", 4)
    df = expand_list_column(df, "commanded_motor_positions", "motor_cmd_", 4)
    df = expand_list_column(df, "commanded_instrument_angles", "cmd_instr_", 4)
    if "measured_instrument_angles" not in df.columns and "current_instrument_angles" in df.columns:
        df["measured_instrument_angles"] = df["current_instrument_angles"]
    df = expand_list_column(df, "measured_instrument_angles", "meas_instr_", 4)
    df = expand_list_column(df, "gearbox_state", "gearbox_", 6)
    df = expand_list_column(df, "predicted_instrument_angles", "pred_instr_", 4)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["merge_time"] = pd.to_numeric(df["ros_timestamp"], errors="coerce")

    df = df.dropna(subset=["time", "merge_time"])
    return df.sort_values("merge_time").reset_index(drop=True)

def build_video_dataframe(video_angles: Path):
    """
    Convert a video angles log into a time-sorted dataframe.
    """
    df = pd.DataFrame(load_jsonl(video_angles))

    if "ros_time_s" not in df.columns:
        raise RuntimeError(f"No ros_time_s column in {video_angles}")

    df["merge_time"] = pd.to_numeric(df["ros_time_s"], errors="coerce")

    if "synced_time_s" in df.columns:
        df["video_synced_time_s"] = pd.to_numeric(df["synced_time_s"], errors="coerce")

    if "video_time_s" in df.columns:
        df["video_time_s"] = pd.to_numeric(df["video_time_s"], errors="coerce")

    df = df.dropna(subset=["merge_time"])
    return df.sort_values("merge_time").reset_index(drop=True)

def choose_video_target_column(video_df):
    candidates = [
        "measured_angle_red_shaft_zeroed_filtered",
        "measured_angle_red_shaft_zeroed",
        "measured_angle_red_shaft",
        "shaft_angle_deg",
        "red_marker_angle_deg",
        "red_shaft_angle_zeroed",
        "red_shaft_video_angle_zeroed",
        "red_shaft_angle",
        "red_shaft_video_angle",
        "shaft_angle_zeroed",
        "shaft_angle",
    ]

    for col in candidates:
        if col in video_df.columns:
            return col

    raise RuntimeError(f"No DOF2 video angle column found. Columns: {list(video_df.columns)}")

def find_ros_video_pairs(search_dir: Path):
    """
    Find matching pairs of ROS log and video angles files in the given directory.
    """
    pairs = []

    # New automated-trial naming:
    # auto_dof2_..._ros_log.jsonl
    # auto_dof2_..._webcam_angles.jsonl
    for ros_log in sorted(search_dir.glob("*_ros_log.jsonl")):
        video_log = ros_log.with_name(
            ros_log.name.replace("_ros_log.jsonl", "_webcam_angles.jsonl")
        )

        if video_log.exists():
            pairs.append((ros_log, video_log))

    # Old naming:
    # trial_01.jsonl + *_webcam_angles.jsonl
    for ros_log in sorted(search_dir.glob("trial_*.jsonl")):
        video_logs = sorted(search_dir.glob("*webcam_angles.jsonl"))

        if video_logs:
            pairs.append((ros_log, video_logs[0]))

    # Remove duplicate pairs
    unique_pairs = []
    seen = set()

    for ros_log, video_log in pairs:
        key = (str(ros_log), str(video_log))

        if key not in seen:
            unique_pairs.append((ros_log, video_log))
            seen.add(key)

    return unique_pairs

def build_trial_table_from_files(ros_log: Path, video_log: Path, trial_id: str):
    ros_df = build_ros_dataframe(ros_log)
    video_df = build_video_dataframe(video_log)

    target_col = choose_video_target_column(video_df)
    dt_pred_col = "pred_instr_1"

    ros_time_zero = ros_df["merge_time"].iloc[0] - ros_df["time"].iloc[0]

    merged = pd.merge_asof(
        video_df[["merge_time", target_col]].rename(
            columns={target_col: "video_angle"}
        ),
        ros_df,
        on="merge_time",
        direction="nearest",
        tolerance=0.05,
    )

    # Use continuous ROS-relative time for plotting.
    merged["time"] = merged["merge_time"] - ros_time_zero
    merged = merged.dropna(
        subset=["video_angle", dt_pred_col]
    ).reset_index(drop=True)

    if len(merged) == 0:
        print(f"Loaded {trial_id}: 0 rows after time matching")
        return None

    merged["video_angle_raw"] = merged["video_angle"]
    merged["video_angle_filtered"] = merged["video_angle"]

    merged["trial_id"] = trial_id

    # pred_instr_1 is in rad, video_angle is in deg
    merged["dt_prediction"] = np.rad2deg(merged[dt_pred_col])
    merged["dt_prediction"] = (
        merged["dt_prediction"] - merged["dt_prediction"].iloc[0]
    )

    merged["prediction_error"] = (
        merged["video_angle"] - merged["dt_prediction"]
    )

    feature_cols = select_feature_columns(merged)

    required_cols = [
        "video_angle",
        "dt_prediction",
        "prediction_error",
    ] + feature_cols

    merged = merged.dropna(subset=required_cols).reset_index(drop=True)

    print(f"Loaded {trial_id}: {len(merged)} rows")
    print(f"  ROS log:   {ros_log.name}")
    print(f"  Video log: {video_log.name}")
    print(f"  Video target: {target_col}")
    print(f"  DT prediction: {dt_pred_col} converted rad -> deg")

    return merged


def build_trial_table(trial_dir: Path):
    pairs = find_ros_video_pairs(trial_dir)

    if not pairs:
        print(
            f"Skipping {trial_dir}: missing *_ros_log.jsonl + "
            f"*_webcam_angles.jsonl pair"
        )
        return None

    trial_tables = []

    for ros_log, video_log in pairs:
        trial_id = ros_log.name.replace("_ros_log.jsonl", "")

        table = build_trial_table_from_files(
            ros_log=ros_log,
            video_log=video_log,
            trial_id=trial_id,
        )

        if table is not None and len(table) > 0:
            trial_tables.append(table)

    if not trial_tables:
        return None

    return pd.concat(trial_tables, ignore_index=True)

def build_dataset(data_dir: Path):
    """
    Combine all usable trials in the given directory into a single dataframe for 
    training and evaluation.
    """
    trial_tables = []

    # First: support flat folder structure, where the files are directly
    # inside DOF 2.
    root_table = build_trial_table(data_dir)

    if root_table is not None and len(root_table) > 0:
        trial_tables.append(root_table)

    # Second: support old structure, where each trial has its own subfolder.
    for trial_dir in sorted(data_dir.iterdir()):
        if trial_dir.is_dir():
            table = build_trial_table(trial_dir)

            if table is not None and len(table) > 0:
                trial_tables.append(table)

    if not trial_tables:
        raise RuntimeError(f"No usable trials found in {data_dir}")

    return pd.concat(trial_tables, ignore_index=True)
    
def select_feature_columns(df):
    """
    Return the available numeric gearbox-state features used by the ML model.
    """
    allowed_columns = [
        "gearbox_0",
        "gearbox_1",
        "gearbox_2",
        "gearbox_3",
        "gearbox_4",
        "gearbox_5",
    ]

    feature_cols = []

    for col in allowed_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols

# -------------------------------------------------------------------------
# Sequence reconstruction and plotting
# -------------------------------------------------------------------------
def plot_test_scatter(test_df, output_path: Path):
    test_df = test_df.sort_values("time").reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.scatter(test_df["time"], test_df["video_angle"], s=8, label="video measurement")
    plt.scatter(test_df["time"], test_df["dt_prediction"], s=8, label="old DT prediction")
    plt.scatter(test_df["time"], test_df["hybrid_prediction"], s=8, label="hybrid DT prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("Bend angle [deg]")
    plt.title("Hybrid bend prediction evaluation: test samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_full_timeline(df, output_path: Path):
    df = df.sort_values("time").reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df["time"], df["video_angle"], label="video measurement")
    plt.plot(df["time"], df["dt_prediction"], label="old DT prediction")
    plt.plot(df["time"], df["hybrid_prediction_all"], label="hybrid DT prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("Bend angle [deg]")
    plt.title("Hybrid bend prediction: full timeline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def get_dof2_sequence_blocks(dof2_config):
    base_frequency = float(dof2_config["frequency"])
    frequency_factors = dof2_config["sequence_frequency_factors"]
    range_factors = dof2_config["sequence_range_factors"]
    modes = dof2_config["sequence_modes"]
    cycles = dof2_config["sequence_cycles"]

    sequence_pause_duration = float(
        dof2_config["sequence_pause_duration"]
    )
    between_frequency_pause = float(
        dof2_config["sequence_between_frequency_pause"]
    )
    between_range_pause = float(
        dof2_config["sequence_between_range_pause"]
    )
    final_pause_duration = float(
        dof2_config["sequence_final_pause_duration"]
    )


    blocks = []
    t = 0.0

    for range_index, range_factor in enumerate(range_factors):
        for frequency_index, frequency_factor in enumerate(frequency_factors):
            frequency = base_frequency * frequency_factor

            for mode_index, (mode, n_cycles) in enumerate(zip(modes, cycles)):
                duration = float(n_cycles) / frequency

                blocks.append(
                    {
                        "start": t,
                        "end": t + duration,
                        "range_factor": range_factor,
                        "frequency": frequency,
                        "mode": mode,
                    }
                )

                t += duration

                is_last_mode = mode_index == len(modes) - 1
                if not is_last_mode:
                    t += sequence_pause_duration

            is_last_frequency = frequency_index == len(frequency_factors) - 1
            if not is_last_frequency:
                t += between_frequency_pause

        is_last_range = range_index == len(range_factors) - 1
        if not is_last_range:
            t += between_range_pause

    t += final_pause_duration
    return blocks

def plot_sequence_blocks(df, output_dir: Path, dof2_config):
    df = df.sort_values("time").reset_index(drop=True)
    block_dir = output_dir / "sequence_block_plots"
    block_dir.mkdir(parents=True, exist_ok=True)

    blocks = get_dof2_sequence_blocks(dof2_config)

    for i, block in enumerate(blocks):
        margin = 0.5

        block_df = df[
            (df["time"] >= block["start"] - margin)
            & (df["time"] <= block["end"] + margin)
        ].copy()

        if len(block_df) < 5:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(block_df["time"], block_df["video_angle"], label="video measurement")
        plt.plot(block_df["time"], block_df["dt_prediction"], label="old DT prediction")
        plt.plot(
            block_df["time"],
            block_df["hybrid_prediction_all"],
            label="hybrid DT prediction",
        )

        title = (
            f"DOF2 block {i:02d} | "
            f"range={block['range_factor']} | "
            f"freq={block['frequency']} Hz | "
            f"{block['mode']}"
        )

        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Bend angle [deg]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = (
            f"block_{i:02d}_"
            f"range_{str(block['range_factor']).replace('.', 'p')}_"
            f"freq_{str(block['frequency']).replace('.', 'p')}_"
            f"{block['mode']}.png"
        )

        plt.savefig(block_dir / filename, dpi=200)
        plt.close()

# -------------------------------------------------------------------------
# Model definitions and training
# -------------------------------------------------------------------------        
def make_models():
    """
    Create the candidate regression models evaluated for Hybrid Bend correction.
    """
    models = {
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
            n_estimators=200,
            random_state=42,
            min_samples_leaf=5,
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

    return models

def train_single_model(
    model_name,
    model,
    df,
    X,
    X_train,
    X_test,
    y_train,
    test_index,
    feature_cols,
    output_dir: Path,
    dof2_config
):
    """
    Train and evaluate one residual-correction model on the held-out trial.
    """
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(X_train, y_train)

    result_df = df.copy()

    result_df.loc[test_index, "ml_error_prediction"] = model.predict(X_test)
    result_df.loc[test_index, "hybrid_prediction"] = (
        result_df.loc[test_index, "dt_prediction"]
        + result_df.loc[test_index, "ml_error_prediction"]
    )

    old_mae = mean_absolute_error(
        result_df.loc[test_index, "video_angle"],
        result_df.loc[test_index, "dt_prediction"],
    )

    hybrid_mae = mean_absolute_error(
        result_df.loc[test_index, "video_angle"],
        result_df.loc[test_index, "hybrid_prediction"],
    )

    old_rmse = mean_squared_error(
        result_df.loc[test_index, "video_angle"],
        result_df.loc[test_index, "dt_prediction"],
        squared=False,
    )

    hybrid_rmse = mean_squared_error(
        result_df.loc[test_index, "video_angle"],
        result_df.loc[test_index, "hybrid_prediction"],
        squared=False,
    )
    test_video_angle = (
        result_df.loc[test_index, "video_angle"]
        .to_numpy(dtype=float)
    )

    finite_test_video_angle = test_video_angle[
        np.isfinite(test_video_angle)
    ]

    if finite_test_video_angle.size >= 2:
        video_angle_min_deg = float(
            np.min(finite_test_video_angle)
        )
        video_angle_max_deg = float(
            np.max(finite_test_video_angle)
        )
        video_angle_range_deg = (
            video_angle_max_deg - video_angle_min_deg
        )
    else:
        video_angle_min_deg = None
        video_angle_max_deg = None
        video_angle_range_deg = None

    # Predict the complete dataset for plotting and analysis
    result_df["ml_error_prediction_all"] = model.predict(X)
    result_df["hybrid_prediction_all"] = (
        result_df["dt_prediction"]
        + result_df["ml_error_prediction_all"]
    )

    metrics = {
        "model": model_name,
        "old_dt_mae_deg": old_mae,
        "hybrid_dt_mae_deg": hybrid_mae,
        "old_dt_rmse_deg": old_rmse,
        "hybrid_dt_rmse_deg": hybrid_rmse,
        "mae_improvement_deg": old_mae - hybrid_mae,
        "rmse_improvement_deg": old_rmse - hybrid_rmse,
        "mae_improvement_percent": 100.0 * (old_mae - hybrid_mae) / old_mae,
        "rmse_improvement_percent": 100.0 * (old_rmse - hybrid_rmse) / old_rmse,
        "n_samples_total": len(result_df),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": len(feature_cols),
    }

    normalized_metrics = dict(metrics)

    normalized_metrics["video_angle_range"] = {
        "minimum_deg": video_angle_min_deg,
        "maximum_deg": video_angle_max_deg,
        "range_deg": video_angle_range_deg,
        "samples": int(finite_test_video_angle.size),
    }

    normalized_metrics["normalization_source"] = (
        "measured_video_angle_range_of_held_out_test_trial"
    )

    if (
        video_angle_range_deg is not None
        and np.isfinite(video_angle_range_deg)
        and video_angle_range_deg > 0.0
    ):
        scale = 100.0 / video_angle_range_deg

        normalized_metrics[
            "old_dt_nmae_percent_of_video_range"
        ] = float(old_mae * scale)

        normalized_metrics[
            "hybrid_dt_nmae_percent_of_video_range"
        ] = float(hybrid_mae * scale)

        normalized_metrics[
            "old_dt_nrmse_percent_of_video_range"
        ] = float(old_rmse * scale)

        normalized_metrics[
            "hybrid_dt_nrmse_percent_of_video_range"
        ] = float(hybrid_rmse * scale)

    else:
        normalized_metrics[
            "old_dt_nmae_percent_of_video_range"
        ] = None
        normalized_metrics[
            "hybrid_dt_nmae_percent_of_video_range"
        ] = None
        normalized_metrics[
            "old_dt_nrmse_percent_of_video_range"
        ] = None
        normalized_metrics[
            "hybrid_dt_nrmse_percent_of_video_range"
        ] = None

    print(f"\nModel: {model_name}")
    print(f"  Old DT MAE:     {old_mae:.3f} deg")
    print(f"  Hybrid DT MAE:  {hybrid_mae:.3f} deg")
    print(f"  Old DT RMSE:    {old_rmse:.3f} deg")
    print(f"  Hybrid DT RMSE: {hybrid_rmse:.3f} deg")

    model_path = model_output_dir / f"{model_name}_bend_hybrid_model.joblib"
    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
            "feature_columns": feature_cols,
            "metrics": metrics,
            "target_normalized_metrics": normalized_metrics,
        },
        model_path,
    )

    result_csv = model_output_dir / f"{model_name}_bend_training_dataset_with_predictions.csv"
    result_df.to_csv(result_csv, index=False)

    plot_path = model_output_dir / f"{model_name}_test_scatter.png"
    plot_test_scatter(result_df.loc[test_index], plot_path)

    full_plot_path = model_output_dir / f"{model_name}_full_timeline.png"
    plot_full_timeline(result_df, full_plot_path)

    plot_sequence_blocks(result_df, model_output_dir, dof2_config)

    return metrics, normalized_metrics

# -------------------------------------------------------------------------
# Reporting
# -------------------------------------------------------------------------
def write_model_comparison_txt(metrics_df, output_path: Path):
    metrics_df = metrics_df.sort_values("hybrid_dt_mae_deg").reset_index(drop=True)

    with open(output_path, "w") as f:
        f.write("Hybrid bend prediction model comparison\n")
        f.write("======================================\n\n")

        f.write("Evaluation method\n")
        f.write("-----------------\n")
        f.write("Target: prediction_error = video_angle - old_DT_prediction\n")
        f.write("Hybrid prediction: old_DT_prediction + ML_predicted_error\n")
        f.write("Evaluation split: trial-based hold-out\n")
        f.write("Complete trials were separated between training and testing.\n\n")

        f.write("Model ranking by Hybrid MAE\n")
        f.write("---------------------------\n")

        for i, row in metrics_df.iterrows():
            f.write(f"{i + 1}. {row['model']}\n")
            f.write(f"   Old DT MAE:          {row['old_dt_mae_deg']:.3f} deg\n")
            f.write(f"   Hybrid DT MAE:       {row['hybrid_dt_mae_deg']:.3f} deg\n")
            f.write(f"   MAE improvement:     {row['mae_improvement_deg']:.3f} deg ")
            f.write(f"({row['mae_improvement_percent']:.1f}%)\n")
            f.write(f"   Old DT RMSE:         {row['old_dt_rmse_deg']:.3f} deg\n")
            f.write(f"   Hybrid DT RMSE:      {row['hybrid_dt_rmse_deg']:.3f} deg\n")
            f.write(f"   RMSE improvement:    {row['rmse_improvement_deg']:.3f} deg ")
            f.write(f"({row['rmse_improvement_percent']:.1f}%)\n")
            f.write(f"   Total samples:       {int(row['n_samples_total'])}\n")
            f.write(f"   Training samples:    {int(row['n_samples_train'])}\n")
            f.write(f"   Test samples:        {int(row['n_samples_test'])}\n")
            f.write(f"   Number of features:  {int(row['n_features'])}\n\n")

        best_mae = metrics_df.iloc[0]
        best_rmse = metrics_df.sort_values("hybrid_dt_rmse_deg").iloc[0]

        f.write("Best models\n")
        f.write("-----------\n")
        f.write(
            f"Best MAE model:  {best_mae['model']} "
            f"({best_mae['hybrid_dt_mae_deg']:.3f} deg)\n"
        )
        f.write(
            f"Best RMSE model: {best_rmse['model']} "
            f"({best_rmse['hybrid_dt_rmse_deg']:.3f} deg)\n"
        )

def train_and_evaluate(df, output_dir: Path, dof2_config):
    """
    Train all candidate models and evaluate them on a held-out trial.
    """
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise RuntimeError(
            "No valid gearbox-state features found in the training dataset."
        )

    print("\nModel input features:")
    for col in feature_cols:
        print(f"  {col}")
    print(f"Total features: {len(feature_cols)}")
    X = df[feature_cols]
    y = df["prediction_error"]

    # Trial-based hold-out split
    trial_ids = sorted(df["trial_id"].unique())

    if len(trial_ids) < 2:
        raise RuntimeError(
            "At least two trials are required for trial-based train/test splitting."
        )

    # Use the latest trial as the fixed held-out test trial.
    test_trial = trial_ids[-1]
    train_trials = trial_ids[:-1]
    train_mask = df["trial_id"].isin(train_trials)
    test_mask = df["trial_id"] == test_trial

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    test_index = X_test.index

    print("\nTrial-based train/test split:")
    print(f"  Training trials ({len(train_trials)}):")
    for trial in train_trials:
        print(f"    {trial}")
    print(f"  Test trial:")
    print(f"    {test_trial}")

    # Train candidate models and evaluate on the held-out trial
    models = make_models()
    all_metrics = []
    all_normalized_metrics = []

    for model_name, model in models.items():
        metrics, normalized_metrics = train_single_model(
            model_name=model_name,
            model=model,
            df=df,
            X=X,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            test_index=test_index,
            feature_cols=feature_cols,
            output_dir=output_dir,
            dof2_config=dof2_config
        )
        all_metrics.append(metrics)
        all_normalized_metrics.append(normalized_metrics)

    # Select the best model
    selected_model = min(
        all_metrics,
        key=lambda row: row["hybrid_dt_mae_deg"],
    )["model"]

    # Save metrics and normalized metrics to JSON files
    metrics_output = {
        "evaluation_split": "trial_based_hold_out",
        "training_trials": train_trials,
        "test_trial": test_trial,
        "selection_metric": "hybrid_dt_mae_deg",
        "selected_model": selected_model,
        "models": {
            row["model"]: row
            for row in all_metrics
        },
    }

    normalized_metrics_output = {
        "evaluation_split": "trial_based_hold_out",
        "training_trials": train_trials,
        "test_trial": test_trial,
        "selection_metric": "hybrid_dt_mae_deg",
        "selected_model": selected_model,
        "target_range_normalization": {
            "formula": (
                "100 * angle_error / "
                "(max_video_angle - min_video_angle)"
            ),
            "denominator_scope": (
                "measured video-angle range of "
                "the complete held-out test trial"
            ),
            "unit": "degrees",
            # "model_retrained": False,
            # "predictions_recomputed": False,
        },
        "models": {
            row["model"]: row
            for row in all_normalized_metrics
        },
    }

    metrics_path = (
        output_dir
        / "hybrid_bend_metrics.json"
    )

    normalized_metrics_path = (
        output_dir
        / "hybrid_bend_metrics_target_normalized.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)

    with open(normalized_metrics_path, "w") as f:
        json.dump(normalized_metrics_output, f, indent=2)

    print(f"\nSaved Hybrid Bend metrics: {metrics_path}")
    print(
        "Saved target-normalized Hybrid Bend metrics: "
        f"{normalized_metrics_path}"
    )

    # Save model comparison metrics to CSV and TXT
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("hybrid_dt_mae_deg").reset_index(drop=True)

    comparison_csv = output_dir / "model_comparison_metrics.csv"
    metrics_df.to_csv(comparison_csv, index=False)
    comparison_txt = output_dir / "model_comparison_metrics.txt"
    write_model_comparison_txt(metrics_df, comparison_txt)

    print("\nModel comparison:")
    print(metrics_df[[
        "model",
        "old_dt_mae_deg",
        "hybrid_dt_mae_deg",
        "old_dt_rmse_deg",
        "hybrid_dt_rmse_deg",
        "mae_improvement_deg",
        "rmse_improvement_deg",
    ]])

    print(f"\nSaved model comparison: {comparison_csv}")
    print(f"Saved model comparison: {comparison_txt}")
    

def main():
    """
    Build the Hybrid Bend dataset and train the candidate models.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="Training configuration YAML.",
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional override for the Hybrid Bend training data directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the Hybrid Bend result directory.",
    )
    
    args = parser.parse_args()

    config, training_data_root = load_training_config(
        args.training_config
    )

    hybrid_config = config["hybrid_bend_dt"]

    dof2_config = load_dof2_pattern_config(
        DEFAULT_DOF_PATTERN_CONFIG_PATH
    )

    data_dir = (
        Path(args.data_dir).expanduser()
        if args.data_dir is not None
        else resolve_training_path(
            training_data_root,
            hybrid_config["input_dir"],
        )
    )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else resolve_training_path(
            training_data_root,
            hybrid_config["output_dir"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(data_dir)

    dataset_path = output_dir / "bend_training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    print(f"\nSaved dataset: {dataset_path}")
    print(f"Total rows: {len(dataset)}")

    train_and_evaluate(dataset, output_dir, dof2_config)


if __name__ == "__main__":
    main()