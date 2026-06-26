#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    df = pd.DataFrame(load_jsonl(ros_log))

    df = expand_list_column(df, "measured_motor_positions", "motor_pos_", 4)
    # df = expand_list_column(df, "measured_currents", "current_", 4)
    df = expand_list_column(df, "commanded_motor_positions", "motor_cmd_", 4)
    df = expand_list_column(df, "commanded_instrument_angles", "cmd_instr_", 4)
    df = expand_list_column(df, "current_instrument_angles", "cur_instr_", 4)
    df = expand_list_column(df, "gearbox_state", "gearbox_", 6)
    df = expand_list_column(df, "predicted_instrument_angles", "pred_instr_", 4)

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["merge_time"] = pd.to_numeric(df["ros_timestamp"], errors="coerce")

    df = df.dropna(subset=["time", "merge_time"])
    return df.sort_values("merge_time").reset_index(drop=True)

def build_video_dataframe(video_angles: Path):
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

def filter_video_angle(df):
    df["video_angle_raw"] = df["video_angle"]

    # Stap 1: median filter tegen losse detectie-outliers
    df["video_angle_filtered"] = (
        df["video_angle_raw"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    # Stap 2: lichte smoothing tegen meetruis
    df["video_angle_filtered"] = (
        df["video_angle_filtered"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    df["video_angle"] = df["video_angle_filtered"]
    return df

def add_history_features(df, feature_columns):
    for col in feature_columns:
        df[f"{col}_prev"] = df[col].shift(1)
        df[f"{col}_delta"] = df[col].diff()
        df[f"{col}_direction"] = np.sign(df[f"{col}_delta"])
    return df


def build_trial_table(trial_dir: Path):
    ros_logs = sorted(trial_dir.glob("trial_*.jsonl"))
    video_logs = sorted(trial_dir.glob("*webcam_angles.jsonl"))

    if not ros_logs or not video_logs:
        print(f"Skipping {trial_dir}: missing trial_*.jsonl or *webcam_angles.jsonl")
        return None

    ros_log = ros_logs[0]
    video_log = video_logs[0]

    ros_df = build_ros_dataframe(ros_log)
    video_df = build_video_dataframe(video_log)

    target_col = choose_video_target_column(video_df)
    dt_pred_col = "pred_instr_1"

    ros_time_zero = ros_df["merge_time"].iloc[0] - ros_df["time"].iloc[0]

    merged = pd.merge_asof(
        video_df[["merge_time", target_col]].rename(columns={target_col: "video_angle"}),
        ros_df,
        on="merge_time",
        direction="nearest",
        tolerance=0.05,
    )

    # Use continuous ROS-relative time for plotting.
    merged["time"] = merged["merge_time"] - ros_time_zero
    merged = merged.dropna(subset=["video_angle", dt_pred_col]).reset_index(drop=True)

    if len(merged) == 0:
        print(f"Loaded {trial_dir.name}: 0 rows after time matching")
        return None

    merged = filter_video_angle(merged)

    merged["trial_id"] = trial_dir.name

    # pred_instr_1 is in rad, video_angle is in deg
    merged["dt_prediction"] = np.rad2deg(merged[dt_pred_col])
    merged["dt_prediction"] = merged["dt_prediction"] - merged["dt_prediction"].iloc[0]
    merged["prediction_error"] = merged["video_angle"] - merged["dt_prediction"]

    # history_cols = [
    #     "motor_pos_0", "motor_pos_1", "motor_pos_2", "motor_pos_3",
    #     "gearbox_0", "gearbox_1", "gearbox_2",
    #     "gearbox_3", "gearbox_4", "gearbox_5",
    #     "dt_prediction",
    # ]

    # history_cols = [col for col in history_cols if col in merged.columns]
    # merged = add_history_features(merged, history_cols)

    feature_cols = select_feature_columns(merged)

    required_cols = [
        "video_angle",
        "dt_prediction",
        "prediction_error",
    ] + feature_cols

    merged = merged.dropna(subset=required_cols).reset_index(drop=True)

    print(f"Loaded {trial_dir.name}: {len(merged)} rows")
    print(f"  ROS log:   {ros_log.name}")
    print(f"  Video log: {video_log.name}")
    print(f"  Video target: {target_col}")
    print(f"  DT prediction: {dt_pred_col} converted rad -> deg")

    return merged

def build_dataset(data_dir: Path):
    trial_tables = []

    for trial_dir in sorted(data_dir.iterdir()):
        if trial_dir.is_dir():
            table = build_trial_table(trial_dir)
            if table is not None and len(table) > 0:
                trial_tables.append(table)

    if not trial_tables:
        raise RuntimeError(f"No usable trials found in {data_dir}")

    return pd.concat(trial_tables, ignore_index=True)

def select_feature_columns(df):
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
    
# def select_feature_columns(df):
#     allowed_prefixes = (
#         "motor_pos_",
#         # "current_",
#         "gearbox_",
#         "dt_prediction",
#     )

#     forbidden_columns = {
#         "time",
#         "video_angle",
#         "prediction_error",
#     }

#     feature_cols = []

#     for col in df.columns:
#         if col in forbidden_columns:
#             continue

#         if any(col.startswith(prefix) for prefix in allowed_prefixes):
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 feature_cols.append(col)

#     return feature_cols

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

def get_dof2_sequence_blocks():
    base_frequency = 0.5
    frequency_factors = [0.5, 1.0, 2.0]
    range_factors = [0.25, 0.5, 1.0]
    modes = ["sinusoid", "triangle"]
    cycles = [3, 3]

    sequence_pause_duration = 5.0
    between_frequency_pause = 5.0
    between_range_pause = 5.0
    final_pause_duration = 5.0

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
def plot_sequence_blocks(df, output_dir: Path):
    df = df.sort_values("time").reset_index(drop=True)
    block_dir = output_dir / "sequence_block_plots"
    block_dir.mkdir(parents=True, exist_ok=True)

    blocks = get_dof2_sequence_blocks()

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
def make_models():
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
    }

    return models

def train_single_model(
    model_name,
    model,
    df,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    test_index,
    feature_cols,
    output_dir: Path,
):
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

    # Predict over volledige dataset voor leesbare tijdlijnplots
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
        },
        model_path,
    )

    result_csv = model_output_dir / f"{model_name}_bend_training_dataset_with_predictions.csv"
    result_df.to_csv(result_csv, index=False)

    metrics_path = model_output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_path = model_output_dir / f"{model_name}_test_scatter.png"
    plot_test_scatter(result_df.loc[test_index], plot_path)

    full_plot_path = model_output_dir / f"{model_name}_full_timeline.png"
    plot_full_timeline(result_df, full_plot_path)

    plot_sequence_blocks(result_df, model_output_dir)

    return metrics

def write_model_comparison_txt(metrics_df, output_path: Path):
    metrics_df = metrics_df.sort_values("hybrid_dt_mae_deg").reset_index(drop=True)

    with open(output_path, "w") as f:
        f.write("Hybrid bend prediction model comparison\n")
        f.write("======================================\n\n")

        f.write("Evaluation method\n")
        f.write("-----------------\n")
        f.write("Target: prediction_error = video_angle - old_DT_prediction\n")
        f.write("Hybrid prediction: old_DT_prediction + ML_predicted_error\n")
        f.write("Evaluation split: random 25% test samples\n\n")

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

def train_and_evaluate(df, output_dir: Path):
    feature_cols = select_feature_columns(df)

    print("\nModel input features:")
    for col in feature_cols:
        print(f"  {col}")
    print(f"Total features: {len(feature_cols)}")

    X = df[feature_cols]
    y = df["prediction_error"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    test_index = X_test.index

    models = make_models()
    all_metrics = []

    for model_name, model in models.items():
        metrics = train_single_model(
            model_name=model_name,
            model=model,
            df=df,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            test_index=test_index,
            feature_cols=feature_cols,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

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
    
def plot_results(test_df, output_path: Path):
    test_df = test_df.sort_values("time").reset_index(drop=True)

    plt.figure(figsize=(14, 6))
    plt.plot(test_df["time"], test_df["video_angle"], label="video measurement")
    plt.plot(test_df["time"], test_df["dt_prediction"], label="old DT prediction")
    plt.plot(test_df["time"], test_df["hybrid_prediction"], label="hybrid DT prediction")
    plt.xlabel("Time [s]")
    plt.ylabel("Bend angle [deg]")
    plt.title("Hybrid bend prediction evaluation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(data_dir)

    dataset_path = output_dir / "bend_training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    print(f"\nSaved dataset: {dataset_path}")
    print(f"Total rows: {len(dataset)}")

    train_and_evaluate(dataset, output_dir)


if __name__ == "__main__":
    main()