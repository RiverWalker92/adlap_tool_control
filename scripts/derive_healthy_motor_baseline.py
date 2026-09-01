#!/usr/bin/env python3
"""
Derive healthy residual bands for motor-only Digital Twin diagnostics.

Scientific approach:
1. Treat each task_label as a repeated measurement, not each sample as an
   independent experiment.
2. Use only task segments in which the evaluated motor is actually commanded.
3. Preserve residual sign (asymmetric bands are allowed).
4. For each repetition, compute the 1st and 99th residual percentiles
for position and the 5th and 95th percentiles for current.
5. Across repetitions, define the final lower/upper band from the 5th
   percentile of repetition-wise lower bounds and the 95th percentile of
   repetition-wise upper bounds.
6. Round limits outward to practical engineering increments.
7. Re-evaluate all healthy repetitions against the final band and report
   outside-band fraction and longest continuous excursion.
8. Perform leave-one-run-out validation to quantify how well the baseline
   generalises to an unseen healthy run.

Residuals:
  position = encoder_deviation
  current  = filtered_current_deviation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POSITION_ROUNDING = 5.0   # encoder pulses
CURRENT_ROUNDING = 5.0    # mA
ACTIVE_TARGET_EPS = 1e-12

POSITION_WITHIN_REP_LOWER_Q = 0.01
POSITION_WITHIN_REP_UPPER_Q = 0.99

CURRENT_WITHIN_REP_LOWER_Q = 0.05
CURRENT_WITHIN_REP_UPPER_Q = 0.95

BETWEEN_REP_LOWER_Q = 0.05
BETWEEN_REP_UPPER_Q = 0.95

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}: invalid JSON at line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def validate(rows: list[dict[str, Any]], path: Path) -> None:
    required = {
        "test_type",
        "task_label",
        "coupling_mode",
        "segment_time",
        "offline_commanded_target",
        "encoder_deviation",
        "filtered_current_deviation",
    }
    missing = sorted(required - rows[0].keys())
    if missing:
        raise ValueError(f"{path.name}: missing fields: {', '.join(missing)}")

    modes = {str(r["coupling_mode"]) for r in rows}
    if modes != {"motor_only"}:
        raise ValueError(f"{path.name}: expected motor_only, found {sorted(modes)}")


def outward_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def outward_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def longest_true_duration(mask: np.ndarray, times: np.ndarray) -> float:
    """Longest consecutive True duration; one sample counts as one sample period."""
    if len(mask) == 0:
        return 0.0

    if len(times) > 1:
        dt = np.diff(times)
        dt = dt[dt > 0]
        sample_period = float(np.median(dt)) if len(dt) else 0.0
    else:
        sample_period = 0.0

    longest = 0.0
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i - 1
            duration = float(times[end] - times[start]) + sample_period
            longest = max(longest, duration)
            start = None

    if start is not None:
        end = len(mask) - 1
        duration = float(times[end] - times[start]) + sample_period
        longest = max(longest, duration)

    return longest


def collect_repetitions(paths: list[Path]) -> list[dict[str, Any]]:
    reps: list[dict[str, Any]] = []

    for path in paths:
        rows = load_jsonl(path)
        validate(rows, path)

        segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            segments[str(row["task_label"])].append(row)

        for task_label, segment in segments.items():
            segment.sort(key=lambda r: float(r["segment_time"]))
            if not segment:
                continue

            test_type = str(segment[0]["test_type"])
            if test_type == "idle_baseline":
                continue

            motor_count = len(segment[0]["offline_commanded_target"])

            for motor in range(motor_count):
                target = np.asarray(
                    [float(r["offline_commanded_target"][motor]) for r in segment],
                    dtype=float,
                )

                if len(target) < 2:
                    continue

                # Only analyse this task for a motor if that motor is actually commanded.
                if not np.any(np.abs(np.diff(target)) > ACTIVE_TARGET_EPS):
                    continue

                times = np.asarray([float(r["segment_time"]) for r in segment], dtype=float)
                pos = np.asarray([float(r["encoder_deviation"][motor]) for r in segment], dtype=float)
                cur = np.asarray([float(r["filtered_current_deviation"][motor]) for r in segment], dtype=float)

                valid_p = np.isfinite(times) & np.isfinite(pos)
                valid_c = np.isfinite(times) & np.isfinite(cur)

                if np.sum(valid_p) < 2 or np.sum(valid_c) < 2:
                    continue

                reps.append(
                    {
                        "run": path.stem.replace("_motor_dt_replay", ""),
                        "source_file": path.name,
                        "task_label": task_label,
                        "test_type": test_type,
                        "motor": motor,
                        "position_times": times[valid_p],
                        "position": pos[valid_p],
                        "current_times": times[valid_c],
                        "current": cur[valid_c],
                    }
                )

    return reps


def repetition_stats(rep: dict[str, Any], signal: str) -> dict[str, Any]:
    x = np.asarray(rep[signal], dtype=float)
    return {
        "run": rep["run"],
        "source_file": rep["source_file"],
        "task_label": rep["task_label"],
        "test_type": rep["test_type"],
        "motor": rep["motor"],
        "signal": signal,
        "n_samples": int(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def derive_band(stats: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    if signal == "position":
        lower_values = np.asarray(
            [s["q01"] for s in stats],
            dtype=float,
        )

        upper_values = np.asarray(
            [s["q99"] for s in stats],
            dtype=float,
        )

    else:
        lower_values = np.asarray(
            [s["q05"] for s in stats],
            dtype=float,
        )

        upper_values = np.asarray(
            [s["q95"] for s in stats],
            dtype=float,
        )

    raw_lower = float(
        np.quantile(
            lower_values,
            BETWEEN_REP_LOWER_Q,
        )
    )

    raw_upper = float(
        np.quantile(
            upper_values,
            BETWEEN_REP_UPPER_Q,
        )
    )

    step = POSITION_ROUNDING if signal == "position" else CURRENT_ROUNDING
    lower = outward_floor(raw_lower, step)
    upper = outward_ceil(raw_upper, step)

    return {
        "n_repetitions": len(stats),
        "raw_lower": raw_lower,
        "raw_upper": raw_upper,
        "lower": lower,
        "upper": upper,
        "rounding_step": step,
        "median_repetition_lower": float(
            np.median(lower_values)
        ),

        "median_repetition_upper": float(
            np.median(upper_values)
        ),
    }


def evaluate_rep(rep: dict[str, Any], signal: str, lower: float, upper: float) -> dict[str, Any]:
    x = np.asarray(rep[signal], dtype=float)
    times = np.asarray(rep[f"{signal}_times"], dtype=float)

    outside = (x < lower) | (x > upper)
    n = len(x)
    n_out = int(np.sum(outside))
    fraction = n_out / n if n else 0.0

    return {
        "run": rep["run"],
        "task_label": rep["task_label"],
        "test_type": rep["test_type"],
        "motor": rep["motor"],
        "signal": signal,
        "n_samples": n,
        "samples_outside": n_out,
        "fraction_outside": fraction,
        "percentage_outside": 100.0 * fraction,
        "longest_outside_s": longest_true_duration(outside, times),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def group_reps(reps: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for rep in reps:
        grouped[(int(rep["motor"]), str(rep["test_type"]))].append(rep)
    return grouped


def derive_all(reps: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_reps(reps)

    bands: dict[str, Any] = {"position": {}, "current": {}}
    rep_stats_all: list[dict[str, Any]] = []
    validation_all: list[dict[str, Any]] = []

    for signal in ("position", "current"):
        for (motor, test_type), group in sorted(grouped.items()):
            stats = [repetition_stats(rep, signal) for rep in group]
            rep_stats_all.extend(stats)

            band = derive_band(stats, signal)
            bands[signal].setdefault(test_type, {})[str(motor)] = band

            for rep in group:
                validation_all.append(
                    evaluate_rep(rep, signal, band["lower"], band["upper"])
                )

    return bands, rep_stats_all, validation_all


def leave_one_run_out(reps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = sorted({rep["run"] for rep in reps})
    output: list[dict[str, Any]] = []

    for held_out in runs:
        train = [r for r in reps if r["run"] != held_out]
        test = [r for r in reps if r["run"] == held_out]

        train_grouped = group_reps(train)
        test_grouped = group_reps(test)

        for signal in ("position", "current"):
            for key, test_group in sorted(test_grouped.items()):
                if key not in train_grouped:
                    continue

                train_stats = [repetition_stats(r, signal) for r in train_grouped[key]]
                if len(train_stats) < 2:
                    continue

                band = derive_band(train_stats, signal)

                for rep in test_group:
                    result = evaluate_rep(
                        rep,
                        signal,
                        band["lower"],
                        band["upper"],
                    )
                    result["held_out_run"] = held_out
                    result["train_repetitions"] = len(train_stats)
                    result["band_lower"] = band["lower"]
                    result["band_upper"] = band["upper"]
                    output.append(result)

    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[(row["signal"], int(row["motor"]), str(row["test_type"]))].append(row)

    summary = []
    for (signal, motor, test_type), vals in sorted(groups.items()):
        outside_pct = np.asarray([v["percentage_outside"] for v in vals], dtype=float)
        duration = np.asarray([v["longest_outside_s"] for v in vals], dtype=float)

        summary.append(
            {
                "signal": signal,
                "motor": motor,
                "test_type": test_type,
                "n_repetitions": len(vals),
                "median_outside_pct": float(np.median(outside_pct)),
                "p95_outside_pct": float(np.quantile(outside_pct, 0.95)),
                "max_outside_pct": float(np.max(outside_pct)),
                "median_longest_outside_ms": float(1000.0 * np.median(duration)),
                "p95_longest_outside_ms": float(1000.0 * np.quantile(duration, 0.95)),
                "max_longest_outside_ms": float(1000.0 * np.max(duration)),
            }
        )

    return summary


def bands_to_python(bands: dict[str, Any], signal: str) -> str:
    title = "POSITION_BANDS" if signal == "position" else "CURRENT_BANDS"
    lines = [f"{title} = {{"]

    for test_type in sorted(bands[signal].keys()):
        lines.append(f'    "{test_type}": {{')
        for motor in sorted(bands[signal][test_type].keys(), key=int):
            b = bands[signal][test_type][motor]
            lines.append(
                f'        {motor}: ({b["lower"]:.1f}, {b["upper"]:.1f}),'
            )
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Replay JSONL file. May be supplied multiple times.",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory searched recursively for *_motor_dt_replay*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="healthy_baseline_analysis",
        help="Output folder.",
    )
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.file]

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        paths.extend(sorted(input_dir.rglob("*motor_dt_replay*.jsonl")))

    # De-duplicate.
    paths = list(dict.fromkeys(paths))

    if not paths:
        raise SystemExit("No replay files supplied.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Healthy replay files: {len(paths)}")
    for p in paths:
        print(f"  - {p}")

    reps = collect_repetitions(paths)
    print(f"\nActive motor/test repetitions extracted: {len(reps)}")

    bands, rep_stats, validation = derive_all(reps)
    loo = leave_one_run_out(reps)

    validation_summary = summarise_validation(validation)
    loo_summary = summarise_validation(loo)

    with (output_dir / "healthy_bands.json").open("w", encoding="utf-8") as f:
        json.dump(bands, f, indent=2)

    write_csv(output_dir / "repetition_residual_statistics.csv", rep_stats)
    write_csv(output_dir / "healthy_band_validation.csv", validation)
    write_csv(output_dir / "healthy_band_validation_summary.csv", validation_summary)
    write_csv(output_dir / "leave_one_run_out_validation.csv", loo)
    write_csv(output_dir / "leave_one_run_out_validation_summary.csv", loo_summary)

    python_text = (
        bands_to_python(bands, "position")
        + "\n\n"
        + bands_to_python(bands, "current")
        + "\n"
    )
    (output_dir / "bands_for_diagnostics.py.txt").write_text(
        python_text,
        encoding="utf-8",
    )

    # # Simple asymmetry summary for M0 position.
    # m0_lines = ["M0 position-band asymmetry (|lower| - upper; positive = more negative):"]
    # for test_type, motors in sorted(bands["position"].items()):
    #     if "0" not in motors:
    #         continue
    #     b = motors["0"]
    #     asym = abs(float(b["lower"])) - float(b["upper"])
    #     m0_lines.append(
    #         f"  {test_type}: [{b['lower']:.1f}, {b['upper']:.1f}] pulses, "
    #         f"asymmetry={asym:.1f} pulses"
    #     )

    # (output_dir / "m0_position_asymmetry.txt").write_text(
    #     "\n".join(m0_lines) + "\n",
    #     encoding="utf-8",
    # )

    print("\nFinal bands:\n")
    print(python_text)

    # print("\nM0 asymmetry:")
    # print("\n".join(m0_lines))

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
