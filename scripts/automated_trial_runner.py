#!/usr/bin/env python3

import argparse
import json
import yaml
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


def run_command(cmd: str, check: bool = True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {cmd}")
    return result.returncode

def extract_ros_parameters(config: dict) -> dict:
    """
    Extracts ros__parameters from a ROS2 yaml file.
    Supports:
    - tool_controller_node:
        ros__parameters:
    - /tool_controller_node:
        ros__parameters:
    - /**:
        ros__parameters:
    - direct parameter dictionary as fallback
    """
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


def infer_active_dof_from_params(params_file: Path) -> int:
    """
    Infers active DOF from tool_params.yaml.
    Returns 2 or 4, because the current video detector supports DOF2 and DOF4.
    """
    with open(params_file, "r") as f:
        config = yaml.safe_load(f)

    params = extract_ros_parameters(config)

    active_dofs = []

    for i in range(1, 5):
        dof_key = f"dof{i}"

        if dof_key in params and isinstance(params[dof_key], dict):
            mode = params[dof_key].get("mode", "constant")
        else:
            mode = params.get(f"{dof_key}.mode", "constant")

        if mode != "constant":
            active_dofs.append(i)

    if not active_dofs:
        raise RuntimeError(
            f"No active DOF found in {params_file}. "
            "All dof*.mode values appear to be constant."
        )

    if len(active_dofs) > 1:
        raise RuntimeError(
            f"Multiple active DOFs found in {params_file}: {active_dofs}. "
            "Automatic video detection expects one active DOF."
        )

    active_dof = active_dofs[0]

    if active_dof not in [2, 4]:
        raise RuntimeError(
            f"Active DOF is dof{active_dof}, but the video detector currently only supports DOF2 and DOF4."
        )

    return active_dof

def apply_camera_settings(camera_device: str, focus: int, sharpness: int):
    """
    Applies webcam settings using v4l2-ctl.
    These commands are allowed to fail softly because not every webcam exposes
    exactly the same control names.
    """
    settings = [
        "focus_automatic_continuous=0",
        f"focus_absolute={focus}",
        f"sharpness={sharpness}",
    ]

    print("\nApplying webcam settings...")
    for setting in settings:
        cmd = f"v4l2-ctl -d {shlex.quote(camera_device)} -c {setting}"
        returncode = run_command(cmd, check=False)
        if returncode != 0:
            print(f"Warning: could not apply camera setting: {setting}")


def record_video(
    camera_device: str,
    output_video: Path,
    width: int,
    height: int,
    fps: float,
    stop_event: threading.Event,
    error_holder: list,
):
    """
    Records webcam video until stop_event is set.
    """
    try:
        cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera: {camera_device}")

        # Request MJPG from the webcam. This is often needed for stable 4K/30fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first webcam frame.")

        actual_height, actual_width = frame.shape[:2]

        print(
            f"\nRecording video: {output_video}\n"
            f"Requested: {width}x{height} @ {fps} fps\n"
            f"Actual frame size: {actual_width}x{actual_height}"
        )

        output_video.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (actual_width, actual_height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_video}")

        frame_count = 0
        start_time = time.time()

        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                frame_count += 1

        duration = time.time() - start_time
        print(f"Stopped recording. Frames: {frame_count}, duration: {duration:.2f} s")

        writer.release()
        cap.release()

    except Exception as exc:
        error_holder.append(exc)
        stop_event.set()


def find_newest_ros_log(test_data_dir: Path, started_after: float) -> Path:
    """
    Finds the newest ROS trial jsonl file created after the trial started.
    Excludes angle detector outputs.
    """
    candidates = []

    for path in test_data_dir.rglob("*.jsonl"):
        name = path.name.lower()

        # Exclude video-angle output files
        if "angle" in name or "debug" in name:
            continue

        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue

        if mtime >= started_after - 2.0:
            candidates.append(path)

    if not candidates:
        raise RuntimeError(
            f"No ROS trial .jsonl found in {test_data_dir} after start time."
        )

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest
    
def reencode_video_to_h264(input_video: Path):
    """
    Re-encodes OpenCV-written video to a broadly compatible H.264 MP4.
    This fixes videos that look black in some players but contain readable frames.
    """
    fixed_video = input_video.with_name(input_video.stem + "_h264.mp4")

    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(input_video))} "
        f"-c:v libx264 -pix_fmt yuv420p -movflags +faststart "
        f"{shlex.quote(str(fixed_video))}"
    )

    returncode = run_command(cmd, check=False)

    if returncode != 0 or not fixed_video.exists() or fixed_video.stat().st_size == 0:
        print("Warning: H.264 re-encoding failed. Keeping original OpenCV video.")
        return input_video

    print(f"Re-encoded video saved as: {fixed_video}")
    return fixed_video

def main():
    parser = argparse.ArgumentParser(
        description="Run full automated AdLap trial: camera settings, video, pattern, angle detection, plotting."
    )

    parser.add_argument("--dof", type=int, choices=[2, 4], default=None)
    parser.add_argument("--params-file", default=str(Path.home()/ "ros2_ws"/ "src"/ "adlap_tool_control"/ "config"/ "tool_params.yaml"), 
    help="Parameter file used to infer active DOF if --dof is not provided.",)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--focus", type=int, default=150)
    parser.add_argument("--sharpness", type=int, default=150)

    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--test-data-dir",
        default=str(Path.home() / "ros2_ws" / "test_data"),
        help="Root folder where pattern runner writes ROS .jsonl files.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "ros2_ws" / "test_data" / "automated_trials"),
        help="Folder where the webcam video and metadata will be saved.",
    )

    parser.add_argument(
        "--pattern-cmd",
        default="ros2 launch adlap_tool_control pattern_runner.launch.py",
        help="Command that starts the pattern runner. It should block until the trial is done.",
    )

    parser.add_argument(
        "--detector-cmd",
        default=(
            "ros2 run adlap_tool_control webcam_video_angle_detector.py "
            "--video {video} --ros-log {ros_log} --dof {dof} --output {angles}"
        ),
        help="Command for video angle detection. Available placeholders: {video}, {ros_log}, {dof}, {angles}.",
    )

    parser.add_argument(
        "--plot-cmd",
        default=(
            "ros2 run adlap_tool_control plot_trial.py "
            "--file {ros_log} --video-angles {angles} --output-dir {plot_dir} --params-file {params_file}"
        ),
        help="Command for plotting. Available placeholders: {ros_log}, {angles}, {plot_dir}.",
    )

    parser.add_argument(
        "--extra-video-seconds",
        type=float,
        default=1.0,
        help="Seconds to keep recording after the pattern command finishes.",
    )

    args = parser.parse_args()
    params_file = Path(args.params_file).expanduser()

    if args.dof is None:
        args.dof = infer_active_dof_from_params(params_file)
        print(f"Inferred active DOF from tool_params.yaml: DOF{args.dof}")
    else:
        print(f"Using DOF from terminal argument: DOF{args.dof}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_name = f"auto_dof{args.dof}_{timestamp}"

    output_dir = Path(args.output_dir).expanduser() / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)

    original_video_path = output_dir / f"{trial_name}_webcam.mp4"
    video_path = original_video_path
    angles_path = output_dir / f"{trial_name}_webcam_angles.jsonl"
    metadata_path = output_dir / f"{trial_name}_metadata.json"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_data_dir = Path(args.test_data_dir).expanduser()

    apply_camera_settings(
        camera_device=args.camera,
        focus=args.focus,
        sharpness=args.sharpness,
    )

    stop_event = threading.Event()
    record_errors = []

    trial_start_time = time.time()

    recorder_thread = threading.Thread(
        target=record_video,
        kwargs={
            "camera_device": args.camera,
            "output_video": video_path,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "stop_event": stop_event,
            "error_holder": record_errors,
        },
        daemon=True,
    )

    recorder_thread.start()

    # Give the camera a short moment to actually start before the LED/pattern starts.
    time.sleep(1.0)

    if record_errors:
        raise RuntimeError(record_errors[0])

    print("\nStarting pattern runner...")
    pattern_returncode = run_command(args.pattern_cmd, check=False)

    print(f"\nPattern command finished with return code: {pattern_returncode}")
    time.sleep(args.extra_video_seconds)

    stop_event.set()
    recorder_thread.join()

    if record_errors:
        raise RuntimeError(record_errors[0])
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video file was not created correctly: {video_path}")
    video_path = reencode_video_to_h264(video_path)

    if pattern_returncode != 0:
        raise RuntimeError("Pattern runner failed, so detector/plot are not started.")

    ros_log_path = find_newest_ros_log(test_data_dir, trial_start_time)

    print(f"\nDetected ROS log:\n{ros_log_path}")
    print(f"\nDetected webcam video:\n{video_path}")

    detector_cmd = args.detector_cmd.format(
        video=shlex.quote(str(video_path)),
        ros_log=shlex.quote(str(ros_log_path)),
        dof=args.dof,
        angles=shlex.quote(str(angles_path)),
    )

    run_command(detector_cmd, check=True)

    plot_cmd = args.plot_cmd.format(
        ros_log=shlex.quote(str(ros_log_path)),
        angles=shlex.quote(str(angles_path)),
        plot_dir=shlex.quote(str(plots_dir)),
        dof=args.dof,
        video=shlex.quote(str(video_path)),
        params_file=shlex.quote(str(params_file)),
    )

    run_command(plot_cmd, check=True)

    metadata = {
        "trial_name": trial_name,
        "dof": args.dof,
        "camera": args.camera,
        "focus_absolute": args.focus,
        "sharpness": args.sharpness,
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "video_path": str(video_path),
        "original_video_path": str(original_video_path),
        "ros_log_path": str(ros_log_path),
        "angles_path": str(angles_path),
        "plots_dir": str(plots_dir),
        "pattern_cmd": args.pattern_cmd,
        "detector_cmd": detector_cmd,
        "plot_cmd": plot_cmd,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFull automated trial completed.")
    print(f"Video:      {video_path}")
    print(f"ROS log:    {ros_log_path}")
    print(f"Angles:     {angles_path}")
    print(f"Plots:      {plots_dir}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()