#!/usr/bin/env python3

import argparse
import json
import yaml
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

# def ask_test_mode():
#     while True:
#         mode = input("Select test mode: [t] tool/instrument, [m] motor-only: ").strip().lower()

#         if mode in ["t", "tool"]:
#             return "tool"

#         if mode in ["m", "motor"]:
#             return "motor"

#         print("Invalid choice. Type 't' for tool or 'm' for motor.")

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
        "tool_pattern_runner_node",
        "/tool_pattern_runner_node",
        "/**",
    ]

    for key in preferred_keys:
        if key in config and isinstance(config[key], dict):
            if "ros__parameters" in config[key]:
                return config[key]["ros__parameters"]

    if "ros__parameters" in config:
        return config["ros__parameters"]

    return config

def run_marker_detection(args, output_path: Path) -> dict:
    """
    Runs marker_detector.py before the trial starts.
    The marker detector opens the webcam, detects hardware, writes JSON,
    and closes the webcam again.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    apply_camera_settings(
        camera_device=args.camera,
        focus=args.focus,
        sharpness=args.sharpness,
    )

    marker_detector_path = (
        Path.home()
        / "ros2_ws"
        / "src"
        / "adlap_tool_control"
        / "scripts"
        / "marker_detector.py"
    )

    marker_cmd = (
        f"python3 {shlex.quote(str(marker_detector_path))} "
        f"--config {shlex.quote(str(Path(args.marker_config).expanduser()))} "
        f"--camera {shlex.quote(str(args.camera))} "
        f"--width {args.width} "
        f"--height {args.height} "
        f"--fps {args.fps} "
        f"--output {shlex.quote(str(output_path))}"
    )

    run_command(marker_cmd, check=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Marker detection output was not created: {output_path}")

    with open(output_path, "r") as f:
        detected_hardware = json.load(f)

    return detected_hardware

def test_mode_from_detected_hardware(detected_hardware: dict) -> str:
    detected_mode = detected_hardware.get("mode")

    if detected_mode == "motor_only":
        return "motor"

    if detected_mode == "full_setup":
        return "tool"

    if detected_mode == "gearbox_only":
        raise RuntimeError(
            "Gearbox-only setup was detected, but gearbox-only automated trial mode "
            "is not implemented yet."
        )

    raise RuntimeError(f"Unknown detected hardware mode: {detected_mode}")

def infer_active_dof_from_params(params_file: Path, require_video_supported: bool = True) -> int:
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

    if require_video_supported and active_dof not in [2, 4]:
        raise RuntimeError(
            f"Active DOF is dof{active_dof}, but the video detector currently only supports DOF2 and DOF4."
        )

    return active_dof

def wait_for_hardware_ready(test_mode: str, detected_hardware: dict | None):
    """
    Pauses the automated trial after marker detection.
    This gives the user time to physically couple the detected hardware and
    update the starting positions before the actual trial starts.
    """
    if detected_hardware is None:
        return

    detected_mode = detected_hardware.get("mode")

    # Motor-only trials do not require mechanical coupling or start-position update.
    if detected_mode == "motor_only":
        return

    print("\nHardware markers detected:")
    print(f"  detected setup: {detected_mode}")
    print(f"  selected mode:  {test_mode}")

    gearbox = detected_hardware.get("gearbox")
    instrument = detected_hardware.get("instrument")

    if gearbox is not None:
        print(f"  gearbox:        {gearbox.get('name')}")

    if instrument is not None:
        print(f"  instrument:     {instrument.get('name')}")

    print("\nThe automated trial is now paused.")
    print("Do NOT press Enter yet.")
    print("")
    print("Next steps:")
    print("  1. Go back to the main launch/control terminal.")
    print("  2. Physically couple the detected gearbox/instrument setup.")
    print("  3. Initialize the hardware if needed.")
    print("  4. Update the starting positions.")
    print("  5. Check that the setup is mechanically ready.")
    print("  6. Return to this terminal.")
    print("")
    input("Press Enter here only when coupling + initialization + start-position update are done...")
def main():
    parser = argparse.ArgumentParser(
        description="Run full automated AdLap trial: camera settings, video, pattern, angle detection, plotting."
    )
    # parser.add_argument("--mode", choices=["ask", "tool", "motor"], default="ask", help="Select trial mode: tool, motor, or ask interactively.")
    parser.add_argument("--mode", choices=["auto", "tool", "motor"], default="auto", help="Select trial mode: auto, tool or motor")
    parser.add_argument("--dof", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--params-file", default=str(Path.home()/ "ros2_ws"/ "src"/ "adlap_tool_control"/ "config"/ "tool_params.yaml"), 
    help="Parameter file used to infer active DOF if --dof is not provided.",)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--focus", type=int, default=150)
    parser.add_argument("--sharpness", type=int, default=150)

    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--marker-config",
        default=str(Path.home() / "ros2_ws" / "src" / "adlap_tool_control" / "config" / "marker_detection.yaml"),
        help="YAML file that maps ArUco marker IDs to hardware.",
    )
    parser.add_argument(
        "--skip-marker-scan",
        action="store_true",
        help="Skip ArUco marker detection and use the active/default configuration.",
    )

    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Run trial without webcam/video detection and plotting.",
    )

    parser.add_argument(
        "--motor-setup-name",
        default="setup_01_motors",
        help="Folder name used for motor-only automated trials.",
    )

    parser.add_argument(
        "--tool-setup-name",
        default="setup_03_motors_gearbox_instrument",
        help="Folder name used for tool/gearbox/instrument automated trials.",
    )

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

    detected_hardware = None
    gearbox_variant = None
    instrument_config = None

    if args.skip_marker_scan:
        print("")
        print("Skipping ArUco marker detection.")
        print("Using active/default configuration.")
        print("")

        if args.mode == "auto":
            test_mode = "tool"
        else:
            test_mode = args.mode

    elif args.mode == "auto":
        marker_output_path = (
            Path(args.output_dir).expanduser()
            / "_latest_detected_hardware.json"
        )

        detected_hardware = run_marker_detection(
            args=args,
            output_path=marker_output_path,
        )

        test_mode = test_mode_from_detected_hardware(detected_hardware)

        if detected_hardware is not None:
            gearbox = detected_hardware.get("gearbox")
            instrument = detected_hardware.get("instrument")

            if gearbox is not None:
                gearbox_variant = gearbox.get("gearbox_variant")

            if instrument is not None:
                instrument_config = instrument.get("instrument_config")
                
        print("\nAutomatically selected trial mode from markers:")
        print(f"  marker_detector mode: {detected_hardware.get('mode')}")
        print(f"  automated trial mode: {test_mode}")

        wait_for_hardware_ready(
            test_mode=test_mode,
            detected_hardware=detected_hardware,
        )

    else:
        test_mode = args.mode
            
    if test_mode == "motor":
        dof_label = "motor"
        print("Motor-only mode selected: no active DOF is inferred from tool_params.yaml.")

    else:
        if args.dof is None:
            args.dof = infer_active_dof_from_params(
                params_file,
                require_video_supported=False,
            )
            print(f"Inferred active DOF from tool_params.yaml: DOF{args.dof}")
        else:
            print(f"Using DOF from terminal argument: DOF{args.dof}")

        dof_label = f"dof{args.dof}"

    if test_mode == "motor":
        use_camera = False
    elif args.no_camera:
        use_camera = False
    else:
        use_camera = args.dof in [2, 4]

    # Create output directories and file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if test_mode == "motor":
        run_name = f"auto_motor_{timestamp}"
        setup_name = args.motor_setup_name
    else:
        run_name = f"auto_{dof_label}_{timestamp}"
        setup_name = args.tool_setup_name

    root_output_dir = Path(args.output_dir).expanduser()
    setup_dir = root_output_dir / setup_name

    if test_mode == "motor":
        dof_dir = setup_dir / "motor"
    else:
        dof_dir = setup_dir / dof_label

    output_dir = dof_dir / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    ros_log_path = output_dir / f"{run_name}_ros_log.jsonl"
    original_video_path = output_dir / f"{run_name}_webcam.mp4"
    video_path = original_video_path
    angles_path = output_dir / f"{run_name}_webcam_angles.jsonl"
    metadata_path = output_dir / f"{run_name}_metadata.json"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if test_mode == "motor":
        args.pattern_cmd = (
            "ros2 launch adlap_tool_control motor_pattern_runner.launch.py "
            f"output_dir:={shlex.quote(str(output_dir))} "
            f"log_file_name:={shlex.quote(ros_log_path.name)}"
        )
    else:
        args.pattern_cmd = (
            "ros2 launch adlap_tool_control pattern_runner.launch.py "
            f"output_dir:={shlex.quote(str(output_dir))} "
            f"log_file_name:={shlex.quote(ros_log_path.name)}"
        )

        if gearbox_variant is not None:
            args.pattern_cmd += f" gearbox_variant:={shlex.quote(gearbox_variant)}"

        if instrument_config is not None:
            args.pattern_cmd += f" instrument_config:={shlex.quote(instrument_config)}"

    print(f"Selected mode: {test_mode}")
    print(f"Setup folder:  {setup_dir}")
    print(f"DOF folder:    {dof_dir}")
    print(f"Run folder:    {output_dir}")
    print(f"ROS log:       {ros_log_path}")
    print(f"Use camera:    {use_camera}")

    # test_data_dir = Path(args.test_data_dir).expanduser()

    trial_start_time = time.time()
    record_errors = []
    stop_event = None
    recorder_thread = None
    detector_cmd = None
    plot_cmd = None
    if use_camera:
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
    if use_camera:
        time.sleep(args.extra_video_seconds)

        stop_event.set()
        recorder_thread.join()

        if record_errors:
            raise RuntimeError(record_errors[0])

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Video file was not created correctly: {video_path}")
    if pattern_returncode != 0:
        raise RuntimeError("Pattern runner failed, so detector/plot are not started.")

    if not ros_log_path.exists() or ros_log_path.stat().st_size == 0:
        raise RuntimeError(f"ROS log was not created correctly: {ros_log_path}")

    print(f"\nROS log saved at:\n{ros_log_path}")

    motor_dt_replay_path = None
    motor_dt_replay_cmd = None
    instrument_current_dt_replay_path = None
    instrument_current_dt_replay_cmd = None

    if test_mode == "motor":
        motor_dt_replay_path = output_dir / f"{run_name}_motor_dt_replay.jsonl"

        motor_dt_replay_cmd = (
            f"ros2 run adlap_tool_control motor_digital_twin.py "
            f"--file {shlex.quote(str(ros_log_path))}"
        )

        run_command(motor_dt_replay_cmd, check=True)

        if not motor_dt_replay_path.exists() or motor_dt_replay_path.stat().st_size == 0:
            raise RuntimeError(
                f"Motor DT replay file was not created correctly: {motor_dt_replay_path}"
            )

        print(f"\nOffline Motor DT replay saved at:\n{motor_dt_replay_path}")

        plot_cmd = (
            f"ros2 run adlap_tool_control plot_motor_trial.py "
            f"--file {shlex.quote(str(ros_log_path))} "
            f"--output-dir {shlex.quote(str(plots_dir))}"
        )

    else:
        instrument_current_dt_replay_path = (
            output_dir / f"{run_name}_instrument_current_dt_replay.jsonl"
        )

        instrument_current_model_path = (
            Path.home()
            / "ros2_ws"
            / "test_data"
            / "automated_trials"
            / "trainings data"
            / "instrument_current_dt_results"
            / f"dof{args.dof}"
            / "models"
            / f"instrument_current_dt_dof{args.dof}_all_motors.pkl"
        )

        instrument_current_dt_replay_cmd = (
            f"ros2 run adlap_tool_control instrument_current_digital_twin.py "
            f"--file {shlex.quote(str(ros_log_path))} "
            f"--model-file {shlex.quote(str(instrument_current_model_path))} "
            f"--output-file {shlex.quote(str(instrument_current_dt_replay_path))} "
            f"--dof {args.dof}"
        )

        run_command(instrument_current_dt_replay_cmd, check=True)

        if (
            not instrument_current_dt_replay_path.exists()
            or instrument_current_dt_replay_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Instrument Current DT replay file was not created correctly: "
                f"{instrument_current_dt_replay_path}"
            )

        print(
            "\nOffline Instrument Current DT replay saved at:\n"
            f"{instrument_current_dt_replay_path}"
        )

        if use_camera:
            print(f"\nDetected webcam video:\n{video_path}")

            detector_cmd = args.detector_cmd.format(
                video=shlex.quote(str(video_path)),
                ros_log=shlex.quote(str(ros_log_path)),
                dof=args.dof,
                angles=shlex.quote(str(angles_path)),
            )

            run_command(detector_cmd, check=True)

            if original_video_path.exists():
                original_video_path.unlink()
                print(f"Removed temporary raw webcam video: {original_video_path}")

            plot_cmd = args.plot_cmd.format(
                ros_log=shlex.quote(str(ros_log_path)),
                angles=shlex.quote(str(angles_path)),
                plot_dir=shlex.quote(str(plots_dir)),
                dof=args.dof,
                video=shlex.quote(str(video_path)),
                params_file=shlex.quote(str(params_file)),
            )

        else:
            plot_cmd = (
                f"ros2 run adlap_tool_control plot_trial.py "
                f"--file {shlex.quote(str(ros_log_path))} "
                f"--output-dir {shlex.quote(str(plots_dir))} "
                f"--params-file {shlex.quote(str(params_file))}"
            )

    if plot_cmd is not None:
        run_command(plot_cmd, check=True)

    metadata = {
        "mode": test_mode,
        "detected_hardware": detected_hardware,
        "setup_name": setup_name,
        "setup_dir": str(setup_dir),
        "dof_dir": str(dof_dir),
        "run_name": run_name,
        "trial_name": run_name,
        "dof": args.dof,
        "used_camera": use_camera,
        "camera": args.camera,
        "focus_absolute": args.focus,
        "sharpness": args.sharpness,
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "video_path": str(video_path) if use_camera else None,
        "original_video_path": str(original_video_path) if use_camera else None,
        "angles_path": str(angles_path) if use_camera else None,
        "ros_log_path": str(ros_log_path),
        "motor_dt_replay_path": str(motor_dt_replay_path) if motor_dt_replay_path is not None else None,
        "motor_dt_replay_cmd": motor_dt_replay_cmd,
        "instrument_current_dt_replay_path": (str(instrument_current_dt_replay_path) if instrument_current_dt_replay_path is not None else None),
        "instrument_current_dt_replay_cmd": instrument_current_dt_replay_cmd,
        "plots_dir": str(plots_dir),
        "pattern_cmd": args.pattern_cmd,
        "detector_cmd": detector_cmd,
        "plot_cmd": plot_cmd,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFull automated trial completed.")
    # print(f"Video:      {video_path}")
    print(f"ROS log:    {ros_log_path}")
    
    if use_camera:
        print(f"Angles:     {angles_path}")

    if plot_cmd is not None:
        print(f"Plots:      {plots_dir}")
    
    print(f"Metadata:   {metadata_path}")

    if motor_dt_replay_path is not None:
        print(f"Motor DT replay: {motor_dt_replay_path}")


if __name__ == "__main__":
    main()