#!/usr/bin/env python3
import cv2
import numpy as np
import math
import json
from pathlib import Path
import subprocess
import argparse
from sympy import fps
import yaml

from ament_index_python.packages import get_package_share_directory

DEFAULT_CONFIG_PATH = (
    Path(get_package_share_directory("adlap_tool_control"))
    / "config"
    / "video_angle_detector.yaml"
)

def load_detector_config(config_path: Path):
    config_path = Path(config_path).expanduser()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["video_angle_detector"]

def read_led_on_times_from_ros_log(path):
    led_on_times = []
    previous_led = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            current_led = record.get("led_command") is True

            if current_led and not previous_led:
                led_on_times.append(
                    float(record["ros_timestamp"])
                )

            previous_led = current_led

    if not led_on_times:
        raise RuntimeError(
            f"No LED ON timestamps found in {path}"
        )

    return led_on_times

def normalize_angle_deg(angle):
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def nan_to_none(value):
    if value is None or np.isnan(value):
        return None
    return float(value)

def centered_median_filter(values, window_size=5):
    """
    Offline centred median filter.

    For sample i, a five-frame window uses:
        i-2, i-1, i, i+1, i+2

    The raw value remains invalid when the marker was not detected in the
    centre frame. The filter therefore does not invent measurements.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "Median filter window must be a positive odd number."
        )

    values = np.asarray(
        [
            np.nan if value is None else float(value)
            for value in values
        ],
        dtype=float,
    )

    filtered = np.full_like(values, np.nan)
    half_window = window_size // 2

    for index in range(len(values)):
        # Keep missing centre measurements missing.
        if not np.isfinite(values[index]):
            continue

        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)

        window = values[start:end]
        valid_window = window[np.isfinite(window)]

        if len(valid_window) >= 3:
            filtered[index] = float(np.median(valid_window))
        else:
            # not enough valid measurements in the window to compute a median
            # use the raw value
            filtered[index] = values[index]

    return filtered    

def start_h264_writer(
    output_path,
    fps,
    width,
    height,
    video_writer_config,
):
    command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", video_writer_config["preset"],
        "-crf", str(video_writer_config["crf"]),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    return subprocess.Popen(command, stdin=subprocess.PIPE)

def detect_led(frame, led_config):
    x, y, w, h = led_config["roi"]
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array(
        led_config["hsv_lower_1"],
        dtype=np.uint8,
    )
    upper_red1 = np.array(
        led_config["hsv_upper_1"],
        dtype=np.uint8,
    )
    lower_red2 = np.array(
        led_config["hsv_lower_2"],
        dtype=np.uint8,
    )
    upper_red2 = np.array(
        led_config["hsv_upper_2"],
        dtype=np.uint8,
    )

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = int(np.count_nonzero(mask))

    pixel_threshold = int(
        led_config["red_pixel_threshold"]
    )

    led_on = red_pixels > pixel_threshold

    return led_on, red_pixels



def detect_red_marker_angle(frame, marker_config, kernel):
    x, y, w, h = marker_config["roi"]
    min_area = float(marker_config["min_area"])

    red_config = marker_config["red"]

    lower_red1 = np.array(red_config["hsv_lower_1"], dtype=np.uint8)
    upper_red1 = np.array(red_config["hsv_upper_1"], dtype=np.uint8)
    lower_red2 = np.array(red_config["hsv_lower_2"], dtype=np.uint8)
    upper_red2 = np.array(red_config["hsv_upper_2"], dtype=np.uint8)

    min_aspect_ratio = float(
        red_config["min_aspect_ratio"]
    )
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw_box, rh_box), _ = rect

        # Rotated aspect ratio: beter voor schuine markers
        aspect = max(rw_box, rh_box) / max(1.0, min(rw_box, rh_box))

        if aspect < min_aspect_ratio:
            continue

        valid.append((area, c))

    if not valid:
        return np.nan, None

    largest = max(valid, key=lambda item: item[0])[1]
    area = cv2.contourArea(largest)

    if area < min_area:
        return np.nan, None

    vx, vy, x0, y0 = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()

    angle = normalize_angle_deg(math.degrees(math.atan2(vy, vx)))

    line = {
        "vx": vx,
        "vy": vy,
        "x0_abs": x + x0,
        "y0_abs": y + y0,
    }

    return angle, line

def detect_yellow_marker_angle(frame, marker_config, kernel):
    x, y, w, h = marker_config["roi"]
    min_area = float(marker_config["min_area"])

    yellow_config = marker_config["yellow"]

    lower_yellow = np.array(
        yellow_config["hsv_lower"],
        dtype=np.uint8,
    )
    upper_yellow = np.array(
        yellow_config["hsv_upper"],
        dtype=np.uint8,
    )

    min_aspect_ratio = float(
        yellow_config["min_aspect_ratio"]
    )

    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw_box, rh_box), _ = rect

        # Rotated aspect ratio: werkt ook als de marker schuin staat
        aspect = max(rw_box, rh_box) / max(1.0, min(rw_box, rh_box))

        if aspect < min_aspect_ratio:
            continue

        valid.append((area, aspect, c))

    if not valid:
        return np.nan, None

    # Kies de grootste geldige langwerpige contour
    largest = max(valid, key=lambda item: item[0])[2]
    area = cv2.contourArea(largest)
    
    vx, vy, x0, y0 = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()

    angle = normalize_angle_deg(math.degrees(math.atan2(vy, vx)))

    line = {
        "vx": vx,
        "vy": vy,
        "x0_abs": x + x0,
        "y0_abs": y + y0,
    }

    return angle, line

def detect_green_marker_angle(frame, marker_config, kernel):
    x, y, w, h = marker_config["roi"]
    min_area = float(marker_config["min_area"])
    green_config = marker_config["green"]
    min_aspect_ratio = float(green_config["min_aspect_ratio"])

    lower_green = np.array(
        green_config["hsv_lower"],
        dtype=np.uint8,
    )

    upper_green = np.array(
        green_config["hsv_upper"],
        dtype=np.uint8,
    )

    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    valid = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (box_width, box_height), _ = rect

        aspect = max(box_width, box_height) / max(
            1.0,
            min(box_width, box_height),
        )

        if aspect < min_aspect_ratio:
            continue

        valid.append((area, aspect, contour))

    if not valid:
        return np.nan, None

    largest = max(valid, key=lambda item: item[0])[2]

    vx, vy, x0, y0 = cv2.fitLine(
        largest,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    )

    vx = vx.item()
    vy = vy.item()
    x0 = x0.item()
    y0 = y0.item()

    angle = normalize_angle_deg(
        math.degrees(math.atan2(vy, vx))
    )

    line = {
        "vx": vx,
        "vy": vy,
        "x0_abs": x + x0,
        "y0_abs": y + y0,
    }

    return angle, line

def detect_blue_marker_angle(frame, marker_config, kernel):
    x, y, w, h = marker_config["roi"]
    min_area = float(marker_config["min_area"])
    blue_config = marker_config["blue"]
    # min_aspect_ratio = float(blue_config["min_aspect_ratio"])

    lower_blue = np.array(
        blue_config["hsv_lower"],
        dtype=np.uint8,
    )

    upper_blue = np.array(
        blue_config["hsv_upper"],
        dtype=np.uint8,
    )

    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        valid.append((cx, area, c))

    if not valid:
        return np.nan, None

    # blauwe marker zit links in de marker ROI, dus kies de meest linkse geldige contour
    _, _, marker = min(valid, key=lambda item: item[0])

    vx, vy, x0, y0 = cv2.fitLine(marker, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()

    angle = normalize_angle_deg(math.degrees(math.atan2(vy, vx)))

    line = {
        "vx": vx,
        "vy": vy,
        "x0_abs": x + x0,
        "y0_abs": y + y0,
    }

    return angle, line

def detect_shaft_angle(frame, shaft_config):
    x, y, w, h = shaft_config["roi"]

    min_points = int(
        shaft_config["min_points"]
    )
    gaussian_kernel_size = int(
        shaft_config["gaussian_kernel_size"]
    )
    canny_low = int(
        shaft_config["canny_low"]
    )
    canny_high = int(
        shaft_config["canny_high"]
    )
    margin = int(
        shaft_config["edge_margin"]
    )

    roi = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (gaussian_kernel_size, gaussian_kernel_size),
        0,
    )

    edges = cv2.Canny(
        blurred,
        canny_low,
        canny_high,
    )

    h_sub, w_sub = edges.shape
    points = []

    # Per column, find the lowest edge pixel (highest y-value) and add it to the list of points.

    for x_col in range(margin, w_sub - margin):
        ys = np.where(edges[:, x_col] > 0)[0]
        if len(ys) > 0:
            y_edge = np.max(ys)
            points.append([x_col, y_edge])

    if len(points) < min_points:
        return np.nan, None

    points = np.array(points, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()

    angle = normalize_angle_deg(math.degrees(math.atan2(vy, vx)))

    line = {
        "vx": vx,
        "vy": vy,
        "x0_abs": x + x0,
        "y0_abs": y + y0,
    }

    return angle, line

def draw_line(frame, line, color):
    if line is None:
        return

    vx = line["vx"]
    vy = line["vy"]
    x0 = line["x0_abs"]
    y0 = line["y0_abs"]

    length = 2000
    pt1 = (int(x0 - vx * length), int(y0 - vy * length))
    pt2 = (int(x0 + vx * length), int(y0 + vy * length))

    cv2.line(frame, pt1, pt2, color, 3)

def relative_angle_between(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return normalize_angle_deg(a - b)


def process_video(
    video_path: Path,
    ros_log_path: Path,
    active_dof: int,
    output_path: Path,
    config: dict,
):
    video_path = Path(video_path)
    ros_log_path = Path(ros_log_path)
    output_path = Path(output_path)
    
    detection_config = config["detection"]
    filter_config = config["filtering"]
    led_config = config["led"]
    marker_config = config["markers"]
    shaft_config = config["shaft"]
    morphology_config = config["morphology"]

    kernel_size = int(
        morphology_config["kernel_size"]
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    debug_config = config["debug"]
    video_writer_config = config["video_writer"]

    use_red_marker = bool(detection_config["use_red_marker"])
    use_yellow_marker = bool(detection_config["use_yellow_marker"])
    use_green_marker = bool(detection_config["use_green_marker"])
    use_blue_marker = bool(detection_config["use_blue_marker"])
    use_shaft = bool(detection_config["use_shaft"])

    secondary_marker_flags = {
        "yellow": use_yellow_marker,
        "green": use_green_marker,
        "blue": use_blue_marker,
    }

    active_secondary_markers = [
        color
        for color, enabled in secondary_marker_flags.items()
        if enabled
    ]

    if active_dof == 4:
        if len(active_secondary_markers) != 1:
            raise ValueError(
                "DOF4 requires exactly one secondary marker "
                "(yellow, green, or blue) to be enabled."
            )

        secondary_marker_color = active_secondary_markers[0]

    else:
        secondary_marker_color = None

    if not use_red_marker:
        raise ValueError(
            "The red marker must be enabled for DOF2 and DOF4."
        )

    if not use_shaft:
        raise ValueError(
            "Shaft detection must be enabled for DOF2 and DOF4."
        )
        
    output_dir = output_path.parent
    video_stem = video_path.stem
    dof_name = f"dof{active_dof}"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_led_on_frames = []
    min_led_pulse_gap_frames = int(round(3.0 * fps))

    frame_idx = 0

    previous_led = False
    led_on_frame = None
    led_off_frame = None

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem

    debug_video_path = output_dir / f"{video_stem}_{dof_name}_h264.mp4"
    debug_frame_path = output_dir / f"{video_stem}_{dof_name}_debug_frame.png"

    open_jaws_debug_frame_saved = False
    open_jaws_debug_frame_path = output_path.with_name(
        output_path.stem + "_open_jaws_debug_frame.png"
    )

    debug_writer = None
    motor_led_on_times_s = read_led_on_times_from_ros_log(
        ros_log_path
    )
    motor_led_on_time_s = motor_led_on_times_s[0]    

    print(f"Using ROS log: {ros_log_path}")
    print(f"Motor LED ON time: {motor_led_on_time_s}")

    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_time_s = frame_idx / fps

        led_on, led_red_pixels = detect_led(frame, led_config)
        if led_on and not previous_led:
            if (
                not video_led_on_frames
                or frame_idx - video_led_on_frames[-1]
                >= min_led_pulse_gap_frames
            ):
                video_led_on_frames.append(frame_idx)

        if led_on_frame is None and led_on:
            led_on_frame = frame_idx
            print(
                f"LED ON frame: {frame_idx}, "
                f"time: {video_time_s:.3f} s, "
                f"red_pixels: {led_red_pixels}"
            )

        if led_on_frame is not None and led_off_frame is None:
            if previous_led and not led_on:
                led_off_frame = frame_idx
                print(
                    f"LED OFF frame: {frame_idx}, "
                    f"time: {video_time_s:.3f} s, "
                    f"red_pixels: {led_red_pixels}"
                )

        if led_on_frame is None:
            synced_time_s = None
        else:
            synced_time_s = (frame_idx - led_on_frame) / fps
        
        if synced_time_s is None:
            ros_time_s = None
        else:
            ros_time_s = motor_led_on_time_s + synced_time_s

        red_marker_angle, red_marker_line = detect_red_marker_angle(
            frame,
            marker_config,
            kernel,
        )

        shaft_angle, shaft_line = detect_shaft_angle(
            frame,
            shaft_config,
        )

        secondary_marker_angle = np.nan
        secondary_marker_line = None

        if active_dof == 4:

            if secondary_marker_color == "yellow":
                secondary_marker_angle, secondary_marker_line = (
                    detect_yellow_marker_angle(
                        frame,
                        marker_config,
                        kernel,
                    )
                )

            elif secondary_marker_color == "green":
                secondary_marker_angle, secondary_marker_line = (
                    detect_green_marker_angle(
                        frame,
                        marker_config,
                        kernel,
                    )
                )

            elif secondary_marker_color == "blue":
                secondary_marker_angle, secondary_marker_line = (
                    detect_blue_marker_angle(
                        frame,
                        marker_config,
                        kernel,
                    )
                )
        
        measured_angle_red_shaft = None
        measured_angle_secondary_shaft = None
        measured_angle_between_jaws = None

        if active_dof == 2:

            measured_angle_red_shaft = relative_angle_between(
                red_marker_angle,
                shaft_angle
            )

        elif active_dof == 4:

            measured_angle_red_shaft = relative_angle_between(
                red_marker_angle,
                shaft_angle
            )

            measured_angle_secondary_shaft = relative_angle_between(
                secondary_marker_angle,
                shaft_angle
            )

            if (
                not np.isnan(secondary_marker_angle)
                and not np.isnan(red_marker_angle)
            ):
                measured_angle_between_jaws = abs(
                    normalize_angle_deg(
                        secondary_marker_angle - red_marker_angle
                    )
                )

            # relative_angle = measured_angle_red_shaft
            # yellow_relative_angle = measured_angle_yellow_shaft
        
        record = {
            "active_dof": active_dof,
            "frame_index": frame_idx,
            "video_time_s": video_time_s,
            "synced_time_s": synced_time_s,
            "ros_time_s": ros_time_s,
            "led_on": bool(led_on),
            "led_red_pixels": led_red_pixels,

            "shaft_angle_deg": nan_to_none(shaft_angle),
            "red_marker_angle_deg": nan_to_none(red_marker_angle),
            "measured_angle_red_shaft": nan_to_none(measured_angle_red_shaft),            
        }
        
        if active_dof == 4:
            record["secondary_marker_color"] = (
                secondary_marker_color
            )

            record[
                f"{secondary_marker_color}_marker_angle_deg"
            ] = nan_to_none(
                secondary_marker_angle
            )

            record[
                f"measured_angle_{secondary_marker_color}_shaft"
            ] = nan_to_none(
                measured_angle_secondary_shaft
            )

            record["measured_angle_between_jaws"] = nan_to_none(
                measured_angle_between_jaws
            )

        records.append(record)

        # Preview
        display = frame.copy()

        x, y, w, h = led_config["roi"]
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 3)

        x, y, w, h = marker_config["roi"]
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)

        x, y, w, h = shaft_config["roi"]
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 3)

        draw_line(display, red_marker_line, (0, 0, 255))
        draw_line(display, shaft_line, (0, 0, 0))
        draw_line(display, secondary_marker_line, (0, 255, 0)) 

        cv2.putText(display, f"frame: {frame_idx}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"LED: {led_on}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"red marker: {red_marker_angle:.2f}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"red marker vs shaft angle: {measured_angle_red_shaft:.2f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"shaft: {shaft_angle:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if active_dof == 4:
            cv2.putText(
                display,
                (
                    f"{secondary_marker_color} marker: "
                    f"{secondary_marker_angle:.2f}"
                ),
                (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        if measured_angle_secondary_shaft is not None:
            cv2.putText(display, f"{secondary_marker_color} marker vs shaft angle: {measured_angle_secondary_shaft:.2f}", (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    
        preview_scale = float(
            debug_config["preview_scale"]
        )

        small = cv2.resize(
            display,
            None,
            fx=preview_scale,
            fy=preview_scale,
        )

        if (
            not open_jaws_debug_frame_saved
            and measured_angle_secondary_shaft is not None
            and not np.isnan(measured_angle_secondary_shaft)
            and abs(measured_angle_secondary_shaft) > float(
                debug_config["open_jaws_threshold_deg"]
            )
            and ros_time_s is not None
        ):
            cv2.imwrite(str(open_jaws_debug_frame_path), display)
            print(f"Saved open jaws debug frame: {open_jaws_debug_frame_path}")
            open_jaws_debug_frame_saved = True

        if debug_writer is None:
            height, width = small.shape[:2]
            debug_writer = start_h264_writer(
                output_path=debug_video_path,
                fps=fps,
                width=width,
                height=height,
                video_writer_config=video_writer_config,
            )

        debug_writer.stdin.write(small.tobytes())

        if frame_idx == led_on_frame or frame_idx == 0:
            cv2.imwrite(str(debug_frame_path), display)
        cv2.imshow("video angle detector", small)

        key = cv2.waitKey(1)
        if key == 27:
            break

        previous_led = led_on
        frame_idx += 1

    sync_correction_s = 0.0

    print(
        f"Detected LED pulses: "
        f"ROS={len(motor_led_on_times_s)}, "
        f"video={len(video_led_on_frames)}"
    )

    if (
        len(motor_led_on_times_s) >= 2
        and len(video_led_on_frames)
            == len(motor_led_on_times_s)
    ):
        first_video_led_frame = video_led_on_frames[0]
        corrections = []

        for ros_led_time, video_led_frame in zip(
            motor_led_on_times_s,
            video_led_on_frames,
        ):
            currently_mapped_time = (
                motor_led_on_time_s
                + (
                    video_led_frame
                    - first_video_led_frame
                ) / fps
            )

            corrections.append(
                ros_led_time - currently_mapped_time
            )

        sync_correction_s = float(
            np.median(corrections[1:])
        )
        for record in records:
            if record["synced_time_s"] is not None:
                record["synced_time_s"] += (
                    sync_correction_s
                )

            if record["ros_time_s"] is not None:
                record["ros_time_s"] += (
                    sync_correction_s
                )

            record["sync_correction_s"] = (
                sync_correction_s
            )

        print(
            f"Applied multi-LED sync correction: "
            f"{sync_correction_s:+.3f} s"
        )

    else:
        print(
            "WARNING: LED pulse counts do not match; "
            "keeping original first-pulse synchronization."
        )

    red_baseline_values = [
        r["measured_angle_red_shaft"]
        for r in records
        if r["measured_angle_red_shaft"] is not None
        and r["ros_time_s"] is not None
    ][:30]

    red_start_baseline = (
        float(np.median(red_baseline_values))
        if red_baseline_values
        else None
    )

    for r in records:
        raw_red = r["measured_angle_red_shaft"]

        if raw_red is None or red_start_baseline is None:
            r["measured_angle_red_shaft_zeroed"] = None
        else:
            r["measured_angle_red_shaft_zeroed"] = (
                raw_red - red_start_baseline
            )

    print(f"Red start baseline: {red_start_baseline}")
 
    secondary_closed_baseline = None

    if active_dof == 4:
        secondary_raw_key = (
            f"measured_angle_{secondary_marker_color}_shaft"
        )

        valid_secondary_values = [
            r[secondary_raw_key]
            for r in records
            if r.get(secondary_raw_key) is not None
            and r["ros_time_s"] is not None
        ]

        if valid_secondary_values:
            secondary_closed_baseline = float(
                np.percentile(
                    valid_secondary_values,
                    5,
                )
            )

        for r in records:
            raw_secondary = r.get(
                secondary_raw_key
            )

            zeroed_key = (
                f"measured_angle_"
                f"{secondary_marker_color}_shaft_zeroed"
            )

            opening_key = (
                f"{secondary_marker_color}"
                f"_gripper_opening_deg"
            )

            if (
                raw_secondary is None
                or secondary_closed_baseline is None
            ):
                r[zeroed_key] = None
                r[opening_key] = None
            else:
                zeroed = (
                    raw_secondary
                    - secondary_closed_baseline
                )

                r[zeroed_key] = zeroed
                r[opening_key] = max(
                    0.0,
                    zeroed,
                )
    # ---------------------------------------------------------
    # Offline centred median filtering of video measurements
    # ---------------------------------------------------------

    signals_to_filter = {
        "measured_angle_red_shaft_zeroed_filtered":
            "measured_angle_red_shaft_zeroed",
    }

    if active_dof == 4:
        signals_to_filter[
            "measured_angle_between_jaws_filtered"
        ] = "measured_angle_between_jaws"

        signals_to_filter[
            f"measured_angle_{secondary_marker_color}_shaft_zeroed_filtered"
        ] = (
            f"measured_angle_{secondary_marker_color}_shaft_zeroed"
        )

    for filtered_key, raw_key in signals_to_filter.items():
        raw_values = [
            record.get(raw_key)
            for record in records
        ]

        filtered_values = centered_median_filter(
            raw_values,
            window_size=int(filter_config["median_window"]),
        )

        for index, record in enumerate(records):
            record[filtered_key] = nan_to_none(
                filtered_values[index]
            )

    print(
        "Applied centred median filter to video measurements: "
        f"window={filter_config['median_window']} frames"
    )

   
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Secondary closed baseline: {secondary_closed_baseline}")

    if debug_writer is not None:
        debug_writer.stdin.close()
        debug_writer.wait()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved debug video: {debug_video_path}")
    print(f"Saved debug frame: {debug_frame_path}")

    if led_on_frame is not None and led_off_frame is not None:
        duration = (led_off_frame - led_on_frame) / fps
        print(f"LED duration: {duration:.3f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--ros-log", required=True)
    parser.add_argument("--dof", type=int, choices=[2, 4], required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Video angle detector configuration YAML.",
    )

    args = parser.parse_args()

    config = load_detector_config(
        args.config
    )

    process_video(
        video_path=args.video,
        ros_log_path=args.ros_log,
        active_dof=args.dof,
        output_path=args.output,
        config=config,
    )