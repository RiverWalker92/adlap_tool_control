#!/usr/bin/env python3

import cv2
import numpy as np
import math
import json
from pathlib import Path

VIDEO = "/home/leanne/Downloads/IMG_1591.MOV"

VIDEO_PATH = Path(VIDEO)

OUTPUT_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "video_data"
)
ROS_DATA_DIR = (
    Path.home()
    / "ros2_ws"
    / "test_data"
    / "setup_03"
    / "gearbox_instrument"
    # / "continuous_dof4_sinusoid"
)

# LED_ROI = (2960, 930, 160, 160)
LED_ROI = (2508, 828, 788, 660)
MARKER_ROI = (485, 325, 1214, 1165)
SHAFT_ROI = (1585, 725, 2254, 400)


LED_RED_PIXEL_THRESHOLD = 100
MARKER_MIN_AREA = 100
SHAFT_MIN_POINTS = 50

KERNEL = np.ones((5, 5), np.uint8)
def find_latest_ros_log():
    files = list(ROS_DATA_DIR.rglob("*.jsonl")) + list(ROS_DATA_DIR.rglob("*.json"))
    if not files:
        raise RuntimeError(f"No ROS log files found in {ROS_DATA_DIR}")

    return max(files, key=lambda p: p.stat().st_mtime)


def read_led_on_time_from_ros_log(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("led_command") is True:
                return float(record["ros_timestamp"])

    raise RuntimeError(f"No LED ON timestamp found in {path}")

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


def detect_led(frame):
    x, y, w, h = LED_ROI
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Streng: alleen fel rood telt als LED aan
    lower_red1 = np.array([0, 120, 180])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 180])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = int(np.count_nonzero(mask))
    led_on = red_pixels > LED_RED_PIXEL_THRESHOLD

    return led_on, red_pixels


def detect_marker_angle(frame):
    x, y, w, h = MARKER_ROI
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Minder streng dan LED, want marker is tape
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.nan, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MARKER_MIN_AREA:
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
def detect_yellow_marker_angle(frame):
    x, y, w, h = MARKER_ROI
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.nan, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MARKER_MIN_AREA:
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

def detect_shaft_angle(frame):
    x, y, w, h = SHAFT_ROI
    roi = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    h_sub, w_sub = edges.shape
    points = []

    # Pak per kolom de bovenste edge; shaft is bijna horizontaal
    margin = 10
    for x_col in range(margin, w_sub - margin):
        ys = np.where(edges[:, x_col] > 0)[0]
        if len(ys) > 0:
            y_edge = np.min(ys)
            points.append([x_col, y_edge])

    if len(points) < SHAFT_MIN_POINTS:
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


def main():
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    previous_led = False
    led_on_frame = None
    led_off_frame = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = Path(VIDEO).stat().st_mtime

    video_stem = VIDEO_PATH.stem

    output_path = (
        OUTPUT_DIR
        / f"{video_stem}_angles.jsonl"
    )

    debug_video_path = OUTPUT_DIR / f"{video_stem}_debug.mp4"
    debug_frame_path = OUTPUT_DIR / f"{video_stem}_debug_frame.png"

    debug_writer = None

    ros_log_path = find_latest_ros_log()
    motor_led_on_time_s = read_led_on_time_from_ros_log(ros_log_path)
    print(f"Using ROS log: {ros_log_path}")
    print(f"Motor LED ON time: {motor_led_on_time_s}")

    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_time_s = frame_idx / fps

        led_on, led_red_pixels = detect_led(frame)

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

        marker_angle, marker_line = detect_marker_angle(frame)
        yellow_marker_angle, yellow_marker_line = detect_yellow_marker_angle(frame)
        shaft_angle, shaft_line = detect_shaft_angle(frame)

        relative_angle = np.nan
        if not np.isnan(marker_angle) and not np.isnan(shaft_angle):
            relative_angle = normalize_angle_deg(marker_angle - shaft_angle)

        yellow_relative_angle = np.nan
        if not np.isnan(yellow_marker_angle) and not np.isnan(shaft_angle):
            yellow_relative_angle = normalize_angle_deg(
                yellow_marker_angle - shaft_angle
            )   
            
        record = {
            "frame_index": frame_idx,
            "video_time_s": video_time_s,
            "synced_time_s": synced_time_s,
            "ros_time_s": ros_time_s,
            "led_on": bool(led_on),
            "led_red_pixels": led_red_pixels,
            "shaft_angle_deg": nan_to_none(shaft_angle),
            "marker_angle_deg": nan_to_none(marker_angle),
            "yellow_marker_angle_deg": nan_to_none(yellow_marker_angle),
            "relative_marker_to_shaft_angle_deg": nan_to_none(relative_angle),
            "yellow_relative_angle_deg": nan_to_none(yellow_relative_angle),
        }

        records.append(record)

        # Preview
        display = frame.copy()

        x, y, w, h = LED_ROI
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 3)

        x, y, w, h = MARKER_ROI
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 3)

        x, y, w, h = SHAFT_ROI
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 3)

        draw_line(display, marker_line, (0, 0, 255))
        draw_line(display, shaft_line, (255, 0, 0))
        draw_line(display, yellow_marker_line, (0, 255, 255))

        cv2.putText(display, f"frame: {frame_idx}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"LED: {led_on}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"marker: {marker_angle:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"shaft: {shaft_angle:.2f}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(display, f"relative: {relative_angle:.2f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"yellow marker: {yellow_marker_angle:.2f}", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"yellow relative: {yellow_relative_angle:.2f}", (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        small = cv2.resize(display, None, fx=0.35, fy=0.35)

        if debug_writer is None:
            height, width = small.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(
                str(debug_video_path),
                fourcc,
                fps,
                (width, height)
            )

        debug_writer.write(small)

        if frame_idx == led_on_frame or frame_idx == 0:
            cv2.imwrite(str(debug_frame_path), small)
        cv2.imshow("video angle detector", small)

        key = cv2.waitKey(1)
        if key == 27:
            break

        previous_led = led_on
        frame_idx += 1

    red_baseline_values = [
        r["relative_marker_to_shaft_angle_deg"]
        for r in records
        if r["relative_marker_to_shaft_angle_deg"] is not None
        and r["ros_time_s"] is not None
    ][:30]

    red_start_baseline = (
        float(np.median(red_baseline_values))
        if red_baseline_values
        else None
    )

    for r in records:
        raw_red = r["relative_marker_to_shaft_angle_deg"]

        if raw_red is None or red_start_baseline is None:
            r["relative_marker_to_shaft_zeroed_deg"] = None
        else:
            r["relative_marker_to_shaft_zeroed_deg"] = (
                raw_red - red_start_baseline
            )

    print(f"Red start baseline: {red_start_baseline}")

    valid_yellow_values = [
        r["yellow_relative_angle_deg"]
        for r in records
        if r["yellow_relative_angle_deg"] is not None
        and r["ros_time_s"] is not None
    ]

    if valid_yellow_values:
        yellow_closed_baseline = float(np.percentile(valid_yellow_values, 5))
    else:
        yellow_closed_baseline = None

    for r in records:
        raw_yellow = r["yellow_relative_angle_deg"]

        if raw_yellow is None or yellow_closed_baseline is None:
            r["yellow_relative_angle_zeroed_deg"] = None
            r["yellow_gripper_opening_deg"] = None
        else:
            zeroed = raw_yellow - yellow_closed_baseline

            r["yellow_relative_angle_zeroed_deg"] = zeroed
            r["yellow_gripper_opening_deg"] = max(0.0, zeroed)

    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Yellow closed baseline: {yellow_closed_baseline}")
    if debug_writer is not None:
        debug_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved debug video: {debug_video_path}")
    print(f"Saved debug frame: {debug_frame_path}")

    if led_on_frame is not None and led_off_frame is not None:
        duration = (led_off_frame - led_on_frame) / fps
        print(f"LED duration: {duration:.3f} s")

if __name__ == "__main__":
    main()