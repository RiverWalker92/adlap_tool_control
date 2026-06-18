#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import math
import json
from pathlib import Path
VIDEO = "/home/leanne/ros2_ws/sequence_DOF4_trial_14.mp4"
VIDEO_PATH = Path(VIDEO)

# if len(sys.argv) < 2:
#     raise RuntimeError("Usage: python3 video_angle_detector.py <video_path>")

# VIDEO = sys.argv[1]
# VIDEO_PATH = Path(VIDEO)
ACTIVE_DOF = 4  # 1, 2, 3, 4
DOF_NAME = f"dof{ACTIVE_DOF}"

USE_RED_MARKER = True
USE_YELLOW_MARKER = True
USE_BLUE_MARKER = False
USE_SHAFT = True

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
# LED_ROI = (1154, 560, 262, 200)
# LED_ROI= (1648, 720, 262, 160)
 #(1514, 634, 265, 182)
# MARKER_ROI = (608, 537, 528, 297)
# SHAFT_ROI = (1228, 582, 610, 180)

LED_ROI    = (1457, 380, 454, 180)
MARKER_ROI = (537, 331, 554, 448)
SHAFT_ROI  = (1194, 451, 691, 165)





LED_RED_PIXEL_THRESHOLD = 100
MARKER_MIN_AREA = 60
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


def detect_red_marker_angle(frame):
    x, y, w, h = MARKER_ROI
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Minder streng dan LED, want marker is tape
    # lower_red1 = np.array([0, 80, 80])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([170, 80, 80])
    # upper_red2 = np.array([180, 255, 255])
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MARKER_MIN_AREA:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw_box, rh_box), _ = rect

        # Rotated aspect ratio: beter voor schuine markers
        aspect = max(rw_box, rh_box) / max(1.0, min(rw_box, rh_box))

        if aspect < 2.5:
            continue

        valid.append((area, c))

    if not valid:
        return np.nan, None

    largest = max(valid, key=lambda item: item[0])[1]
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

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MARKER_MIN_AREA:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw_box, rh_box), _ = rect

        # Rotated aspect ratio: werkt ook als de marker schuin staat
        aspect = max(rw_box, rh_box) / max(1.0, min(rw_box, rh_box))

        if aspect < 2.0:
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

def detect_blue_marker_angle(frame):
    x, y, w, h = MARKER_ROI
    roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # lower_blue = np.array([95, 60, 80])
    # upper_blue = np.array([115, 255, 255])
    lower_blue = np.array([95, 40, 50])
    upper_blue = np.array([120, 255, 255])


    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MARKER_MIN_AREA:
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

def detect_shaft_angle(frame):
    x, y, w, h = SHAFT_ROI
    roi = frame[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    h_sub, w_sub = edges.shape
    points = []

    # Pak per kolom de onderste edge; shaft is bijna horizontaal
    margin = 10
    for x_col in range(margin, w_sub - margin):
        ys = np.where(edges[:, x_col] > 0)[0]
        if len(ys) > 0:
            y_edge = np.max(ys)#was min
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

def relative_angle_between(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return normalize_angle_deg(a - b)

def main(output_path_override=None, ros_log_path_override=None):
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

    if output_path_override is None:
        output_path = OUTPUT_DIR / f"{video_stem}_angles.jsonl"
    else:
        output_path = Path(output_path_override)

    debug_video_path = OUTPUT_DIR / f"{video_stem}_{DOF_NAME}_debug.mp4"
    debug_frame_path = OUTPUT_DIR / f"{video_stem}_{DOF_NAME}_debug_frame.png"

    open_jaws_debug_frame_saved = False
    open_jaws_debug_frame_path = output_path.with_name(
        output_path.stem + "_open_jaws_debug_frame.png"
    )

    debug_writer = None

    if ros_log_path_override is None:
        ros_log_path = find_latest_ros_log()
    else:
        ros_log_path = Path(ros_log_path_override)
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

        if ACTIVE_DOF == 4:

            red_marker_angle, red_marker_line = detect_red_marker_angle(frame)
            blue_marker_angle, blue_marker_line = np.nan, None
            yellow_marker_angle, yellow_marker_line = detect_yellow_marker_angle(frame) 
            shaft_angle, shaft_line = detect_shaft_angle(frame)

        elif ACTIVE_DOF == 2:

            red_marker_angle, red_marker_line = detect_red_marker_angle(frame)
            shaft_angle, shaft_line = detect_shaft_angle(frame)

            yellow_marker_angle, yellow_marker_line = np.nan, None
            blue_marker_angle, blue_marker_line = np.nan, None

        else:

            red_marker_angle, red_marker_line = detect_red_marker_angle(frame)
            shaft_angle, shaft_line = detect_shaft_angle(frame)

            yellow_marker_angle, yellow_marker_line = np.nan, None
            # blue_marker_angle, blue_marker_line = np.nan, None
    

        measured_angle_red_shaft = None
        measured_angle_yellow_shaft = None
        # measured_angle_yellow_blue = None
        # measured_angle_red_blue = None
        measured_angle_between_jaws = None

        if ACTIVE_DOF == 2:

            measured_angle_red_shaft = relative_angle_between(
                red_marker_angle,
                shaft_angle
            )

            # relative_angle = measured_angle_red_shaft

        elif ACTIVE_DOF == 4:

            measured_angle_red_shaft = relative_angle_between(
                red_marker_angle,
                shaft_angle
            )

            measured_angle_yellow_shaft = relative_angle_between(
                yellow_marker_angle,
                shaft_angle
            )
            measured_angle_blue_shaft = relative_angle_between(
                blue_marker_angle,
                shaft_angle
            )

            # measured_angle_yellow_blue = relative_angle_between(
            #     yellow_marker_angle,
            #     blue_marker_angle
            # )

            # measured_angle_red_blue = relative_angle_between(
            #     red_marker_angle,
            #     blue_marker_angle
            # )

            if (
                not np.isnan(yellow_marker_angle)
                and not np.isnan(red_marker_angle)
            ):
                measured_angle_between_jaws = abs(
                    normalize_angle_deg(
                        yellow_marker_angle - red_marker_angle
                    )
                )

            # relative_angle = measured_angle_red_shaft
            # yellow_relative_angle = measured_angle_yellow_shaft
        
        record = {
            "active_dof": ACTIVE_DOF,
            "frame_index": frame_idx,
            "video_time_s": video_time_s,
            "synced_time_s": synced_time_s,
            "ros_time_s": ros_time_s,
            "led_on": bool(led_on),
            "led_red_pixels": led_red_pixels,

            "shaft_angle_deg": nan_to_none(shaft_angle),
            "red_marker_angle_deg": nan_to_none(red_marker_angle),
            "yellow_marker_angle_deg": nan_to_none(yellow_marker_angle),
            # "blue_marker_angle_deg": nan_to_none(blue_marker_angle),
            "measured_angle_between_jaws": nan_to_none(measured_angle_between_jaws),
            # "measured_angle_yellow_blue": nan_to_none(measured_angle_yellow_blue),
            # "measured_angle_red_blue": nan_to_none(measured_angle_red_blue),
            "measured_angle_yellow_shaft": nan_to_none(measured_angle_yellow_shaft),
            "measured_angle_red_shaft": nan_to_none(measured_angle_red_shaft),
            # "blue_marker_angle_deg": nan_to_none(blue_marker_angle),
            # "measured_angle_blue_shaft": nan_to_none(measured_angle_blue_shaft),
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

        draw_line(display, red_marker_line, (0, 0, 255))
        draw_line(display, shaft_line, (0, 0, 0))
        # draw_line(display, yellow_marker_line, (0, 255, 255))
        draw_line(display, yellow_marker_line, (0, 255, 255))

        cv2.putText(display, f"frame: {frame_idx}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"LED: {led_on}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, f"red marker: {red_marker_angle:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"shaft: {shaft_angle:.2f}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(display, f"measured red marker vs shaft angle: {measured_angle_red_shaft:.2f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"yellow marker: {yellow_marker_angle:.2f}", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if measured_angle_yellow_shaft is not None:
            cv2.putText(
                display,
                f"measured yellow marker vs shaft angle: {measured_angle_yellow_shaft:.2f}",
                (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
        # cv2.putText(display, f"blue marker: {blue_marker_angle:.2f}", (30, 250),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # cv2.putText(display, f"measured blue marker vs shaft angle: {measured_angle_blue_shaft:.2f}", (30, 290),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        small = cv2.resize(display, None, fx=0.35, fy=0.35)

        if (
            not open_jaws_debug_frame_saved
            and measured_angle_yellow_shaft is not None
            and not np.isnan(measured_angle_yellow_shaft)
            and abs(measured_angle_yellow_shaft) > 3.0
            and ros_time_s is not None
        ):
            cv2.imwrite(str(open_jaws_debug_frame_path), small)
            print(f"Saved open jaws debug frame: {open_jaws_debug_frame_path}")
            open_jaws_debug_frame_saved = True

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

    # valid_blue_values = [
    #     r["measured_angle_blue_shaft"]
    #     for r in records
    #     if r["measured_angle_blue_shaft"] is not None
    #     and r["ros_time_s"] is not None
    # ]

    # if valid_blue_values:
    #     blue_closed_baseline = float(np.percentile(valid_blue_values, 5))
    # else:
    #     blue_closed_baseline = None

    # for r in records:
    #     raw_blue = r["measured_angle_blue_shaft"]

    #     if raw_blue is None or blue_closed_baseline is None:
    #         r["measured_angle_blue_shaft_zeroed"] = None
    #         r["blue_relative_angle_zeroed_deg"] = None
    #         r["blue_gripper_opening_deg"] = None
    #     else:
    #         zeroed = raw_blue - blue_closed_baseline

    #         r["measured_angle_blue_shaft_zeroed"] = zeroed
    #         r["blue_relative_angle_zeroed_deg"] = zeroed
    #         r["blue_gripper_opening_deg"] = max(0.0, zeroed)
    # 
    valid_yellow_values = [
        r["measured_angle_yellow_shaft"]
        for r in records
        if r["measured_angle_yellow_shaft"] is not None
        and r["ros_time_s"] is not None
    ]
    if valid_yellow_values:
        yellow_closed_baseline = float(np.percentile(valid_yellow_values, 5))
    else:
        yellow_closed_baseline = None

    for r in records:
        raw_yellow = r["measured_angle_yellow_shaft"]

        if raw_yellow is None or yellow_closed_baseline is None:
            r["measured_angle_yellow_shaft_zeroed"] = None
            r["yellow_relative_angle_zeroed_deg"] = None
            r["yellow_gripper_opening_deg"] = None
        else:
            zeroed = raw_yellow - yellow_closed_baseline

            r["measured_angle_yellow_shaft_zeroed"] = zeroed
            r["yellow_relative_angle_zeroed_deg"] = zeroed
            r["yellow_gripper_opening_deg"] = max(0.0, zeroed)
   
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Yellow closed baseline: {yellow_closed_baseline}")
    # print(f"Blue closed baseline: {blue_closed_baseline}")
    if debug_writer is not None:
        debug_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved debug video: {debug_video_path}")
    print(f"Saved debug frame: {debug_frame_path}")

    if led_on_frame is not None and led_off_frame is not None:
        duration = (led_off_frame - led_on_frame) / fps
        print(f"LED duration: {duration:.3f} s")

def process_video(video_path, ros_log_path, dof, output_path):
    global VIDEO, VIDEO_PATH, ACTIVE_DOF, DOF_NAME, OUTPUT_DIR

    VIDEO = str(video_path)
    VIDEO_PATH = Path(video_path)
    ACTIVE_DOF = int(dof)
    DOF_NAME = f"dof{ACTIVE_DOF}"
    OUTPUT_DIR = Path(output_path).parent

    main(
        output_path_override=Path(output_path),
        ros_log_path_override=Path(ros_log_path),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ros-log", required=True)
    parser.add_argument("--dof", type=int, choices=[2, 4], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        ros_log_path=args.ros_log,
        dof=args.dof,
        output_path=args.output,
    )