#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import cv2
import yaml


def load_marker_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_aruco_dictionary(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install/check OpenCV with ArUco support."
        )

    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_marker_in_frame(frame, dictionary):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Newer OpenCV API
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

    # Older OpenCV API
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return [], corners, ids

    detected_ids = [int(marker_id[0]) for marker_id in ids]
    return detected_ids, corners, ids


def wait_for_stable_marker(cap, dictionary, known_markers, stable_frames=5, preview_width=960):
    """
    Wait until the same known marker is detected for multiple frames.
    This avoids accidentally accepting a one-frame false detection.
    """
    last_id = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from webcam.")

        detected_ids, corners, ids = detect_marker_in_frame(frame, dictionary)
        if detected_ids:
            print("Camera sees marker IDs:", detected_ids)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Show a smaller preview, but keep detection on the original full-resolution frame.
        display_frame = frame

        if preview_width is not None and preview_width > 0:
            h, w = frame.shape[:2]

            if w > preview_width:
                scale = preview_width / float(w)
                preview_height = int(h * scale)

                display_frame = cv2.resize(
                    frame,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                )

        cv2.putText(
            display_frame,
            "Show marker. Known marker is accepted automatically. d = done, q = quit.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Marker detector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            return None
        if key == ord("q"):
            raise KeyboardInterrupt

        known_detected_ids = [
            marker_id for marker_id in detected_ids
            if str(marker_id) in known_markers
        ]
        unknown_detected_ids = [
            marker_id for marker_id in detected_ids
            if str(marker_id) not in known_markers
        ]

        if unknown_detected_ids:
            print(f"Detected unknown marker ID(s): {unknown_detected_ids}. Add them to marker_detection.yaml.")

        if not known_detected_ids:
            last_id = None
            count = 0
            continue

        marker_id = known_detected_ids[0]

        if marker_id == last_id:
            count += 1
        else:
            last_id = marker_id
            count = 1

        if count >= stable_frames:
            return marker_id


def classify_detected_hardware(detected_markers):
    gearbox_markers = [
        marker for marker in detected_markers
        if marker["type"] == "gearbox"
    ]

    instrument_markers = [
        marker for marker in detected_markers
        if marker["type"] == "instrument"
    ]

    if len(gearbox_markers) > 1:
        raise RuntimeError("Multiple gearbox markers detected. Use only one gearbox marker.")

    if len(instrument_markers) > 1:
        raise RuntimeError("Multiple instrument markers detected. Use only one instrument marker.")

    gearbox = gearbox_markers[0] if gearbox_markers else None
    instrument = instrument_markers[0] if instrument_markers else None

    if gearbox is None and instrument is None:
        mode = "motor_only"

    elif gearbox is not None and instrument is None:
        mode = "gearbox_only"

    elif gearbox is not None and instrument is not None:
        mode = "full_setup"

    else:
        raise RuntimeError("Instrument marker detected without gearbox marker.")

    return {
        "mode": mode,
        "gearbox": gearbox,
        "instrument": instrument,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--preview-width", type=int, default=960, help="Width of the marker detector preview window. Detection still uses the full camera frame.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default="detected_hardware.json")
    args = parser.parse_args()

    config = load_marker_config(args.config)
    known_markers = {
        str(marker_id): marker_info
        for marker_id, marker_info in config["markers"].items()
    }

    dictionary_name = config.get("aruco_dictionary", "DICT_4X4_50")
    dictionary = get_aruco_dictionary(dictionary_name)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    detected_marker_ids = []
    detected_markers = []

    print("")
    print("Marker detector started.")
    print("Show a marker to the webcam.")
    print("Press 'd' in the webcam window when done without scanning more markers.")
    print("Press 'q' to cancel.")
    print("")

    try:
        while True:
            marker_id = wait_for_stable_marker(
                cap=cap,
                dictionary=dictionary,
                known_markers=known_markers,
                preview_width=args.preview_width,
            )

            if marker_id is None:
                break

            if marker_id in detected_marker_ids:
                print(f"Marker {marker_id} already scanned.")
            else:
                marker_info = known_markers[str(marker_id)]
                marker_info = dict(marker_info)
                marker_info["id"] = marker_id

                detected_marker_ids.append(marker_id)
                detected_markers.append(marker_info)

                print("")
                print(f"Detected marker {marker_id}:")
                print(f"  type: {marker_info.get('type')}")
                print(f"  name: {marker_info.get('name')}")
                print(f"  description: {marker_info.get('description', '')}")
                print("")

            answer = input("Scan another marker? [y/n]: ").strip().lower()
            if answer != "y":
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Marker detection cancelled.")
        return

    finally:
        cap.release()
        cv2.destroyAllWindows()

    result = classify_detected_hardware(detected_markers)

    output_data = {
        "timestamp": time.time(),
        "detected_marker_ids": detected_marker_ids,
        "detected_markers": detected_markers,
        "mode": result["mode"],
        "gearbox": result["gearbox"],
        "instrument": result["instrument"],
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("")
    print("Detected hardware result:")
    print(json.dumps(output_data, indent=2))
    print("")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()