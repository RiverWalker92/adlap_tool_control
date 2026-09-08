#!/usr/bin/env python3

import cv2
import argparse


SCALE = 0.35

def scale_roi_to_original(roi, scale):
    x, y, w, h = roi
    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


def draw_text(frame, text, y):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 2")
    parser.add_argument("--scale", type=float, default=SCALE)
    args = parser.parse_args()

    camera = 0  # dit is /dev/video0

    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0")

    # Even wat frames weggooien zodat de webcam kan opstarten
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Could not read frame from /dev/video0")

    print("Opened /dev/video0 correctly")
    print("Frame shape:", frame.shape)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    # Optional: force webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Live preview started.")
    print("Press SPACE to freeze current frame and select ROIs.")
    print("Press q to quit.")

    frozen_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from webcam")

        preview = frame.copy()
        draw_text(preview, "SPACE = freeze/select ROIs", 40)
        draw_text(preview, "q = quit", 80)

        small_preview = cv2.resize(preview, None, fx=args.scale, fy=args.scale)
        cv2.imshow("Live webcam preview", small_preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord(" "):
            frozen_frame = frame.copy()
            break

    cap.release()
    cv2.destroyWindow("Live webcam preview")

    small = cv2.resize(frozen_frame, None, fx=args.scale, fy=args.scale)

    led_small = cv2.selectROI("Select LED ROI, then press ENTER", small)
    LED_ROI = scale_roi_to_original(led_small, args.scale)

    marker_small = cv2.selectROI("Select MARKER ROI, then press ENTER", small)
    MARKER_ROI = scale_roi_to_original(marker_small, args.scale)

    shaft_small = cv2.selectROI("Select SHAFT ROI, then press ENTER", small)
    SHAFT_ROI = scale_roi_to_original(shaft_small, args.scale)

    cv2.destroyAllWindows()

    print()
    print("Selected ROIs in original video coordinates:")
    print("LED_ROI    =", LED_ROI)
    print("MARKER_ROI =", MARKER_ROI)
    print("SHAFT_ROI  =", SHAFT_ROI)

    # Save debug image with boxes
    debug = frozen_frame.copy()

    x, y, w, h = LED_ROI
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(debug, "LED_ROI", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    x, y, w, h = MARKER_ROI
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(debug, "MARKER_ROI", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    x, y, w, h = SHAFT_ROI
    cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(debug, "SHAFT_ROI", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out_path = "live_roi_debug.png"
    cv2.imwrite(out_path, debug)
    print(f"Saved debug image: {out_path}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import sys
# import cv2
# import cv2

# VIDEO = "/home/leanne/ros2_ws/test_data/automated_trials/auto_dof4_20260617_100857/auto_dof4_20260617_100857_webcam.mp4"  #"/home/leanne/sequence_DOF4_trial_03.mp4"
# FRAME_TO_SELECT = 10 #180
# SCALE = 0.35


# def scale_roi_to_original(roi, scale):
#     x, y, w, h = roi
#     return (
#         int(x / scale),
#         int(y / scale),
#         int(w / scale),
#         int(h / scale),
#     )


# cap = cv2.VideoCapture(VIDEO)
# cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_SELECT)

# ret, frame = cap.read()
# if not ret:
#     raise RuntimeError("Could not read frame")

# small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

# marker_small = cv2.selectROI("Select MARKER ROI, then press ENTER", small)
# MARKER_ROI = scale_roi_to_original(marker_small, SCALE)

# shaft_small = cv2.selectROI("Select SHAFT ROI, then press ENTER", small)
# SHAFT_ROI = scale_roi_to_original(shaft_small, SCALE)

# cv2.destroyAllWindows()
# cap.release()

# print("MARKER_ROI =", MARKER_ROI)
# print("SHAFT_ROI =", SHAFT_ROI)
