
#!/usr/bin/env python3
import sys
import cv2
import cv2

VIDEO = "/home/leanne/sequence_DOF4_trial_03.mp4"
FRAME_TO_SELECT = 10 #180
SCALE = 0.35


def scale_roi_to_original(roi, scale):
    x, y, w, h = roi
    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_SELECT)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read frame")

small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

marker_small = cv2.selectROI("Select MARKER ROI, then press ENTER", small)
MARKER_ROI = scale_roi_to_original(marker_small, SCALE)

shaft_small = cv2.selectROI("Select SHAFT ROI, then press ENTER", small)
SHAFT_ROI = scale_roi_to_original(shaft_small, SCALE)

cv2.destroyAllWindows()
cap.release()

print("MARKER_ROI =", MARKER_ROI)
print("SHAFT_ROI =", SHAFT_ROI)
