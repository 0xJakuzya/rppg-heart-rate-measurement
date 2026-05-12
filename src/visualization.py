import cv2
import numpy as np
from src import config

def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return frame
    for x, y in landmarks:
        cv2.circle(frame, (x, y), 1, config.DETECTED_COLOR, -1)
    return frame

def draw_roi(frame, roi_mask, color, alpha=0.3):
    overlay = frame.copy()
    colored = np.zeros_like(frame)
    colored[:] = color
    cv2.bitwise_and(colored, colored, mask=roi_mask, dst=colored)
    gray_mask = roi_mask > 0
    overlay[gray_mask] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[gray_mask]
    return overlay

def roi_to_mask(roi):
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def draw_status(frame, hr, status, color):
    label = f"{status}  |  HR: {hr:.0f} BPM" if hr is not None else f"{status}  |  HR: --"
    cv2.putText(frame, label, config.STATUS_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, color, config.FONT_THICKNESS)
    return frame

# def bvp_plot(bvp, width, height, hr):
#     panel = np.zeros((height, width, 3), dtype=np.uint8)
#     if len(bvp) < 2:
#         return panel
#     signal = bvp[-width:]
#     min_value = signal.min()
#     max_value = signal.max()
#     if max_value - min_value < config.EPS:
#         return panel
#     normalized = (signal - min_value) / (max_value - min_value)
#     padding = config.PLOT_PADDING
#     ys = ((1 - normalized) * (height - 2 * padding) + padding).astype(int)
#     xs = np.linspace(0, width - 1, len(ys)).astype(int)
#     for index in range(len(xs) - 1):
#         cv2.line(panel, (xs[index], ys[index]), (xs[index + 1], ys[index + 1]), config.PLOT_LINE_COLOR, 1)
#     label = f"BVP  |  HR: {hr:.0f} BPM" if hr is not None else "BVP  |  HR: --"
#     cv2.putText(panel, label, config.BVP_LABEL_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX,
#                 config.ROI_LABEL_FONT_SCALE, config.FONT_COLOR, config.FONT_THICKNESS)
#     return panel
