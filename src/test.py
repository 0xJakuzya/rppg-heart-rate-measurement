from numpy import dtype, ndarray
from numpy._typing._shape import _AnyShape
from typing import Any
import logging
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from src import config
from src.face_detector import FaceDetector
from src.utils import estimate_hr, extract_multi_rois_patches, load_physnet_model, make_patch_preview, normalize_signal, prepare_patch_window, resolve_device, setup_logging
from src.video import VideoCapture
from src.visualization import bvp_plot, draw_landmarks, draw_status

logger = logging.getLogger(__name__)

@torch.no_grad()
def predict_bvp(model, patch_buffer, device,use_frame_diff,):
    patch_window = np.asarray(patch_buffer, dtype=np.float32)
    model_input = prepare_patch_window(patch_window, use_frame_diff=use_frame_diff).to(device)
    bvp = model(model_input)[0].cpu().numpy().astype(np.float32)
    return normalize_signal(bvp)

def run_tester(model_path):
    setup_logging()
    device = resolve_device(config.DEMO_DEVICE)
    resolved_model_path = Path(model_path or config.DEMO_MODEL_PATH)
    model = load_physnet_model(resolved_model_path, device, config.MODEL_NAME)
    logger.info("demo model: %s", resolved_model_path)
    camera = VideoCapture()
    camera.start()
    detector = FaceDetector()
    fps = float(config.FPS_TARGET)
    patch_buffer: deque[np.ndarray] = deque[ndarray[_AnyShape, dtype[Any]]](maxlen=config.CNN_WINDOW)
    bvp = np.array([], dtype=np.float32)
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            display = frame.copy()
            landmarks = detector.get_landmarks(frame)
            heart_rate = None
            patch_preview = None
            if landmarks is None:
                patch_buffer.clear()
                bvp = np.array([], dtype=np.float32)
                status = "NO FACE"
                color = config.NO_FACE_COLOR
            else:
                draw_landmarks(display, landmarks)
                patches = extract_multi_rois_patches(detector, frame, landmarks)
                patch_preview = make_patch_preview(patches)
                patch_buffer.append(patches)
                if len(patch_buffer) == config.CNN_WINDOW:
                    bvp = predict_bvp(
                        model,
                        patch_buffer,
                        device,
                        use_frame_diff=config.DEMO_USE_FRAME_DIFF,
                    )
                    heart_rate = estimate_hr(bvp, fps)
                status = f"DETECTED {len(patch_buffer)}/{config.CNN_WINDOW}"
                color = config.DETECTED_COLOR
            draw_status(display, heart_rate, status, color)
            if bvp.size:
                plot = bvp_plot(bvp, display.shape[1], config.PLOT_H, heart_rate)
                display[-config.PLOT_H :, :] = plot
            cv2.imshow(config.WINDOW_NAME, display)
            if patch_preview is not None:
                cv2.imshow(config.ROI_WINDOW_NAME, patch_preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tester()
