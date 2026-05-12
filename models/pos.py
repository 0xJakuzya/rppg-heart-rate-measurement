import math
import numpy as np
from src import config
from src.utils import bandpass_filter, detrend

def pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    """POS-Wang baseline for RGB traces shaped [time, 3]."""
    n_samples = rgb.shape[0]
    output = np.zeros(n_samples, dtype=np.float64)
    window = math.ceil(config.BASELINE_WINDOW_SEC * fps)
    for end in range(n_samples):
        start = end - window
        if start < 0:
            continue
        segment = rgb[start:end, :].astype(np.float64)
        normalized = (segment / (np.mean(segment, axis=0) + config.BASELINE_EPS)).T
        projected = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64) @ normalized
        std_ratio = np.std(projected[0]) / (np.std(projected[1]) + config.BASELINE_EPS)
        pulse = projected[0] + std_ratio * projected[1]
        pulse -= pulse.mean()
        output[start:end] += pulse
    output = detrend(output)
    bvp = bandpass_filter(output, fps, config.CHEBY_LO, config.CHEBY_HI)
    bvp -= bvp.mean()
    bvp /= bvp.std() + config.BASELINE_EPS
    return bvp.astype(np.float32)

class POS:
    def __init__(self, fps: float):
        self.fps = fps

    def run(self, rgb: np.ndarray) -> np.ndarray:
        return pos(rgb, self.fps)
