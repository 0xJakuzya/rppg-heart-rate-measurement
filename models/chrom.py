import numpy as np
from src import config
from src.utils import bandpass_filter, detrend

def chrom(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM baseline for RGB traces shaped [time, 3]."""
    n_samples = rgb.shape[0]
    window = max(int(config.BASELINE_WINDOW_SEC * fps), 2)
    output = np.zeros(n_samples, dtype=np.float64)
    for end in range(window, n_samples + 1):
        segment = rgb[end - window : end, :].astype(np.float64)
        mean_rgb = np.mean(segment, axis=0) + config.BASELINE_EPS
        normalized = segment / mean_rgb
        red, green, blue = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        x_signal = 3 * red - 2 * green
        y_signal = 1.5 * red + green - 1.5 * blue
        if window >= 9:
            x_signal = bandpass_filter(x_signal, fps, config.CHEBY_LO, config.CHEBY_HI)
            y_signal = bandpass_filter(y_signal, fps, config.CHEBY_LO, config.CHEBY_HI)
        alpha = np.std(x_signal) / (np.std(y_signal) + config.BASELINE_EPS)
        pulse = x_signal - alpha * y_signal
        pulse -= pulse.mean()
        output[end - window : end] += pulse
    output = detrend(output)
    bvp = bandpass_filter(output, fps, config.CHEBY_LO, config.CHEBY_HI)
    bvp -= bvp.mean()
    std = bvp.std()
    if std < config.EPS:
        return np.zeros(n_samples, dtype=np.float32)
    bvp /= std
    return bvp.astype(np.float32)

class CHROM:
    def __init__(self, fps: float):
        self.fps = fps

    def run(self, rgb: np.ndarray) -> np.ndarray:
        return chrom(rgb, self.fps)
