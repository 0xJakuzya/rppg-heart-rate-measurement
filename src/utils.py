import logging
import random
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import scipy.signal
import scipy.signal as scipy_signal
import scipy.sparse
import torch
from src import config

logger = logging.getLogger(__name__)

def setup_logging(level=config.LOG_LEVEL):
    logging.basicConfig(level=level, format=config.LOG_FORMAT, force=True)

def fix_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(preferred=None):
    requested = preferred or config.DEVICE
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)

def build_model(model_name=config.MODEL_NAME):
    if model_name == "physnet":
        from models.physnet import PhysNet
        return PhysNet()
    raise ValueError(f"Unknown model: {model_name}")

def load_model(model_path, device, model_name=config.MODEL_NAME):
    model = build_model(model_name).to(device)
    state_dict = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_physnet_model(model_path, device, model_name=config.MODEL_NAME):
    return load_model(model_path, device, model_name)

def load_ppg_sync(path):
    data = np.atleast_2d(np.loadtxt(path))
    values = data[:, 0].astype(np.float32)
    total_time = float(data[:, 1].sum()) if data.shape[1] > 1 else 0.0
    fps = len(values) / total_time if total_time > 0 else 100.0
    return values, fps

def resample_ppg(ppg, ppg_fps, video_fps, n_frames):
    resampled = scipy_signal.resample(ppg, int(len(ppg) / ppg_fps * video_fps))
    if len(resampled) >= n_frames:
        return resampled[:n_frames].astype(np.float32)
    pad = np.zeros(n_frames - len(resampled), dtype=np.float32)
    return np.concatenate([resampled, pad]).astype(np.float32)

def extract_mean_rgb(frame, roi):
    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = frame[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)

def extract_rois_rgb(frame, forehead, left_cheek, right_cheek):
    roi_weights = ((forehead, config.ROI_RGB_WEIGHTS[0]), (left_cheek, config.ROI_RGB_WEIGHTS[1]), (right_cheek, config.ROI_RGB_WEIGHTS[2]),)
    weighted = np.zeros(3)
    total_weight = 0.0
    for roi, weight in roi_weights:
        sample = extract_mean_rgb(frame, roi)
        if sample is not None:
            weighted += sample * weight
            total_weight += weight
    if total_weight == 0:
        return None
    bgr = weighted / total_weight
    return np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32)

def extract_multi_rois_patches(detector, frame, landmarks, patch_size=None):
    patches = detector.get_multi_roi_patches(frame, landmarks, patch_size=patch_size)
    return np.stack(patches, axis=0)

def extract_mean_rgb_from_patches(patches):
    if len(patches) == 0:
        return None
    pixels = []
    for patch in patches:
        valid_mask = np.any(patch > 0, axis=-1)
        if np.any(valid_mask):
            pixels.append(patch[valid_mask])
    if not pixels:
        return None
    mean_bgr = np.concatenate(pixels, axis=0).mean(axis=0)
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float32)

def make_patch_preview(patches, scale=None, margin=None):
    scale = scale or config.ROI_PREVIEW_SCALE
    margin = margin or config.ROI_PREVIEW_MARGIN
    if len(patches) == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    resized = [cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for patch in patches]
    patch_h, patch_w = resized[0].shape[:2]
    cols = min(config.ROI_PREVIEW_COLUMNS, len(resized))
    rows = int(np.ceil(len(resized) / cols))

    canvas_h = rows * patch_h + (rows + 1) * margin
    canvas_w = cols * patch_w + (cols + 1) * margin
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for index, patch in enumerate(resized):
        row = index // cols
        col = index % cols
        y0 = margin + row * (patch_h + margin)
        x0 = margin + col * (patch_w + margin)
        canvas[y0 : y0 + patch_h, x0 : x0 + patch_w] = patch
        cv2.putText(canvas, str(index), (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 
                    config.ROI_LABEL_FONT_SCALE, config.DETECTED_COLOR, config.FONT_THICKNESS,)
    return canvas

def normalize_patch_window(patches):
    patches = patches.astype(np.float32)
    if patches.max(initial=0.0) > 2.0:
        patches = patches / 255.0
    valid_mask = np.any(patches > 0, axis=-1, keepdims=True)
    if not np.any(valid_mask):
        return patches.astype(np.float32)
    valid_pixels = patches[valid_mask.repeat(3, axis=-1)].reshape(-1, 3)
    mean = valid_pixels.mean(axis=0)
    std = valid_pixels.std(axis=0)
    normalized = (patches - mean.reshape(1, 1, 1, 1, 3)) / (std.reshape(1, 1, 1, 1, 3) + config.EPS)
    normalized *= valid_mask.astype(np.float32)
    return normalized.astype(np.float32)

def normalize_signal(signal):
    signal = signal.astype(np.float32)
    signal -= signal.mean()
    std = signal.std()
    if std > config.EPS:
        signal /= std
    return signal.astype(np.float32)

def apply_frame_diff(patches,eps=config.EPS):
    if isinstance(patches, torch.Tensor):
        diff = torch.zeros_like(patches)
        current = patches[1:]
        previous = patches[:-1]
        diff[1:] = (current - previous) / (current.abs() + previous.abs() + eps)
        return diff
    diff_np = np.zeros_like(patches, dtype=np.float32)
    current_np = patches[1:]
    previous_np = patches[:-1]
    diff_np[1:] = (current_np - previous_np) / (np.abs(current_np) + np.abs(previous_np) + eps)
    return diff_np

def prepare_patch_window(patch_window, use_frame_diff, normalize_raw=True):
    patches = patch_window.astype(np.float32)
    if normalize_raw:
        patches = normalize_patch_window(patches)
    elif patches.max(initial=0.0) > 2.0:
        patches = patches / 255.0
    if use_frame_diff:
        patches = apply_frame_diff(patches)
    patches = np.transpose(patches, (0, 1, 4, 2, 3)).copy()
    return torch.from_numpy(patches).unsqueeze(0)

def physnet_bvp(model, rgb_buf, device):
    x = torch.from_numpy(rgb_buf).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
    return normalize_signal(output[0].cpu().numpy().astype(np.float32))

def detrend(signal, lam=config.DETREND_LAMBDA):
    n = len(signal)
    if n < 3:
        return signal.astype(np.float64)
    identity = np.eye(n)
    ones = np.ones(n)
    second_difference = scipy.sparse.spdiags(
        np.array([ones, -2 * ones, ones]), [0, 1, 2], n - 2, n
    ).toarray()
    smoothing = np.linalg.inv(identity + lam**2 * second_difference.T @ second_difference)
    return (identity - smoothing) @ signal.astype(np.float64)

def bandpass_filter(signal, fps, lo, hi):
    nyquist = fps / 2.0
    if nyquist <= 0:
        raise ValueError("fps must be positive")
    low = lo / nyquist
    high = min(hi / nyquist, 0.999)
    if low <= 0 or low >= high:
        return signal.astype(np.float32)
    signal64 = signal.astype(np.float64)
    if config.FILTER_TYPE == "chebyshev2":
        b, a = scipy.signal.cheby2(config.CHEBY_ORDER, config.CHEBY_RS, [low, high], btype="bandpass",)
    else:
        b, a = scipy.signal.butter(1, [low, high], btype="bandpass")
    return scipy.signal.filtfilt(b, a, signal64).astype(np.float32)

def process_bvp(rgb_buf, fps):
    signal = rgb_buf[:, 1].astype(np.float64)
    signal = detrend(signal)
    if len(signal) >= int(fps * 2):
        signal = bandpass_filter(signal, fps, config.CHEBY_LO, config.CHEBY_HI)
    return normalize_signal(signal)

def empty_spectral_metrics():
    return {
        "hr_bpm": float("nan"),
        "peak_ratio": float("nan"),
        "peak_power_fraction": float("nan"),
        "spectral_entropy": float("nan"),
    }

def spectral_metrics(signal, fps, lo=config.HR_LO_HZ, hi=config.HR_HI_HZ, eps=config.FFT_EPS):
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2:
        return empty_spectral_metrics()
    signal = signal - signal.mean()
    n = 2 ** int(np.ceil(np.log2(len(signal))))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal, n=n)) ** 2
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return empty_spectral_metrics()
    band_freqs = freqs[mask]
    band_power = power[mask]
    order = np.argsort(band_power)[::-1]
    top_power = float(band_power[order[0]])
    second_power = float(band_power[order[1]]) if len(order) > 1 else 0.0
    total_power = float(band_power.sum())
    probabilities = band_power / (total_power + eps)
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    if len(probabilities) > 1:
        entropy /= float(np.log(len(probabilities)))
    return {
        "hr_bpm": float(band_freqs[order[0]] * 60.0),
        "peak_ratio": top_power / (second_power + eps),
        "peak_power_fraction": top_power / (total_power + eps),
        "spectral_entropy": entropy,
    }

def fft_hr(signal, fps):
    return spectral_metrics(signal, fps)["hr_bpm"]

def estimate_hr(bvp, fps):
    if len(bvp) < int(fps * 2):
        return None
    hr = fft_hr(bvp, fps)
    return hr if np.isfinite(hr) else None

def hr_metrics(predictions, targets, fps):
    pairs = []
    for prediction, target in zip(predictions, targets):
        pred_hr = fft_hr(prediction, fps)
        target_hr = fft_hr(target, fps)
        if np.isfinite(pred_hr) and np.isfinite(target_hr):
            pairs.append((pred_hr, target_hr))
    if not pairs:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    pred_hr, true_hr = np.array(pairs).T
    error = pred_hr - true_hr
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "n": int(len(pairs)),
    }
