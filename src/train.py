import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from models.loss import ShiftLoss
from src import config
from src.dataset import DatasetSplit, RPPGDataset, build_dataloaders, describe_dataset, discover_window_files, get_patient_id, split_by_patient
from src.utils import build_model, fft_hr, fix_seed, hr_metrics, resolve_device, setup_logging

logger = logging.getLogger(__name__)

def build_loss(loss_name, fps):
    if loss_name == "shiftloss":
        return ShiftLoss(max_shift_sec=config.SHIFT_LOSS_MAX_SHIFT_SEC, fps=fps)
    raise ValueError(f"Unknown loss: {loss_name}")

def limit_split_by_patient(files, split):
    if config.TRAIN_MAX_TRAIN_PATIENTS is None and config.TRAIN_MAX_VAL_PATIENTS is None:
        return split
    rng = random.Random(config.TRAIN_SEED)
    train_patients = sorted(split.train_patients)
    val_patients = sorted(split.val_patients)
    rng.shuffle(train_patients)
    rng.shuffle(val_patients)
    if config.TRAIN_MAX_TRAIN_PATIENTS is not None:
        train_patients = train_patients[: config.TRAIN_MAX_TRAIN_PATIENTS]
    if config.TRAIN_MAX_VAL_PATIENTS is not None:
        val_patients = val_patients[: config.TRAIN_MAX_VAL_PATIENTS]
    train_set = set(train_patients)
    val_set = set(val_patients)
    train_indices = [index for index in split.train_indices if get_patient_id(files[index]) in train_set]
    val_indices = [index for index in split.val_indices if get_patient_id(files[index]) in val_set]
    return DatasetSplit(train_indices, val_indices, train_set, val_set)

def train_one_epoch(model, loader, optimizer, criterion, device, fps, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}/{total_epochs} train",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
    )
    for patches, ppg in progress:
        patches = patches.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(patches)
        loss = criterion(pred, ppg)
        loss.backward()
        optimizer.step()
        batch_size = patches.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        predictions.append(pred.detach().cpu().numpy())
        targets.append(ppg.detach().cpu().numpy())
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")
    metrics = hr_metrics(np.concatenate(predictions), np.concatenate(targets), fps)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, fps, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}/{total_epochs} val  ",
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
    )
    for patches, ppg in progress:
        patches = patches.to(device, non_blocking=True)
        ppg = ppg.to(device, non_blocking=True)
        pred = model(patches)
        loss = criterion(pred, ppg)
        batch_size = patches.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        predictions.append(pred.cpu().numpy())
        targets.append(ppg.cpu().numpy())
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")
    pred_np = np.concatenate(predictions)
    target_np = np.concatenate(targets)
    metrics = hr_metrics(pred_np, target_np, fps)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics, pred_np, target_np

def save_plots(history, best_pred, best_target, fps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [record["train_loss"] for record in history], label="train", marker="o")
    axes[0].plot(epochs, [record["val_loss"] for record in history], label="val", marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [record["train_hr_mae"] for record in history], label="train", marker="o")
    axes[1].plot(epochs, [record["val_hr_mae"] for record in history], label="val", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("HR MAE, BPM")
    axes[1].set_title("HR MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [record["val_hr_rmse"] for record in history], color="C2", marker="o")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("HR RMSE, BPM")
    axes[2].set_title("Validation HR RMSE")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=config.PLOT_DPI)
    plt.close(fig)

    n_show = min(config.TRAIN_PLOT_SAMPLES, len(best_pred))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    timeline = np.arange(best_pred.shape[1]) / fps
    for index in range(n_show):
        axes[index].plot(timeline, best_target[index], label="target PPG", color="C0", lw=1.4)
        axes[index].plot(timeline, best_pred[index], label="predicted BVP", color="C3", lw=1.4, alpha=0.85)
        pred_hr = fft_hr(best_pred[index], fps)
        target_hr = fft_hr(best_target[index], fps)
        axes[index].set_title(f"sample {index} | pred HR={pred_hr:.1f} target HR={target_hr:.1f} BPM")
        axes[index].grid(alpha=0.3)
        if index == 0:
            axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("time, s")
    fig.tight_layout()
    fig.savefig(out_dir / "best_predictions.png", dpi=config.PLOT_DPI)
    plt.close(fig)

    pred_hr = np.array([fft_hr(pred, fps) for pred in best_pred])
    true_hr = np.array([fft_hr(target, fps) for target in best_target])
    valid = np.isfinite(pred_hr) & np.isfinite(true_hr)
    pred_hr = pred_hr[valid]
    true_hr = true_hr[valid]
    if len(pred_hr) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lo = float(min(pred_hr.min(), true_hr.min())) - 5
    hi = float(max(pred_hr.max(), true_hr.max())) + 5
    axes[0].scatter(true_hr, pred_hr, alpha=0.5, s=14)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("target HR, BPM")
    axes[0].set_ylabel("predicted HR, BPM")
    axes[0].set_title(f"Validation HR scatter (n={len(pred_hr)})")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)

    mean_hr = (pred_hr + true_hr) / 2
    diff_hr = pred_hr - true_hr
    bias = float(diff_hr.mean())
    sd = float(diff_hr.std())
    axes[1].scatter(mean_hr, diff_hr, alpha=0.5, s=14)
    axes[1].axhline(bias, color="C3", lw=1.2, label=f"bias={bias:+.2f}")
    axes[1].axhline(bias + 1.96 * sd, color="C3", ls="--", lw=1, label=f"+/-1.96 SD")
    axes[1].axhline(bias - 1.96 * sd, color="C3", ls="--", lw=1)
    axes[1].set_xlabel("mean HR, BPM")
    axes[1].set_ylabel("pred - target, BPM")
    axes[1].set_title("Bland-Altman")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hr_scatter.png", dpi=config.PLOT_DPI)
    plt.close(fig)

def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

def training_config_summary(data_dir):
    return {
        "data_dir": str(data_dir),
        "model": config.MODEL_NAME,
        "loss": config.TRAIN_LOSS,
        "epochs": config.TRAIN_EPOCHS,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "lr": config.TRAIN_LR,
        "val_split": config.TRAIN_VAL_SPLIT,
        "num_workers": config.TRAIN_NUM_WORKERS,
        "seed": config.TRAIN_SEED,
        "device": config.TRAIN_DEVICE,
        "use_frame_diff": config.TRAIN_USE_FRAME_DIFF,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": config.EARLY_STOPPING_MIN_DELTA,
    }

def create_run_dir() -> Path:
    run_name = config.RUN_NAME or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def run(run_dir=None):
    setup_logging()
    fix_seed(config.TRAIN_SEED)
    out_dir = Path(run_dir) if run_dir is not None else create_run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / config.MODEL_FILENAME
    data_dir = Path(config.TRAIN_DATA_DIR)
    fps = float(config.FPS_TARGET)
    device = resolve_device(config.TRAIN_DEVICE)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("[1/5] Discover windows in %s", data_dir)
    files = discover_window_files(data_dir)
    if not files:
        raise ValueError(f"No .npz windows found in {data_dir}")
    describe_dataset(files)
    dataset = RPPGDataset(files, use_frame_diff=config.TRAIN_USE_FRAME_DIFF)

    logger.info("[2/5] Subject-stratified split")
    split = split_by_patient(files, val_split=config.TRAIN_VAL_SPLIT, seed=config.TRAIN_SEED)
    split = limit_split_by_patient(files, split)
    if not split.train_indices or not split.val_indices:
        raise ValueError("Train/validation split is empty. Use at least two patients for training.")
    logger.info("train: %s windows | %s patients", len(split.train_indices), len(split.train_patients))
    logger.info("val: %s windows | %s patients", len(split.val_indices), len(split.val_patients))

    logger.info("[3/5] Build dataloaders (batch_size=%s, workers=%s)", config.TRAIN_BATCH_SIZE, config.TRAIN_NUM_WORKERS)
    train_loader, val_loader = build_dataloaders(dataset=dataset, split=split, batch_size=config.TRAIN_BATCH_SIZE,
                                                num_workers=config.TRAIN_NUM_WORKERS, pin_memory=device.type == "cuda",)
    logger.info("train batches: %s | val batches: %s", len(train_loader), len(val_loader))

    logger.info("[4/5] Build model")
    model = build_model(config.MODEL_NAME).to(device)
    if config.TRAIN_PRETRAINED_PATH is not None:
        state_dict = torch.load(config.TRAIN_PRETRAINED_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("loaded pretrained weights: %s", config.TRAIN_PRETRAINED_PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN_EPOCHS)
    criterion = build_loss(config.TRAIN_LOSS, fps)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    logger.info("device: %s | model: %s | loss: %s | params: %s", device, config.MODEL_NAME, config.TRAIN_LOSS, f"{n_params:,}",)
    logger.info("output: %s", out_dir)

    logger.info("[5/5] Train (%s epochs)", config.TRAIN_EPOCHS)
    logger.info("%5s %11s %10s %10s %9s %10s %10s %7s", "epoch", "train_loss", "val_loss", "train_mae", "val_mae", "val_rmse", "lr", "time")

    history = []
    best_mae = float("inf")
    best_pred: np.ndarray | None = None
    best_target: np.ndarray | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None

    for epoch in range(1, config.TRAIN_EPOCHS + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            fps,
            epoch,
            config.TRAIN_EPOCHS,
        )
        val_metrics, val_pred, val_target = eval_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            fps,
            epoch,
            config.TRAIN_EPOCHS,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "train_hr_mae": float(train_metrics["mae"]),
            "val_hr_mae": float(val_metrics["mae"]),
            "val_hr_rmse": float(val_metrics["rmse"]),
            "lr": float(lr_now),
            "time": float(elapsed),
        }
        history.append(record)

        val_mae = float(val_metrics["mae"])
        improved = best_pred is None or (
            np.isfinite(val_mae) and val_mae < best_mae - config.EARLY_STOPPING_MIN_DELTA
        )
        marker = "* " if improved else "  "
        if improved:
            best_mae = val_mae
            best_pred = val_pred
            best_target = val_target
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "%s%4d %11.4f %10.4f %10.2f %9.2f %10.2f %10.2e %6.1fs",
            marker,
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["mae"],
            val_metrics["mae"],
            val_metrics["rmse"],
            lr_now,
            elapsed,
        )

        save_json(out_dir / "history.json", history)

        if (
            config.EARLY_STOPPING_PATIENCE > 0
            and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            stop_reason = (
                f"val_hr_mae plateau: no improvement >= "
                f"{config.EARLY_STOPPING_MIN_DELTA:.4f} for "
                f"{config.EARLY_STOPPING_PATIENCE} epochs"
            )
            logger.info("early stopping: %s", stop_reason)
            break

    logger.info("-" * 100)
    logger.info("best val HR MAE: %.2f BPM", best_mae)

    summary = {
        "best_val_hr_mae": best_mae,
        "n_epochs": config.TRAIN_EPOCHS,
        "n_epochs_completed": len(history),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "fps": fps,
        "window": config.CNN_WINDOW,
        "window_sec": config.CNN_WINDOW / fps,
        "n_train_windows": len(split.train_indices),
        "n_val_windows": len(split.val_indices),
        "n_train_patients": len(split.train_patients),
        "n_val_patients": len(split.val_patients),
        "model_params": n_params,
        "model_path": str(model_path),
        "config": training_config_summary(data_dir),
    }
    save_json(out_dir / "summary.json", summary)

    if best_pred is not None and best_target is not None:
        save_plots(history, best_pred, best_target, fps, out_dir)
        logger.info("plots and summary saved: %s", out_dir)

    return {
        "run_dir": out_dir,
        "model_path": model_path,
        "best_val_hr_mae": best_mae,
        "summary_path": out_dir / "summary.json",
    }


if __name__ == "__main__":
    run()
