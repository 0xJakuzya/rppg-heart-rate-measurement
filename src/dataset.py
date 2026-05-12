import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src import config
from src.utils import apply_frame_diff

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    train_patients: set[str]
    val_patients: set[str]

def discover_window_files(data_dir: str | Path) -> list[str]:
    files = sorted(str(path) for path in Path(data_dir).rglob("*.npz"))
    if not files:
        logger.warning("No .npz windows found in %s.", data_dir)
    return files

def describe_dataset(files: Sequence[str]) -> dict[str, object]:
    if not files:
        return {"windows": 0, "patients": 0}
    with np.load(files[0]) as sample:
        patch_shape = sample["patches"].shape
        ppg_shape = sample["ppg"].shape
    patients = {get_patient_id(file) for file in files}
    description = {
        "windows": len(files),
        "patients": len(patients),
        "patch_shape": patch_shape,
        "ppg_shape": ppg_shape,
    }
    logger.info("windows: %s", description["windows"])
    logger.info("patients: %s", description["patients"])
    logger.info("sample patch shape: %s", description["patch_shape"])
    logger.info("sample ppg shape: %s", description["ppg_shape"])
    return description

def split_by_patient(files: Sequence[str], val_split: float, seed: int) -> DatasetSplit:
    patient_to_indices: dict[str, list[int]] = {}
    for index, file in enumerate(files):
        patient_to_indices.setdefault(get_patient_id(file), []).append(index)
    patient_ids = sorted(patient_to_indices)
    if not patient_ids:
        return DatasetSplit([], [], set(), set())
    random.Random(seed).shuffle(patient_ids)
    if len(patient_ids) == 1:
        return DatasetSplit(
            train_indices=indices_for_patients(patient_to_indices, set(patient_ids)),
            val_indices=[],
            train_patients=set(patient_ids),
            val_patients=set(),
        )
    val_count = int(len(patient_ids) * val_split)
    val_count = min(max(1, val_count), len(patient_ids) - 1)
    val_patients = set(patient_ids[:val_count])
    train_patients = set(patient_ids[val_count:])
    return DatasetSplit(
        train_indices=indices_for_patients(patient_to_indices, train_patients),
        val_indices=indices_for_patients(patient_to_indices, val_patients),
        train_patients=train_patients,
        val_patients=val_patients,
    )

def indices_for_patients(patient_to_indices: dict[str, list[int]], patients: set[str]) -> list[int]:
    return sorted(index for patient_id in patients for index in patient_to_indices[patient_id])

def get_patient_id(file: str | Path) -> str:
    return Path(file).stem.split("_", 1)[0]

def build_dataloaders(dataset: Dataset, split: DatasetSplit, batch_size: int, 
                      num_workers: int, pin_memory: bool = True,) -> tuple[DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(dataset, split.train_indices),
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        Subset(dataset, split.val_indices),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader

class RPPGDataset(Dataset):
    def __init__(self, files: Sequence[str | Path], use_frame_diff: bool = False, eps: float = config.EPS,):
        self.files = [str(file) for file in files]
        self.use_frame_diff = use_frame_diff
        self.eps = eps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as sample:
            patches_np = sample["patches"]
            ppg_np = sample["ppg"]
        patches = torch.from_numpy(patches_np).float().permute(0, 1, 4, 2, 3).contiguous()
        if self.use_frame_diff:
            patches = apply_frame_diff(patches, self.eps)
        ppg = torch.from_numpy(ppg_np).float()
        return patches, ppg
