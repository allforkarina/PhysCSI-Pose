from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.normalization import NormalizationStats


class ResampledPoseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, int]]]):
    def __init__(
        self,
        *,
        x_path: Path,
        y_path: Path,
        meta_path: Path,
        frame_indices: Iterable[int],
        normalization_stats: NormalizationStats,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.meta = np.load(meta_path, allow_pickle=False)
        self.frame_indices = np.asarray(list(frame_indices), dtype=np.int64)
        if self.frame_indices.size == 0:
            raise ValueError("frame_indices must not be empty")
        if self.x.ndim != 4:
            raise ValueError(f"expected X layout [sample, antenna, subcarrier, time], got {self.x.shape}")
        if self.y.ndim != 3 or self.y.shape[1:] != (17, 2):
            raise ValueError(f"expected Y layout [sample,17,2], got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/Y sample count mismatch: {self.x.shape[0]} != {self.y.shape[0]}")
        if int(self.frame_indices.max()) >= self.x.shape[0] or int(self.frame_indices.min()) < 0:
            raise IndexError("frame_indices contain out-of-range sample indices")
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        frame_index = int(self.frame_indices[index])
        x = np.asarray(self.x[frame_index], dtype=np.float32)
        y = np.asarray(self.y[frame_index], dtype=np.float32)
        x = (x - self.normalization_stats.mean[0]) / self.normalization_stats.std[0]
        return (
            torch.as_tensor(x.copy(), dtype=torch.float32),
            torch.as_tensor(y.copy(), dtype=torch.float32),
            self._meta_for_frame(frame_index),
        )

    def _meta_for_frame(self, frame_index: int) -> dict[str, int]:
        return {
            "env": int(self.meta["env"][frame_index]),
            "subject": int(self.meta["subject"][frame_index]),
            "action": int(self.meta["action"][frame_index]),
            "frame": int(self.meta["frame"][frame_index]),
            "sequence_id": int(self.meta["sequence_id"][frame_index]),
        }
