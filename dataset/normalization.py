from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    mode: str = "per_antenna_subcarrier"


def compute_normalization_stats(
    x: np.ndarray,
    frame_indices: Iterable[int],
    *,
    eps: float = 1.0e-6,
) -> NormalizationStats:
    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("frame_indices must not be empty")
    selected = np.asarray(x[indices], dtype=np.float32)
    if selected.ndim != 4:
        raise ValueError(f"expected X layout [sample, antenna, subcarrier, time], got {selected.shape}")

    mean = selected.mean(axis=(0, 3), keepdims=True)
    std = selected.std(axis=(0, 3), keepdims=True)
    std = np.maximum(std, float(eps)).astype(np.float32, copy=False)
    return NormalizationStats(
        mean=mean.astype(np.float32, copy=False),
        std=std,
    )


def save_normalization_stats(path: Path, stats: NormalizationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=stats.mean, std=stats.std, mode=np.asarray(stats.mode))


def load_normalization_stats(path: Path) -> NormalizationStats:
    loaded = np.load(path, allow_pickle=False)
    return NormalizationStats(
        mean=np.asarray(loaded["mean"], dtype=np.float32),
        std=np.asarray(loaded["std"], dtype=np.float32),
        mode=str(np.asarray(loaded["mode"]).item()),
    )
