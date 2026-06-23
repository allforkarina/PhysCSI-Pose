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
    chunk_size: int = 1024,
) -> NormalizationStats:
    indices = np.asarray(list(frame_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("frame_indices must not be empty")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if x.ndim != 4:
        raise ValueError(f"expected X layout [sample, antenna, subcarrier, time], got {x.shape}")

    total_count = 0
    total_sum = np.zeros((x.shape[1], x.shape[2]), dtype=np.float64)
    total_square_sum = np.zeros((x.shape[1], x.shape[2]), dtype=np.float64)
    for start in range(0, indices.size, int(chunk_size)):
        chunk_indices = indices[start : start + int(chunk_size)]
        chunk = np.asarray(x[chunk_indices], dtype=np.float32)
        if chunk.ndim != 4:
            raise ValueError(f"expected X layout [sample, antenna, subcarrier, time], got {chunk.shape}")
        total_count += int(chunk.shape[0] * chunk.shape[3])
        total_sum += chunk.sum(axis=(0, 3), dtype=np.float64)
        total_square_sum += np.square(chunk, dtype=np.float64).sum(axis=(0, 3), dtype=np.float64)

    mean_2d = total_sum / float(total_count)
    variance_2d = np.maximum(total_square_sum / float(total_count) - np.square(mean_2d), 0.0)
    mean = mean_2d[None, :, :, None]
    std = np.sqrt(variance_2d)[None, :, :, None]
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
