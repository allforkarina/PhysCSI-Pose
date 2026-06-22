from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_arrays(root: Path) -> tuple[Path, Path, Path]:
    x = np.zeros((4, 3, 2, 4), dtype=np.float32)
    x[0] = 0.0
    x[1] = 10.0
    x[2] = 1000.0
    x[3] = 2000.0
    y = np.zeros((4, 17, 2), dtype=np.float32)
    for sample in range(4):
        y[sample, :, 0] = float(sample)
        y[sample, :, 1] = float(sample + 1)
    x_path = root / "X_amp_resampled.npy"
    y_path = root / "Y_2d_clean.npy"
    meta_path = root / "meta.npz"
    np.save(x_path, x)
    np.save(y_path, y)
    np.savez(
        meta_path,
        env=np.array([1, 1, 2, 2], dtype=np.int16),
        subject=np.array([1, 1, 2, 2], dtype=np.int16),
        action=np.array([1, 2, 1, 2], dtype=np.int16),
        frame=np.array([1, 2, 1, 2], dtype=np.int16),
        sequence_id=np.array([0, 0, 1, 1], dtype=np.int32),
    )
    return x_path, y_path, meta_path


def test_compute_normalization_stats_uses_only_training_indices(tmp_path: Path) -> None:
    from dataset.normalization import compute_normalization_stats

    x_path, _, _ = write_arrays(tmp_path)
    x = np.load(x_path, mmap_mode="r")

    stats = compute_normalization_stats(x, frame_indices=[0, 1])

    assert stats.mean.shape == (1, 3, 2, 1)
    assert stats.std.shape == (1, 3, 2, 1)
    assert np.allclose(stats.mean, 5.0)
    assert np.allclose(stats.std, 5.0)


def test_normalization_stats_roundtrip(tmp_path: Path) -> None:
    from dataset.normalization import NormalizationStats, load_normalization_stats, save_normalization_stats

    stats = NormalizationStats(
        mean=np.ones((1, 3, 2, 1), dtype=np.float32),
        std=np.full((1, 3, 2, 1), 2.0, dtype=np.float32),
        mode="per_antenna_subcarrier",
    )
    path = tmp_path / "normalization_stats.npz"

    save_normalization_stats(path, stats)
    loaded = load_normalization_stats(path)

    assert loaded.mode == "per_antenna_subcarrier"
    assert np.array_equal(loaded.mean, stats.mean)
    assert np.array_equal(loaded.std, stats.std)


def test_resampled_pose_dataset_reuses_training_stats_for_validation(tmp_path: Path) -> None:
    from dataset.normalization import compute_normalization_stats
    from dataset.resampled_pose_dataset import ResampledPoseDataset

    x_path, y_path, meta_path = write_arrays(tmp_path)
    x = np.load(x_path, mmap_mode="r")
    train_stats = compute_normalization_stats(x, frame_indices=[0, 1])

    dataset = ResampledPoseDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
        frame_indices=[2],
        normalization_stats=train_stats,
    )

    x_item, y_item, meta = dataset[0]

    assert len(dataset) == 1
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)
    assert x_item.shape == (3, 2, 4)
    assert y_item.shape == (17, 2)
    assert torch.allclose(x_item, torch.full((3, 2, 4), 199.0))
    assert torch.allclose(y_item[:, 0], torch.full((17,), 2.0))
    assert meta == {"env": 2, "subject": 2, "action": 1, "frame": 1, "sequence_id": 1}
