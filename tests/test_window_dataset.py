from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.window_dataset import WindowMemmapPoseDataset, build_or_load_window_index


def write_sequence_cache(root: Path) -> None:
    n = 10
    x = np.zeros((n, 12, 10, 114), dtype=np.float32)
    for i in range(n):
        x[i] = float(i)
    np.save(root / "X_all.npy", x)
    np.save(root / "Y_all.npy", np.zeros((n, 17, 2), dtype=np.float32))
    np.save(root / "Conf_all.npy", np.ones((n, 17), dtype=np.float32))
    np.savez(
        root / "meta.npz",
        global_idx=np.arange(n, dtype=np.int64),
        env_id=np.ones(n, dtype=np.uint8),
        subject_id=np.ones(n, dtype=np.uint8),
        action_id=np.ones(n, dtype=np.uint8),
        frame_id=np.arange(n, dtype=np.uint16),
        seq_id=np.zeros(n, dtype=np.uint16),
    )


def test_build_window_index_counts_contiguous_windows(tmp_path: Path):
    write_sequence_cache(tmp_path)
    index_path = tmp_path / "window_index" / "train.npz"

    index = build_or_load_window_index(
        memmap_root=tmp_path,
        index_path=index_path,
        protocol="source_only",
        env_id=1,
        split="train",
        window_length=4,
        stride=1,
        rebuild=True,
    )

    assert index["start_global_idx"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert index["seq_id"].tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert index_path.exists()


def test_build_window_index_reuses_existing_cache(tmp_path: Path):
    write_sequence_cache(tmp_path)
    index_path = tmp_path / "window_index" / "train.npz"
    index_path.parent.mkdir(parents=True)
    np.savez(
        index_path,
        start_global_idx=np.asarray([2], dtype=np.int64),
        seq_id=np.asarray([0], dtype=np.int64),
        start_frame=np.asarray([2], dtype=np.int64),
        window_length=np.asarray(4, dtype=np.int64),
        stride=np.asarray(1, dtype=np.int64),
    )

    index = build_or_load_window_index(
        memmap_root=tmp_path,
        index_path=index_path,
        protocol="source_only",
        env_id=1,
        split="train",
        window_length=4,
        stride=1,
        rebuild=False,
    )

    assert index["start_global_idx"].tolist() == [2]


def test_window_dataset_returns_fixed_shapes_and_feature_selection(tmp_path: Path):
    write_sequence_cache(tmp_path)
    ds = WindowMemmapPoseDataset(
        memmap_root=tmp_path,
        index_path=tmp_path / "window_index" / "train.npz",
        protocol="source_only",
        env_id=1,
        split="train",
        window_length=4,
        stride=1,
        features=["l_norm", "f_sub"],
        rebuild_index=True,
    )

    item = ds[0]

    assert item["x"].shape == (4, 6, 10, 114)
    assert item["y"].shape == (4, 17, 2)
    assert item["conf"].shape == (4, 17)
    assert item["global_idx"].tolist() == [0, 1, 2, 3]
