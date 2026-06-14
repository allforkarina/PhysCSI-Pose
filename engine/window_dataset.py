from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dataset.features import selected_feature_channels
from dataset.splits import source_only_subjects


def _load_npz_as_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _eligible_frame_indices(
    meta: np.lib.npyio.NpzFile,
    *,
    protocol: str,
    env_id: int,
    split: str,
) -> np.ndarray:
    if protocol == "finetune":
        raise NotImplementedError("finetune protocol is not implemented yet")
    if protocol != "source_only":
        raise ValueError(f"protocol must be 'source_only', got {protocol!r}")

    subjects = np.array(source_only_subjects(env_id=env_id, split=split), dtype=np.uint8)
    mask = (meta["env_id"] == env_id) & np.isin(meta["subject_id"], subjects)
    return meta["global_idx"][mask].astype(np.int64)


def _build_window_index_from_meta(
    meta: np.lib.npyio.NpzFile,
    eligible_indices: np.ndarray,
    *,
    window_length: int,
    stride: int,
) -> dict[str, np.ndarray]:
    starts: list[int] = []
    seq_ids: list[int] = []
    start_frames: list[int] = []
    eligible = set(int(idx) for idx in eligible_indices.tolist())

    for seq_id in np.unique(meta["seq_id"][eligible_indices]):
        seq_mask = meta["seq_id"] == seq_id
        seq_indices = meta["global_idx"][seq_mask].astype(np.int64)
        seq_indices = np.asarray(
            [idx for idx in seq_indices.tolist() if int(idx) in eligible],
            dtype=np.int64,
        )
        seq_indices.sort()

        for offset in range(0, len(seq_indices) - window_length + 1, stride):
            window_indices = seq_indices[offset : offset + window_length]
            frame_ids = meta["frame_id"][window_indices].astype(np.int64)
            if np.all(np.diff(frame_ids) == 1):
                starts.append(int(window_indices[0]))
                seq_ids.append(int(seq_id))
                start_frames.append(int(frame_ids[0]))

    return {
        "start_global_idx": np.asarray(starts, dtype=np.int64),
        "seq_id": np.asarray(seq_ids, dtype=np.int64),
        "start_frame": np.asarray(start_frames, dtype=np.int64),
        "window_length": np.asarray(window_length, dtype=np.int64),
        "stride": np.asarray(stride, dtype=np.int64),
    }


def build_or_load_window_index(
    memmap_root: str | Path,
    index_path: str | Path,
    *,
    protocol: str,
    env_id: int,
    split: str,
    window_length: int,
    stride: int,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    index_path = Path(index_path)
    if index_path.exists() and not rebuild:
        return _load_npz_as_dict(index_path)

    memmap_root = Path(memmap_root)
    with np.load(memmap_root / "meta.npz") as meta:
        eligible_indices = _eligible_frame_indices(
            meta,
            protocol=protocol,
            env_id=env_id,
            split=split,
        )
        index = _build_window_index_from_meta(
            meta,
            eligible_indices,
            window_length=window_length,
            stride=stride,
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path, **index)
    return index


class WindowMemmapPoseDataset:
    def __init__(
        self,
        memmap_root: str | Path,
        *,
        index_path: str | Path,
        protocol: str,
        env_id: int,
        split: str,
        window_length: int,
        stride: int,
        features: list[str] | tuple[str, ...] | None = None,
        rebuild_index: bool = False,
    ) -> None:
        self.memmap_root = Path(memmap_root)
        self.feature_channels = selected_feature_channels(features)
        self.index = build_or_load_window_index(
            self.memmap_root,
            index_path,
            protocol=protocol,
            env_id=env_id,
            split=split,
            window_length=window_length,
            stride=stride,
            rebuild=rebuild_index,
        )
        self.window_length = int(np.asarray(self.index["window_length"]).item())
        self.stride = int(np.asarray(self.index["stride"]).item())
        self.x_all = np.load(self.memmap_root / "X_all.npy", mmap_mode="r")
        self.y_all = np.load(self.memmap_root / "Y_all.npy", mmap_mode="r")
        self.conf_all = np.load(self.memmap_root / "Conf_all.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.index["start_global_idx"].shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        start_global_idx = int(self.index["start_global_idx"][item])
        global_idx = np.arange(
            start_global_idx,
            start_global_idx + self.window_length,
            dtype=np.int64,
        )

        return {
            "x": self.x_all[global_idx][:, self.feature_channels],
            "y": self.y_all[global_idx],
            "conf": self.conf_all[global_idx],
            "global_idx": global_idx,
            "seq_id": int(self.index["seq_id"][item]),
            "start_frame": int(self.index["start_frame"][item]),
        }
