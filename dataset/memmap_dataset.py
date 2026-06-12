from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dataset.features import selected_feature_channels
from dataset.splits import source_only_subjects


class MemmapPoseDataset:
    def __init__(
        self,
        root: str | Path,
        *,
        protocol: str,
        env_id: int,
        split: str,
        features: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if protocol == "finetune":
            raise NotImplementedError("finetune protocol is not implemented yet")
        if protocol != "source_only":
            raise ValueError(f"protocol must be 'source_only', got {protocol!r}")

        self.root = Path(root)
        self.feature_channels = selected_feature_channels(features)
        self.x_all = np.load(self.root / "X_all.npy", mmap_mode="r")
        self.y_all = np.load(self.root / "Y_all.npy", mmap_mode="r")
        self.conf_all = np.load(self.root / "Conf_all.npy", mmap_mode="r")
        self.meta = np.load(self.root / "meta.npz")

        subjects = np.array(source_only_subjects(env_id=env_id, split=split), dtype=np.uint8)
        mask = (self.meta["env_id"] == env_id) & np.isin(self.meta["subject_id"], subjects)
        self.indices = self.meta["global_idx"][mask].astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        global_idx = int(self.indices[item])
        meta_item = {
            "global_idx": global_idx,
            "env_id": int(self.meta["env_id"][global_idx]),
            "subject_id": int(self.meta["subject_id"][global_idx]),
            "action_id": int(self.meta["action_id"][global_idx]),
            "frame_id": int(self.meta["frame_id"][global_idx]),
            "seq_id": int(self.meta["seq_id"][global_idx]),
        }
        return {
            "x": self.x_all[global_idx, self.feature_channels],
            "y": self.y_all[global_idx],
            "conf": self.conf_all[global_idx],
            "meta": meta_item,
        }
