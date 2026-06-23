from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_build_memmap_preserves_h36m17_xy_order_and_raw_coordinate_range(tmp_path: Path) -> None:
    from scripts.build_memmap import process_trial

    csi_trial = tmp_path / "dataset" / "A09" / "S34"
    wifi_dir = csi_trial / "wifi-csi"
    wifi_dir.mkdir(parents=True)
    scipy.io.savemat(
        wifi_dir / "frame001.mat",
        {"CSIamp": np.ones((3, 114, 10), dtype=np.float32)},
    )

    gt_dir = tmp_path / "ground_truth_npy"
    gt_dir.mkdir()
    gt = np.zeros((1, 17, 3), dtype=np.float32)
    gt[0, :, 0] = np.linspace(-1.75, 4.40, 17, dtype=np.float32)
    gt[0, :, 1] = np.linspace(-2.50, 3.25, 17, dtype=np.float32)
    gt[0, :, 2] = 1.0
    np.save(gt_dir / "E04_S34_A09.npy", gt)

    result = process_trial(csi_trial, pose_min=-0.8, pose_max=0.8, gt_dir=gt_dir)

    assert result is not None
    assert result["keypoints"].shape == (1, 17, 2)
    np.testing.assert_allclose(result["keypoints"][0], gt[0, :, :2])
    assert result["keypoints"][0, :, 0].max() > 0.8
    assert result["keypoints"][0, :, 0].min() < -0.8


def test_memmap_dataset_and_collate_expose_keypoints_as_h36m17(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset
    from dataloader import memmap_collate_fn

    data_dir = tmp_path / "memmap"
    data_dir.mkdir()
    csi = np.zeros((2, 64, 3, 114), dtype=np.float32)
    keypoints = np.arange(2 * 17 * 2, dtype=np.float32).reshape(2, 17, 2)
    for name in ("csi_gminmax.npy", "csi_gzscore.npy", "csi_zscore.npy"):
        np.save(data_dir / name, csi)
    np.save(data_dir / "ground_truth.npy", keypoints)
    np.savez(
        data_dir / "meta.npz",
        environment=np.array(["env1", "env1"]),
        sample=np.array(["S01", "S01"]),
        action=np.array(["A01", "A01"]),
        frame_idx=np.array([1, 2]),
    )

    dataset = MemmapDataset(data_dir, split="all")
    item = dataset[0]
    batch = memmap_collate_fn([dataset[0], dataset[1]])

    assert "keypoints" in item
    assert "kpts18" not in item
    assert item["keypoints"].shape == (17, 2)
    assert batch["keypoints"].shape == (2, 17, 2)
    assert batch["csi_amplitude"].shape == (2, 3, 114, 64)
    assert torch.equal(batch["keypoints"], torch.from_numpy(keypoints))


def test_skeleton_contract_is_human36m17() -> None:
    from models.skeleton import H36M17_BONE_EDGES, H36M17_JOINT_NAMES, NUM_H36M_KEYPOINTS, build_normalized_adjacency

    adjacency = build_normalized_adjacency()

    assert NUM_H36M_KEYPOINTS == 17
    assert len(H36M17_JOINT_NAMES) == 17
    assert adjacency.shape == (17, 17)
    assert H36M17_JOINT_NAMES[0] == "pelvis"
    assert H36M17_JOINT_NAMES[16] == "right_wrist"
    assert all(0 <= start < 17 and 0 <= end < 17 for start, end in H36M17_BONE_EDGES)
