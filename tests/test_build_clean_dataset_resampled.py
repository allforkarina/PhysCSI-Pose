from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_clean_dataset.py"


def write_gt(path: Path, frames: int = 2) -> None:
    gt = np.zeros((frames, 17, 3), dtype=np.float32)
    for frame in range(frames):
        for joint in range(17):
            gt[frame, joint, 0] = 0.1 + 0.01 * joint
            gt[frame, joint, 1] = 0.2 + 0.01 * frame
            gt[frame, joint, 2] = 1.0
    np.save(path, gt)


def write_csi_frame(path: Path, offset: float) -> np.ndarray:
    raw = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5) + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"CSIamp": raw, "CSIphase": raw + 100.0})
    return raw


def run_builder(csi_root: Path, gt_root: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--csi-root",
            str(csi_root),
            "--gt-root",
            str(gt_root),
            "--output-root",
            str(output_root),
            "--expected-csi-shape",
            "3,4,5",
            "--expected-gt-shape",
            "2,17,3",
            "--expected-frames",
            "2",
            "--resample-time-steps",
            "8",
            "--gt-source-x-min",
            "0.0",
            "--gt-source-x-max",
            "1.0",
            "--gt-source-y-min",
            "0.0",
            "--gt-source-y-max",
            "1.0",
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def run_builder_with_default_gt_range(csi_root: Path, gt_root: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--csi-root",
            str(csi_root),
            "--gt-root",
            str(gt_root),
            "--output-root",
            str(output_root),
            "--expected-csi-shape",
            "3,4,5",
            "--expected-gt-shape",
            "2,17,3",
            "--expected-frames",
            "2",
            "--resample-time-steps",
            "8",
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_clean_dataset_writes_resampled_model_ready_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    csi_root = dataset_root / "dataset"
    gt_root = dataset_root / "ground_truth_npy"
    output_root = dataset_root / "clean_dataset"
    gt_root.mkdir(parents=True)
    write_gt(gt_root / "E01_S01_A01.npy")

    frame1 = write_csi_frame(csi_root / "A01" / "S01" / "wifi-csi" / "frame001.mat", offset=0.0)
    write_csi_frame(csi_root / "A01" / "S01" / "wifi-csi" / "frame002.mat", offset=10.0)

    result = run_builder(csi_root, gt_root, output_root)

    assert result.returncode == 0, result.stderr + result.stdout
    x_path = output_root / "X_amp_resampled.npy"
    assert x_path.exists()
    x_all = np.load(x_path, mmap_mode="r")
    y_all = np.load(output_root / "Y_2d_clean.npy", mmap_mode="r")
    assert x_all.shape == (2, 3, 4, 8)
    assert y_all.shape == (2, 17, 2)
    assert np.isfinite(x_all).all()
    assert not np.allclose(x_all[0, :, :, :5], frame1)

    manifest = json.loads((output_root / "clean_manifest.json").read_text(encoding="utf-8"))
    assert manifest["storage_layout"]["x_layout"] == "sample,antenna,subcarrier,time"
    assert manifest["raw_csi_shape"] == [3, 4, 5]
    assert manifest["resampled_csi_shape"] == [3, 4, 8]
    assert manifest["resample_method"] == "scipy.signal.resample"
    assert manifest["training_io"]["x_file"] == "X_amp_resampled.npy"


def test_build_clean_dataset_uses_fixed_gt_range_by_default(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    csi_root = dataset_root / "dataset"
    gt_root = dataset_root / "ground_truth_npy"
    output_root = dataset_root / "clean_dataset"
    gt_root.mkdir(parents=True)
    write_gt(gt_root / "E01_S01_A01.npy")
    write_csi_frame(csi_root / "A01" / "S01" / "wifi-csi" / "frame001.mat", offset=0.0)
    write_csi_frame(csi_root / "A01" / "S01" / "wifi-csi" / "frame002.mat", offset=10.0)

    result = run_builder_with_default_gt_range(csi_root, gt_root, output_root)

    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((output_root / "clean_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gt_source_range_policy"] == "fixed"
    assert manifest["gt_source_range"]["x_min"] == -1.0
    assert manifest["gt_source_range"]["x_max"] == 1.0


def test_build_clean_dataset_writes_source_val_and_target_test_splits(tmp_path: Path) -> None:
    from scripts.build_clean_dataset import build_splits, write_split_index_npz

    sequence_rows = [
        {"sequence_id": 0, "env": 1},
        {"sequence_id": 1, "env": 1},
        {"sequence_id": 2, "env": 2},
        {"sequence_id": 3, "env": 2},
        {"sequence_id": 4, "env": 3},
        {"sequence_id": 5, "env": 3},
    ]
    splits = build_splits(sequence_rows, source_val_fraction=0.5)
    path = tmp_path / "split_index.npz"

    write_split_index_npz(path, splits, expected_frames=2)
    arrays = np.load(path, allow_pickle=False)

    assert "env_3_source_train_frame_indices" in arrays
    assert "env_3_source_val_frame_indices" in arrays
    assert "env_3_target_test_frame_indices" in arrays
    source_train = set(arrays["env_3_source_train_frame_indices"].tolist())
    source_val = set(arrays["env_3_source_val_frame_indices"].tolist())
    target_test = set(arrays["env_3_target_test_frame_indices"].tolist())
    assert source_train
    assert source_val
    assert target_test == {8, 9, 10, 11}
    assert source_train.isdisjoint(source_val)
    assert source_train.isdisjoint(target_test)
    assert source_val.isdisjoint(target_test)
