from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_training_arrays(root: Path) -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 3, 114, 64)).astype(np.float32)
    y = rng.normal(scale=0.1, size=(4, 17, 2)).astype(np.float32)
    np.save(root / "X_amp_resampled.npy", x)
    np.save(root / "Y_2d_clean.npy", y)
    np.savez(
        root / "meta.npz",
        env=np.array([1, 1, 2, 2], dtype=np.int16),
        subject=np.array([1, 1, 2, 2], dtype=np.int16),
        action=np.array([1, 2, 1, 2], dtype=np.int16),
        frame=np.array([1, 2, 1, 2], dtype=np.int16),
        sequence_id=np.array([0, 0, 1, 1], dtype=np.int32),
    )
    np.savez(
        root / "split_index.npz",
        env_2_train_frame_indices=np.array([0, 1], dtype=np.int64),
        env_2_eval_frame_indices=np.array([2, 3], dtype=np.int64),
    )


def _training_config(data_root: Path, output_root: Path) -> dict:
    return {
        "metadata": {"skeleton_name": "human36m17", "input_layout": "antenna,subcarrier,time"},
        "data": {
            "root": str(data_root),
            "x_file": "X_amp_resampled.npy",
            "y_file": "Y_2d_clean.npy",
            "meta_file": "meta.npz",
            "split_index_file": "split_index.npz",
            "eval_env": 2,
            "normalization": "train_split_only",
            "normalization_chunk_size": 2,
            "precompute_wavelet": False,
        },
        "model": {
            "type": "baseline",
            "num_joints": 17,
            "d_model": 32,
            "wavelet_bands": ["raw"],
            "fine_branch": False,
            "gate": False,
            "graph_refinement": False,
        },
        "losses": {"coordinate_l1": 1.0, "bone_length": 0.0},
        "trainable_groups": ["all"],
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "num_workers": 0,
            "checkpoint_dir": str(output_root),
        },
        "logging": {
            "metrics": ["overall_mpjpe", "per_joint_mpjpe", "joint_group_mpjpe", "wrist_mpjpe", "ankle_mpjpe"],
            "diagnostics": [],
        },
    }


def test_run_training_saves_best_and_resume_checkpoint(tmp_path: Path) -> None:
    from train import run_training

    data_root = tmp_path / "data"
    output_root = tmp_path / "checkpoints"
    data_root.mkdir()
    _write_training_arrays(data_root)
    config = _training_config(data_root, output_root)

    first = run_training(config)
    second = run_training(config, resume_from=output_root / "last.pt")

    assert first["start_epoch"] == 0
    assert first["epochs_completed"] == 1
    assert second["start_epoch"] == 1
    assert (output_root / "last.pt").exists()
    assert (output_root / "best.pt").exists()


def test_wavelet_collate_precomputes_bands_on_dataloader_side() -> None:
    from train import build_collate_fn

    batch = [
        (torch.randn(3, 114, 64), torch.zeros(17, 2), {"frame": 1}),
        (torch.randn(3, 114, 64), torch.zeros(17, 2), {"frame": 2}),
    ]
    collate = build_collate_fn({"data": {"precompute_wavelet": True}, "model": {"wavelet": "haar", "wavelet_bands": ["raw", "A3"]}})

    x, y, meta = collate(batch)

    assert tuple(x) == ("raw", "A3")
    assert x["raw"].shape == (2, 3, 114, 64)
    assert y.shape == (2, 17, 2)
    assert meta == [{"frame": 1}, {"frame": 2}]
