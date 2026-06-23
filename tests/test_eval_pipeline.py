from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_arrays(root: Path) -> None:
    rng = np.random.default_rng(321)
    np.save(root / "X_amp_resampled.npy", rng.normal(size=(4, 3, 114, 64)).astype(np.float32))
    np.save(root / "Y_2d_clean.npy", rng.normal(scale=0.1, size=(4, 17, 2)).astype(np.float32))
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
        env_2_source_train_frame_indices=np.array([0], dtype=np.int64),
        env_2_source_val_frame_indices=np.array([1], dtype=np.int64),
        env_2_target_test_frame_indices=np.array([2, 3], dtype=np.int64),
    )


def _config(data_root: Path, checkpoint_root: Path) -> dict:
    return {
        "metadata": {"skeleton_name": "human36m17", "input_layout": "antenna,subcarrier,time"},
        "data": {
            "root": str(data_root),
            "x_file": "X_amp_resampled.npy",
            "y_file": "Y_2d_clean.npy",
            "meta_file": "meta.npz",
            "split_index_file": "split_index.npz",
            "eval_env": 2,
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
            "checkpoint_dir": str(checkpoint_root),
        },
        "evaluation": {"pck_thresholds": [0.05, 0.1]},
        "logging": {"metrics": [], "diagnostics": []},
    }


def test_run_evaluation_loads_checkpoint_and_exports_target_metrics(tmp_path: Path) -> None:
    from eval import run_evaluation
    from train import run_training

    data_root = tmp_path / "data"
    checkpoint_root = tmp_path / "checkpoints"
    output_root = tmp_path / "eval"
    data_root.mkdir()
    _write_arrays(data_root)
    config = _config(data_root, checkpoint_root)
    run_training(config)

    result = run_evaluation(config, checkpoint_path=checkpoint_root / "best.pt", output_dir=output_root)

    assert result["split"] == "target_test"
    assert result["sample_count"] == 2
    assert "pck@0.05" in result
    assert "per_action" in result
    assert "per_environment" in result
    metrics_path = output_root / "metrics.json"
    predictions_path = output_root / "predictions.npz"
    assert metrics_path.exists()
    assert predictions_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["split"] == "target_test"
