from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
        env_2_source_train_frame_indices=np.array([0], dtype=np.int64),
        env_2_source_val_frame_indices=np.array([1], dtype=np.int64),
        env_2_target_test_frame_indices=np.array([2, 3], dtype=np.int64),
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
    assert first["split_counts"] == {"source_train": 1, "source_val": 1, "target_test": 2}


def test_training_does_not_use_target_environment_for_model_selection(tmp_path: Path) -> None:
    from train import build_datasets

    data_root = tmp_path / "data"
    output_root = tmp_path / "checkpoints"
    data_root.mkdir()
    _write_training_arrays(data_root)
    config = _training_config(data_root, output_root)

    train_dataset, val_dataset, test_dataset = build_datasets(config, checkpoint_dir=output_root)

    assert train_dataset.frame_indices.tolist() == [0]
    assert val_dataset.frame_indices.tolist() == [1]
    assert test_dataset.frame_indices.tolist() == [2, 3]


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


def test_move_to_device_supports_tensor_collections_and_optimizer_state() -> None:
    from train import move_optimizer_to_device, move_to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = move_to_device(({"raw": torch.zeros(1), "A3": torch.ones(1)}, torch.ones(1), [{"frame": 1}]), device)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = model(torch.ones(1, 1)).sum()
    loss.backward()
    optimizer.step()

    move_optimizer_to_device(optimizer, device)

    assert moved[0]["raw"].device == device
    assert moved[1].device == device
    assert moved[2] == [{"frame": 1}]
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device == device


def test_base_pose_auxiliary_loss_contributes_when_available() -> None:
    from train import training_step

    class ToyPoseModel(nn.Module):
        def forward(self, x: torch.Tensor, *, return_intermediates: bool = False):
            output = {
                "pose": torch.zeros(x.shape[0], 17, 2),
                "P_base": torch.ones(x.shape[0], 17, 2),
            }
            return output if return_intermediates else output["pose"]

    config = {"losses": {"coordinate_l1": 1.0, "base_coordinate_l1": 0.5, "bone_length": 0.0}}
    result = training_step(ToyPoseModel(), (torch.zeros(2, 1), torch.zeros(2, 17, 2)), config)

    assert torch.allclose(result["loss"], torch.tensor(0.5))


def test_run_training_writes_diagnostics_and_reproducibility_metadata(tmp_path: Path) -> None:
    from train import run_training

    data_root = tmp_path / "data"
    output_root = tmp_path / "checkpoints"
    data_root.mkdir()
    _write_training_arrays(data_root)
    config = _training_config(data_root, output_root)
    config["reproducibility"] = {"seed": 123}

    run_training(config)

    assert (output_root / "diagnostics.jsonl").exists()
    assert (output_root / "resolved_config.yaml").exists()
    metadata = (output_root / "run_metadata.json").read_text(encoding="utf-8")
    assert "git_commit" in metadata
    assert "seed" in metadata
