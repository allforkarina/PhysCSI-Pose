from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as eval_module
import train as train_module
from eval import _compute_diagnostics


def test_compute_diagnostics_reports_per_joint_std_and_group() -> None:
    targets = np.zeros((3, 17, 2), dtype=np.float32)
    predictions = np.zeros((3, 17, 2), dtype=np.float32)
    targets[:, 16, 0] = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    predictions[:, 16, 0] = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    diagnostic = _compute_diagnostics([predictions], [targets])
    wrist_row = diagnostic["joint_rows"][16]

    assert wrist_row["joint_name"] == "right_wrist"
    assert wrist_row["joint_group"] == "distal_limb"
    assert np.isclose(wrist_row["gt_std"], np.sqrt(wrist_row["gt_var"]))
    assert np.isclose(wrist_row["pred_std"], np.sqrt(wrist_row["pred_var"]))
    assert np.isclose(wrist_row["std_ratio"], wrist_row["pred_std"] / wrist_row["gt_std"])
    assert diagnostic["overall"]["overall_std_ratio"] > 0.0


def _minimal_eval_args(tmp_path: Path, *, eval_split: str) -> Namespace:
    return Namespace(
        dataset_root=str(tmp_path / "memmap"),
        checkpoint=str(tmp_path / "checkpoint.pth"),
        output_dir=str(tmp_path / "eval"),
        batch_size=2,
        num_workers=0,
        device="cpu",
        eval_envs=None,
        eval_split=eval_split,
        exclude_indices=None,
        feature_viz=False,
        pose_viz=False,
        num_action_samples=3,
        output_format="both",
        figure_width=None,
        figure_height=None,
    )


def test_eval_main_uses_test_split_by_default_and_labels_metrics_as_test(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    seen: dict[str, str] = {}

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, split, envs=None):
            seen["split"] = split

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            raise AssertionError("run_evaluation is mocked and should not iterate")

    monkeypatch.setattr(eval_module, "parse_args", lambda: _minimal_eval_args(tmp_path, eval_split="test"))
    monkeypatch.setattr(eval_module, "select_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(eval_module, "load_checkpoint_model", lambda checkpoint, device: object())
    monkeypatch.setattr(eval_module, "MemmapDataset", FakeDataset)
    monkeypatch.setattr(eval_module, "run_evaluation", lambda model, loader, device: {
        "overall": {"mpjpe": 1.0},
        "joint_rows": [],
        "action_rows": [],
        "environment_rows": [],
        "diagnostic": {"overall": {}, "joint_rows": []},
    })
    monkeypatch.setattr(eval_module, "_write_csv", lambda path, rows: None)

    eval_module.main()

    assert seen["split"] == "test"
    assert "--- Test Metrics ---" in capsys.readouterr().out


def test_eval_main_uses_explicit_all_split_and_labels_metrics_as_all(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    seen: dict[str, str] = {}

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, split, envs=None):
            seen["split"] = split

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            raise AssertionError("run_evaluation is mocked and should not iterate")

    monkeypatch.setattr(eval_module, "parse_args", lambda: _minimal_eval_args(tmp_path, eval_split="all"))
    monkeypatch.setattr(eval_module, "select_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(eval_module, "load_checkpoint_model", lambda checkpoint, device: object())
    monkeypatch.setattr(eval_module, "MemmapDataset", FakeDataset)
    monkeypatch.setattr(eval_module, "run_evaluation", lambda model, loader, device: {
        "overall": {"mpjpe": 1.0},
        "joint_rows": [],
        "action_rows": [],
        "environment_rows": [],
        "diagnostic": {"overall": {}, "joint_rows": []},
    })
    monkeypatch.setattr(eval_module, "_write_csv", lambda path, rows: None)

    eval_module.main()

    assert seen["split"] == "all"
    assert "--- All Metrics ---" in capsys.readouterr().out


def test_train_and_eval_reject_removed_split_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "source_only",
            "--dataset-root",
            "data",
            "--split-strategy",
            "frame_random",
        ],
    )
    with pytest.raises(SystemExit):
        train_module.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval.py",
            "--dataset-root",
            "data",
            "--checkpoint",
            "model.pth",
            "--split-strategy",
            "frame_random",
        ],
    )
    with pytest.raises(SystemExit):
        eval_module.parse_args()
