from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_normalization_dataset(data_dir: Path) -> None:
    data_dir.mkdir()
    shape = (2, 64, 3, 114)
    np.save(data_dir / "csi_gminmax.npy", np.full(shape, 1.0, dtype=np.float32))
    np.save(data_dir / "csi_gzscore.npy", np.full(shape, 2.0, dtype=np.float32))
    np.save(data_dir / "csi_zscore.npy", np.full(shape, 3.0, dtype=np.float32))
    np.save(data_dir / "ground_truth.npy", np.zeros((2, 17, 2), dtype=np.float32))
    np.savez(
        data_dir / "meta.npz",
        environment=np.array(["env1", "env1"]),
        sample=np.array(["S01", "S01"]),
        action=np.array(["A01", "A01"]),
        frame_idx=np.array([1, 2]),
    )


@pytest.mark.parametrize(
    ("normalization", "expected"),
    [
        ("global_minmax", 1.0),
        ("global_zscore", 2.0),
        ("per_sample_zscore", 3.0),
    ],
)
def test_memmap_dataset_selects_normalization_file(
    tmp_path: Path,
    normalization: str,
    expected: float,
) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    dataset = MemmapDataset(data_dir, split="all", normalization=normalization)

    assert dataset.normalization == normalization
    assert np.all(dataset[0]["csi"].numpy() == expected)


def test_memmap_dataset_rejects_unknown_normalization(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        MemmapDataset(data_dir, split="all", normalization="unknown")


@pytest.mark.parametrize(
    "normalization",
    ["global_minmax", "global_zscore", "per_sample_zscore"],
)
def test_loader_factories_propagate_normalization(
    tmp_path: Path,
    normalization: str,
) -> None:
    from dataloader import (
        create_few_shot_data_loader,
        create_memmap_data_loader,
        create_memmap_data_loaders,
    )

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    loader = create_memmap_data_loader(
        data_dir=data_dir,
        split="all",
        batch_size=1,
        normalization=normalization,
    )
    split_loaders = create_memmap_data_loaders(
        data_dir=data_dir,
        batch_size=1,
        normalization=normalization,
    )
    few_shot_loader, _ = create_few_shot_data_loader(
        data_dir=data_dir,
        target_envs=("env1",),
        few_shot_subjects=1,
        few_shot_frames=1,
        batch_size=1,
        normalization=normalization,
    )

    assert loader.dataset.normalization == normalization
    assert all(item.dataset.normalization == normalization for item in split_loaders.values())
    assert few_shot_loader.dataset.dataset.normalization == normalization


@pytest.mark.parametrize(
    "normalization",
    ["global_minmax", "global_zscore", "per_sample_zscore"],
)
def test_train_cli_accepts_normalization(normalization: str) -> None:
    from train import parse_args

    argv = [
        "train.py",
        "--mode", "source_only",
        "--dataset-root", "data/mmfi_pose",
        "--source-envs", "env1",
        "--normalization", normalization,
    ]
    with patch("sys.argv", argv):
        args = parse_args()

    assert args.normalization == normalization


def test_train_config_defaults_to_global_minmax() -> None:
    from train import TrainConfig

    config = TrainConfig(dataset_root="data/mmfi_pose")

    assert config.normalization == "global_minmax"
    assert asdict(config)["normalization"] == "global_minmax"


def test_checkpoint_records_normalization(tmp_path: Path) -> None:
    from train import TrainConfig, save_checkpoint

    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "checkpoint.pth"
    config = TrainConfig(
        dataset_root="data/mmfi_pose",
        normalization="global_zscore",
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        epoch=1,
        best_metric=0.5,
        config=config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["train_config"]["normalization"] == "global_zscore"
