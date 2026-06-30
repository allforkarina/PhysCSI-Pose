from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train as train_module
from train import (
    ALIGN_LAYERS,
    ALIGN_LOSSES,
    TrainConfig,
    _flatten_alignment_feature,
    coral_loss,
    compute_alignment_loss,
    parse_args,
    run_alignment_finetune_epoch,
)


def test_coral_loss_is_zero_for_matching_features() -> None:
    feature = torch.randn(4, 3, 2, 2)

    loss = coral_loss(feature, feature.clone())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_coral_loss_increases_for_shifted_covariance() -> None:
    source = torch.randn(4, 3, 2, 2)
    target = source.clone()
    target[:, 0] = target[:, 0] * 3.0

    loss = coral_loss(source, target)

    assert loss > 0


def test_compute_alignment_loss_supports_none_and_coral() -> None:
    source = torch.randn(4, 3, 2, 2)
    target = torch.randn(4, 3, 2, 2)

    assert compute_alignment_loss(source, target, "none").item() == 0.0
    assert compute_alignment_loss(source, target, "coral").item() >= 0.0
    assert ALIGN_LOSSES == ("none", "coral")
    assert ALIGN_LAYERS == ("axial",)


def test_alignment_feature_uses_channel_statistics_for_axial_maps() -> None:
    feature = torch.randn(2, 8, 5, 4)

    flattened = _flatten_alignment_feature(feature)

    assert flattened.shape == (2, 8)
    assert torch.allclose(flattened, feature.mean(dim=(2, 3)))


class TinyAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 34, bias=False)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x).view(x.shape[0], 17, 2)


def _tiny_batch(values: torch.Tensor) -> dict:
    return {
        "csi_amplitude": values,
        "keypoints": torch.zeros(values.shape[0], 17, 2),
    }


def test_run_alignment_finetune_epoch_logs_pose_and_alignment_terms() -> None:
    model = TinyAlignmentModel()
    source_loader = [_tiny_batch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))]
    target_loader = [_tiny_batch(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))]
    config = TrainConfig(
        dataset_root="unused",
        mode="finetune_align",
        align_loss="coral",
        align_weight=0.25,
        bone_loss_weight=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_alignment_finetune_epoch(
        model,
        source_loader,
        target_loader,
        config,
        torch.device("cpu"),
        optimizer,
    )

    assert metrics["loss"] >= metrics["source_loss"]
    assert metrics["loss"] >= metrics["target_loss"]
    assert metrics["align_loss"] > 0.0


def test_parse_args_accepts_finetune_align_options() -> None:
    argv = [
        "train.py",
        "--mode", "finetune_align",
        "--dataset-root", "data/mmfi_pose",
        "--source-envs", "env1",
        "--target-envs", "env2",
        "--finetune-from", "outputs/source/best_val_mpjpe.pth",
        "--align-loss", "coral",
        "--align-layer", "axial",
        "--align-weight", "0.1",
    ]

    with patch("sys.argv", argv):
        args = parse_args()

    assert args.mode == "finetune_align"
    assert args.align_loss == "coral"
    assert args.align_layer == "axial"
    assert args.align_weight == 0.1


def test_validate_source_envs_requires_exactly_one_environment() -> None:
    with pytest.raises(ValueError, match="exactly one source environment"):
        train_module._validate_source_envs(None)
    with pytest.raises(ValueError, match="exactly one source environment"):
        train_module._validate_source_envs(("env1", "env2"))
    assert train_module._validate_source_envs(("env1",)) == ("env1",)
