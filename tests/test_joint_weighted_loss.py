from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import JOINT_LOSS_PRESETS, compute_losses, parse_args


def test_uniform_joint_loss_matches_original_l1_coordinate_loss() -> None:
    prediction = torch.zeros(2, 18, 2)
    target = torch.randn(2, 18, 2)

    losses = compute_losses(
        prediction,
        target,
        bone_loss_weight=0.0,
        joint_loss_preset="uniform",
        lower_limb_weight=1.0,
    )

    assert torch.allclose(losses["coord_loss"], F.l1_loss(prediction, target))
    assert JOINT_LOSS_PRESETS == ("uniform", "lower_limb")


def test_lower_limb_weighted_loss_emphasizes_lower_limb_errors() -> None:
    target = torch.zeros(1, 18, 2)
    prediction = torch.zeros(1, 18, 2)
    prediction[:, 10, :] = 1.0

    uniform = compute_losses(
        prediction,
        target,
        bone_loss_weight=0.0,
        joint_loss_preset="uniform",
        lower_limb_weight=1.0,
    )
    weighted = compute_losses(
        prediction,
        target,
        bone_loss_weight=0.0,
        joint_loss_preset="lower_limb",
        lower_limb_weight=2.0,
    )

    assert weighted["coord_loss"] > uniform["coord_loss"]


def test_lower_limb_weight_one_keeps_original_loss_scale() -> None:
    prediction = torch.randn(2, 18, 2)
    target = torch.randn(2, 18, 2)

    uniform = compute_losses(
        prediction,
        target,
        bone_loss_weight=0.0,
        joint_loss_preset="uniform",
        lower_limb_weight=1.0,
    )
    weighted = compute_losses(
        prediction,
        target,
        bone_loss_weight=0.0,
        joint_loss_preset="lower_limb",
        lower_limb_weight=1.0,
    )

    assert torch.allclose(weighted["coord_loss"], uniform["coord_loss"])


def test_parse_args_accepts_lower_limb_weighted_loss_options() -> None:
    argv = [
        "train.py",
        "--mode", "finetune",
        "--dataset-root", "data/mmfi_pose",
        "--target-envs", "env2",
        "--finetune-from", "outputs/source/best_val_mpjpe.pth",
        "--joint-loss-preset", "lower_limb",
        "--lower-limb-weight", "1.5",
    ]

    with patch("sys.argv", argv):
        args = parse_args()

    assert args.joint_loss_preset == "lower_limb"
    assert args.lower_limb_weight == 1.5
