from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_coordinate_l1_loss() -> None:
    from losses.pose_losses import coordinate_l1_loss

    pred = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])

    assert torch.isclose(coordinate_l1_loss(pred, target), torch.tensor(0.75))


def test_bone_length_loss_can_use_explicit_edges() -> None:
    from losses.pose_losses import bone_length_loss

    pred = torch.zeros((1, 2, 2), dtype=torch.float32)
    target = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])

    assert torch.isclose(bone_length_loss(pred, target, edges=((0, 1),)), torch.tensor(5.0))


def test_bone_length_loss_defaults_to_h36m17_edges() -> None:
    from losses.pose_losses import bone_length_loss

    pred = torch.zeros((2, 17, 2), dtype=torch.float32)
    target = torch.zeros((2, 17, 2), dtype=torch.float32)

    assert torch.isclose(bone_length_loss(pred, target), torch.tensor(0.0))


def test_per_joint_mpjpe() -> None:
    from metrics.pose_metrics import per_joint_mpjpe

    pred = torch.zeros((2, 3, 2), dtype=torch.float32)
    target = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]],
            [[0.0, 0.0], [5.0, 12.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(per_joint_mpjpe(pred, target), torch.tensor([2.5, 6.5, 5.0]))


def test_joint_group_mpjpe_uses_h36m17_groups() -> None:
    from metrics.pose_metrics import ankle_mpjpe, joint_group_mpjpe, wrist_mpjpe

    pred = torch.zeros((1, 17, 2), dtype=torch.float32)
    target = torch.zeros((1, 17, 2), dtype=torch.float32)
    target[:, 13] = torch.tensor([3.0, 4.0])
    target[:, 16] = torch.tensor([0.0, 12.0])
    target[:, 3] = torch.tensor([5.0, 12.0])
    target[:, 6] = torch.tensor([8.0, 15.0])

    groups = joint_group_mpjpe(pred, target)

    assert torch.isclose(wrist_mpjpe(pred, target), torch.tensor(8.5))
    assert torch.isclose(ankle_mpjpe(pred, target), torch.tensor(15.0))
    assert torch.isclose(groups["distal"], torch.tensor(11.75))
    assert torch.isclose(groups["torso"], torch.tensor(0.0))
