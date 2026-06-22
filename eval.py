from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from metrics.pose_metrics import ankle_mpjpe, joint_group_mpjpe, overall_mpjpe, per_joint_mpjpe, wrist_mpjpe
from train import extract_pose


def evaluate_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    x, target = _unpack_batch(batch)
    output = model(x)
    pred = extract_pose(output)
    return {
        "overall_mpjpe": overall_mpjpe(pred, target),
        "per_joint_mpjpe": per_joint_mpjpe(pred, target),
        "joint_group_mpjpe": joint_group_mpjpe(pred, target),
        "wrist_mpjpe": wrist_mpjpe(pred, target),
        "ankle_mpjpe": ankle_mpjpe(pred, target),
    }


def _unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"]
    return batch[0], batch[1]
