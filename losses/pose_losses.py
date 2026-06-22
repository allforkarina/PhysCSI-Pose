from __future__ import annotations

from collections.abc import Sequence

import torch

from dataset.h36m17 import H36M17_EDGES


def coordinate_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pose_pair(pred, target)
    return torch.mean(torch.abs(pred - target))


def bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    edges: Sequence[tuple[int, int]] = H36M17_EDGES,
) -> torch.Tensor:
    _validate_pose_pair(pred, target)
    if not edges:
        raise ValueError("edges must not be empty")
    losses = []
    for left, right in edges:
        pred_len = torch.linalg.vector_norm(pred[:, left] - pred[:, right], dim=-1)
        target_len = torch.linalg.vector_norm(target[:, left] - target[:, right], dim=-1)
        losses.append(torch.abs(pred_len - target_len))
    return torch.mean(torch.stack(losses, dim=0))


def _validate_pose_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"expected pose tensors [batch,joint,xy], got {tuple(pred.shape)}")
