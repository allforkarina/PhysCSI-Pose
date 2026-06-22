from __future__ import annotations

import torch

from dataset.h36m17 import H36M17_JOINT_GROUPS


def per_joint_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_pose_pair(pred, target)
    return torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=0)


def overall_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return per_joint_mpjpe(pred, target).mean()


def joint_group_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    per_joint = per_joint_mpjpe(pred, target)
    return {
        name: per_joint[torch.as_tensor(indices, device=per_joint.device)].mean()
        for name, indices in H36M17_JOINT_GROUPS.items()
    }


def wrist_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return joint_group_mpjpe(pred, target)["wrist"]


def ankle_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return joint_group_mpjpe(pred, target)["ankle"]


def _validate_pose_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"expected pose tensors [batch,joint,xy], got {tuple(pred.shape)}")
