from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smooth_l1_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    conf: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    assert pred_xy.shape == target_xy.shape, (
        f"pred and target shape mismatch: {pred_xy.shape} vs {target_xy.shape}"
    )
    assert pred_xy.shape[:-1] == conf.shape, (
        f"conf shape {conf.shape} does not match pred shape {pred_xy.shape}"
    )

    valid = (conf > 0).to(dtype=pred_xy.dtype)
    loss = F.smooth_l1_loss(pred_xy, target_xy, reduction="none", beta=beta)
    loss = loss * valid.unsqueeze(-1)
    denom = valid.sum() * pred_xy.shape[-1]
    return loss.sum() / (denom + eps)
