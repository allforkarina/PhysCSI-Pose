from __future__ import annotations

import torch


def aggregate_window_predictions_mean(
    pred_xy: torch.Tensor,
    global_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_pred = pred_xy.reshape(-1, pred_xy.shape[-2], pred_xy.shape[-1])
    flat_idx = global_idx.reshape(-1).to(device=pred_xy.device)
    unique_idx = torch.unique(flat_idx, sorted=True)

    out = torch.zeros(
        unique_idx.numel(),
        pred_xy.shape[-2],
        pred_xy.shape[-1],
        dtype=pred_xy.dtype,
        device=pred_xy.device,
    )
    count = torch.zeros(
        unique_idx.numel(),
        1,
        1,
        dtype=pred_xy.dtype,
        device=pred_xy.device,
    )
    inverse = torch.searchsorted(unique_idx, flat_idx)
    out.index_add_(0, inverse, flat_pred)
    count.index_add_(0, inverse, torch.ones_like(count[inverse]))
    return out / count.clamp_min(1.0), unique_idx


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def compute_pose_metrics(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    conf: torch.Tensor,
    *,
    pck_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
) -> dict[str, float | list[float]]:
    assert pred_xy.shape == target_xy.shape, (
        f"pred and target shape mismatch: {pred_xy.shape} vs {target_xy.shape}"
    )
    assert pred_xy.shape[:-1] == conf.shape, (
        f"conf shape {conf.shape} does not match pred shape {pred_xy.shape}"
    )

    valid = conf > 0
    dist = torch.linalg.norm(pred_xy - target_xy, dim=-1)
    denom = valid.sum().clamp_min(1)
    masked_dist = dist[valid]

    metrics: dict[str, float | list[float]] = {
        "mpjpe_norm": _safe_float(masked_dist.sum() / denom),
    }

    for threshold in pck_thresholds:
        correct = ((dist <= threshold) & valid).sum()
        key = f"pck_{threshold:.2f}"
        metrics[key] = _safe_float(correct / denom)

        per_joint = []
        for joint_id in range(pred_xy.shape[-2]):
            joint_valid = valid[..., joint_id]
            joint_denom = joint_valid.sum().clamp_min(1)
            joint_correct = ((dist[..., joint_id] <= threshold) & joint_valid).sum()
            per_joint.append(_safe_float(joint_correct / joint_denom))
        metrics[f"per_joint_{key}"] = per_joint

    pred_flat = pred_xy.reshape(-1, pred_xy.shape[-2], pred_xy.shape[-1])
    target_flat = target_xy.reshape(-1, target_xy.shape[-2], target_xy.shape[-1])
    pred_joint_std = pred_flat.std(dim=0, unbiased=False)
    gt_joint_std = target_flat.std(dim=0, unbiased=False)
    pred_joint_std_l2 = torch.linalg.norm(pred_joint_std, dim=-1)
    gt_joint_std_l2 = torch.linalg.norm(gt_joint_std, dim=-1)

    metrics["mean_joint_std"] = _safe_float(pred_joint_std_l2.mean())
    metrics["min_joint_std"] = _safe_float(pred_joint_std_l2.min())
    metrics["per_joint_std"] = [float(v) for v in pred_joint_std_l2.detach().cpu().tolist()]
    metrics["gt_mean_joint_std"] = _safe_float(gt_joint_std_l2.mean())
    metrics["gt_min_joint_std"] = _safe_float(gt_joint_std_l2.min())
    metrics["gt_per_joint_std"] = [float(v) for v in gt_joint_std_l2.detach().cpu().tolist()]
    return metrics
