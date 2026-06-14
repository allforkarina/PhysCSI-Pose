from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch

from engine.losses import masked_smooth_l1_loss
from engine.metrics import aggregate_window_predictions_mean, compute_pose_metrics


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    use_scaler = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            pred = model(batch["x"])
            loss = masked_smooth_l1_loss(pred, batch["y"], batch["conf"])

        if scaler is None:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return {"train_loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_enabled: bool,
    pck_thresholds: tuple[float, ...],
    loss_prefix: str = "val",
) -> dict[str, float | list[float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    confs: list[torch.Tensor] = []
    global_indices: list[torch.Tensor] = []

    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast_context(device, amp_enabled):
            pred = model(batch["x"])
            loss = masked_smooth_l1_loss(pred, batch["y"], batch["conf"])

        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        preds.append(pred.detach().cpu())
        targets.append(batch["y"].detach().cpu())
        confs.append(batch["conf"].detach().cpu())
        if "global_idx" in batch:
            global_indices.append(batch["global_idx"].detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    conf_all = torch.cat(confs, dim=0)
    if global_indices:
        global_idx_all = torch.cat(global_indices, dim=0)
        pred_all, unique_idx = aggregate_window_predictions_mean(pred_all, global_idx_all)
        flat_idx = global_idx_all.reshape(-1)
        flat_target = target_all.reshape(-1, target_all.shape[-2], target_all.shape[-1])
        flat_conf = conf_all.reshape(-1, conf_all.shape[-1])
        first_positions = torch.stack([(flat_idx == idx).nonzero(as_tuple=False)[0, 0] for idx in unique_idx])
        target_all = flat_target[first_positions]
        conf_all = flat_conf[first_positions]

    metrics = compute_pose_metrics(
        pred_all,
        target_all,
        conf_all,
        pck_thresholds=pck_thresholds,
    )
    metrics[f"{loss_prefix}_loss"] = total_loss / max(total_items, 1)
    return metrics
