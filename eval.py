from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.pose_metrics import ankle_mpjpe, joint_group_mpjpe, overall_mpjpe, per_joint_mpjpe, wrist_mpjpe
from train import (
    build_collate_fn,
    build_datasets,
    build_model,
    collect_model_metadata,
    extract_pose,
    load_config,
    move_to_device,
    resolve_device,
    validate_checkpoint_metadata,
)


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


def run_evaluation(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint["metadata"], collect_model_metadata(config))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_dataset = build_datasets(config, checkpoint_dir=output_root)
    loader = DataLoader(
        test_dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 0)),
        collate_fn=build_collate_fn(config),
    )

    preds = []
    targets = []
    metas = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            x, target, meta = _unpack_batch_with_meta(batch)
            pred = extract_pose(model(x))
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            metas.extend(meta)

    if not preds:
        raise ValueError("test loader is empty")
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    metrics = compute_evaluation_metrics(
        pred_all,
        target_all,
        metas,
        pck_thresholds=config.get("evaluation", {}).get("pck_thresholds", (0.05, 0.1, 0.2)),
    )
    metrics["split"] = "target_test"
    metrics["sample_count"] = int(pred_all.shape[0])

    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(
        output_root / "predictions.npz",
        pred=pred_all.numpy(),
        target=target_all.numpy(),
        env=np.asarray([int(meta["env"]) for meta in metas], dtype=np.int16),
        action=np.asarray([int(meta["action"]) for meta in metas], dtype=np.int16),
        frame=np.asarray([int(meta["frame"]) for meta in metas], dtype=np.int16),
    )
    return metrics


def compute_evaluation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metas: list[dict[str, int]],
    *,
    pck_thresholds: Any,
) -> dict[str, Any]:
    distances = torch.linalg.vector_norm(pred - target, dim=-1)
    metrics: dict[str, Any] = {
        "overall_mpjpe": float(overall_mpjpe(pred, target)),
        "per_joint_mpjpe": per_joint_mpjpe(pred, target).tolist(),
        "joint_group_mpjpe": {key: float(value) for key, value in joint_group_mpjpe(pred, target).items()},
        "wrist_mpjpe": float(wrist_mpjpe(pred, target)),
        "ankle_mpjpe": float(ankle_mpjpe(pred, target)),
        "per_action": _group_mpjpe(distances, metas, "action"),
        "per_environment": _group_mpjpe(distances, metas, "env"),
    }
    for threshold in pck_thresholds:
        threshold_value = float(threshold)
        metrics[f"pck@{threshold_value:g}"] = float((distances <= threshold_value).float().mean())
    return metrics


def _group_mpjpe(distances: torch.Tensor, metas: list[dict[str, int]], key: str) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for index, meta in enumerate(metas):
        groups.setdefault(str(int(meta[key])), []).append(index)
    return {
        group: float(distances[torch.as_tensor(indices, dtype=torch.long)].mean())
        for group, indices in sorted(groups.items(), key=lambda item: int(item[0]))
    }


def _unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"]
    return batch[0], batch[1]


def _unpack_batch_with_meta(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
) -> tuple[Any, torch.Tensor, list[dict[str, int]]]:
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"], batch.get("meta", [])
    if len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PhysCSI-Pose checkpoint on the target test split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_evaluation(load_config(args.config), checkpoint_path=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
