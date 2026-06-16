from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.features import selected_feature_channels
from engine.loops import evaluate_one_epoch
from engine.window_dataset import WindowMemmapPoseDataset
from train import _build_model, _feature_names, _fmt_float, _setup_logging, load_config_with_overrides, resolve_device

LOGGER = logging.getLogger("physcsi_pose.eval")


def make_eval_name(checkpoint: Path, split: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = checkpoint.parents[1].name
    return f"{run_name}_{split}_{stamp}"


def build_eval_dir(output_dir: str | Path, eval_name: str) -> Path:
    eval_dir = Path(output_dir) / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def _format_eval_log(
    *,
    split: str,
    metrics: dict[str, Any],
    elapsed_sec: float,
) -> str:
    return (
        f"split={split} "
        f"{split}_loss={_fmt_float(metrics.get(f'{split}_loss'))} "
        f"mpjpe_norm={_fmt_float(metrics.get('mpjpe_norm'))} "
        f"pck@0.05={_fmt_float(metrics.get('pck_0.05'), precision=4)} "
        f"pck@0.10={_fmt_float(metrics.get('pck_0.10'), precision=4)} "
        f"pck@0.20={_fmt_float(metrics.get('pck_0.20'), precision=4)} "
        f"pck@0.50={_fmt_float(metrics.get('pck_0.50'), precision=4)} "
        f"joint_std_mean={_fmt_float(metrics.get('mean_joint_std'))} "
        f"joint_std_min={_fmt_float(metrics.get('min_joint_std'))} "
        f"gt_joint_std_mean={_fmt_float(metrics.get('gt_mean_joint_std'))} "
        f"time={elapsed_sec:.1f}s"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhysCSI-Pose from a checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-id", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--eval-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true", default=None)
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment.env_id": args.env_id,
        "experiment.split": args.split,
        "experiment.eval_name": args.eval_name,
        "experiment.device": args.device,
        "experiment.save_predictions": args.save_predictions,
    }


def _checkpoint_run_dir(checkpoint: Path) -> Path | None:
    if len(checkpoint.parents) < 2:
        return None
    return checkpoint.parents[1]


def _eval_index_path(checkpoint: Path, eval_dir: Path, split: str) -> Path:
    run_dir = _checkpoint_run_dir(checkpoint)
    if run_dir is not None:
        run_index = run_dir / "window_index" / f"{split}.npz"
        if run_index.exists():
            return run_index
    return eval_dir / "window_index" / f"{split}.npz"


@torch.no_grad()
def _save_raw_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    path: Path,
) -> None:
    model.eval()
    preds: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    autocast = torch.amp.autocast if device.type == "cuda" else None

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        if autocast is None or not amp_enabled:
            pred = model(x)
        else:
            with autocast(device_type="cuda", enabled=True):
                pred = model(x)
        preds.append(pred.detach().cpu().numpy().astype("float32", copy=False))
        global_indices.append(batch["global_idx"].detach().cpu().numpy().astype("int64", copy=False))

    # These are raw overlapping window predictions. Formal metrics are computed
    # by evaluate_one_epoch, which aggregates overlapping windows before scoring.
    np.savez(
        path,
        pred_xy=np.concatenate(preds, axis=0),
        global_idx=np.concatenate(global_indices, axis=0),
    )


def main() -> None:
    _setup_logging()
    args = _parse_args()
    cfg = load_config_with_overrides(args.config, overrides=_cli_overrides(args))

    experiment_cfg = cfg.setdefault("experiment", {})
    data_cfg = cfg.setdefault("data", {})
    eval_cfg = cfg.setdefault("eval", {})
    metrics_cfg = cfg.setdefault("metrics", {})
    window_cfg = data_cfg.setdefault("window", {})

    checkpoint_path = args.checkpoint
    split = str(experiment_cfg.get("split", "test"))
    eval_name = experiment_cfg.get("eval_name") or make_eval_name(checkpoint_path, split)
    experiment_cfg["eval_name"] = str(eval_name)
    eval_dir = build_eval_dir(experiment_cfg.get("output_dir", "outputs"), str(eval_name))
    LOGGER.info("eval_dir=%s checkpoint=%s split=%s", eval_dir, checkpoint_path, split)

    feature_names = _feature_names(cfg)
    input_channels = len(selected_feature_channels(feature_names))
    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("input_channels", "auto") == "auto":
        model_cfg["input_channels"] = input_channels
    else:
        input_channels = int(model_cfg["input_channels"])

    with (eval_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    LOGGER.info("resolved_config=%s", eval_dir / "config_resolved.yaml")

    device = resolve_device(str(experiment_cfg.get("device", "auto")))
    protocol = str(experiment_cfg.get("protocol", "source_only"))
    env_id = int(experiment_cfg.get("env_id", 1))
    window_length = int(window_cfg.get("length", 4))
    stride = int(window_cfg.get("stride", 1))
    rebuild_index = bool(window_cfg.get("rebuild_index", False))
    LOGGER.info("device=%s protocol=%s env_id=%d", device, protocol, env_id)

    dataset = WindowMemmapPoseDataset(
        data_cfg["memmap_root"],
        index_path=_eval_index_path(checkpoint_path, eval_dir, split),
        protocol=protocol,
        env_id=env_id,
        split=split,
        window_length=window_length,
        stride=stride,
        features=feature_names,
        rebuild_index=rebuild_index,
    )
    LOGGER.info(
        "data memmap_root=%s features=%s input_channels=%d window_length=%d stride=%d windows=%d",
        data_cfg["memmap_root"],
        feature_names or ["l_norm", "d_center", "f_sub", "c_ant"],
        input_channels,
        window_length,
        stride,
        len(dataset),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    model = _build_model(cfg, input_channels=input_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    LOGGER.info(
        "loaded_checkpoint epoch=%s best_metric=%s",
        checkpoint.get("epoch", "unknown"),
        checkpoint.get("best_metric", "unknown"),
    )

    amp_enabled = bool(eval_cfg.get("amp", True)) and device.type == "cuda"
    pck_thresholds = tuple(float(v) for v in metrics_cfg.get("pck_thresholds", [0.05, 0.10, 0.20, 0.50]))
    eval_start = time.perf_counter()
    metrics = evaluate_one_epoch(
        model,
        loader,
        device=device,
        amp_enabled=amp_enabled,
        pck_thresholds=pck_thresholds,
        loss_prefix=split,
    )
    elapsed_sec = time.perf_counter() - eval_start

    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    LOGGER.info("%s", _format_eval_log(split=split, metrics=metrics, elapsed_sec=elapsed_sec))
    LOGGER.info("metrics_path=%s", eval_dir / "metrics.json")

    if bool(experiment_cfg.get("save_predictions", False)):
        _save_raw_predictions(
            model,
            loader,
            device=device,
            amp_enabled=amp_enabled,
            path=eval_dir / "predictions.npz",
        )
        LOGGER.info("predictions_path=%s", eval_dir / "predictions.npz")


if __name__ == "__main__":
    main()
