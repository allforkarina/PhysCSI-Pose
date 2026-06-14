from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.features import selected_feature_channels
from engine.loops import evaluate_one_epoch, train_one_epoch
from engine.window_dataset import WindowMemmapPoseDataset
from models import PhysCSIPoseNet


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def load_config_with_overrides(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_by_dotted_key(config, key, value)
    return config


def make_run_name(protocol: str, env_id: int, window_length: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{protocol}_env{env_id:02d}_L{window_length}_{stamp}"


def build_run_dir(output_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(output_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "window_index").mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhysCSI-Pose from fixed CSI windows.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--env-id", type=int, default=None)
    parser.add_argument("--protocol", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment.env_id": args.env_id,
        "experiment.protocol": args.protocol,
        "experiment.run_name": args.run_name,
        "experiment.device": args.device,
        "train.batch_size": args.batch_size,
        "train.epochs": args.epochs,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_names(cfg: dict[str, Any]) -> list[str] | None:
    names = cfg.get("data", {}).get("feature_selection", {}).get("names")
    if names is None:
        return None
    return [str(name).lower() for name in names]


def _build_model(cfg: dict[str, Any], input_channels: int) -> PhysCSIPoseNet:
    model_cfg = cfg.get("model", {})
    temporal_cfg = model_cfg.get("temporal", {})
    token_projection_cfg = model_cfg.get("token_projection", {})
    return PhysCSIPoseNet(
        input_channels=input_channels,
        token_dim=int(model_cfg.get("token_dim", 128)),
        num_joints=int(model_cfg.get("num_joints", 17)),
        temporal_layers=int(temporal_cfg.get("num_layers", 2)),
        temporal_heads=int(temporal_cfg.get("num_heads", 4)),
        temporal_max_window_length=int(temporal_cfg.get("max_window_length", 8)),
        dropout=float(token_projection_cfg.get("dropout", 0.1)),
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0), cosine_epochs) / cosine_epochs
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_factor + (1.0 - min_factor) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _is_better(value: float, best: float | None, *, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    cfg = load_config_with_overrides(args.config, overrides=_cli_overrides(args))

    experiment_cfg = cfg.setdefault("experiment", {})
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("train", {})
    scheduler_cfg = cfg.setdefault("scheduler", {})
    checkpoint_cfg = cfg.setdefault("checkpoint", {})
    early_cfg = cfg.setdefault("early_stopping", {})
    metrics_cfg = cfg.setdefault("metrics", {})
    window_cfg = data_cfg.setdefault("window", {})
    split_cfg = data_cfg.setdefault("splits", {})

    protocol = str(experiment_cfg.get("protocol", "source_only"))
    env_id = int(experiment_cfg.get("env_id", 1))
    window_length = int(window_cfg.get("length", 4))
    run_name = experiment_cfg.get("run_name") or make_run_name(protocol, env_id, window_length)
    experiment_cfg["run_name"] = run_name
    run_dir = build_run_dir(experiment_cfg.get("output_dir", "runs"), str(run_name))

    seed = int(experiment_cfg.get("seed", 42))
    _seed_everything(seed)
    device = resolve_device(str(experiment_cfg.get("device", "auto")))

    feature_names = _feature_names(cfg)
    input_channels = len(selected_feature_channels(feature_names))
    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("input_channels", "auto") == "auto":
        model_cfg["input_channels"] = input_channels
    else:
        input_channels = int(model_cfg["input_channels"])

    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    memmap_root = data_cfg["memmap_root"]
    stride = int(window_cfg.get("stride", 1))
    rebuild_index = bool(window_cfg.get("rebuild_index", False))
    train_dataset = WindowMemmapPoseDataset(
        memmap_root,
        index_path=run_dir / "window_index" / "train.npz",
        protocol=protocol,
        env_id=env_id,
        split=str(split_cfg.get("train", "train")),
        window_length=window_length,
        stride=stride,
        features=feature_names,
        rebuild_index=rebuild_index,
    )
    val_dataset = WindowMemmapPoseDataset(
        memmap_root,
        index_path=run_dir / "window_index" / "val.npz",
        protocol=protocol,
        env_id=env_id,
        split=str(split_cfg.get("val", "val")),
        window_length=window_length,
        stride=stride,
        features=feature_names,
        rebuild_index=rebuild_index,
    )

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_model(cfg, input_channels=input_channels).to(device)
    base_lr = float(train_cfg.get("lr", 1.0e-3))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    scheduler = _make_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 10)),
        min_lr=float(scheduler_cfg.get("min_lr", 1.0e-5)),
        base_lr=base_lr,
    )

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    pck_thresholds = tuple(float(v) for v in metrics_cfg.get("pck_thresholds", [0.05, 0.10, 0.20, 0.50]))
    monitor = str(checkpoint_cfg.get("monitor", "val_loss"))
    mode = str(checkpoint_cfg.get("mode", "min"))
    early_monitor = str(early_cfg.get("monitor", monitor))
    early_mode = str(early_cfg.get("mode", mode))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    patience = int(early_cfg.get("patience", 20))
    early_enabled = bool(early_cfg.get("enabled", True))
    best_metric: float | None = None
    early_best_metric: float | None = None
    stale_epochs = 0
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
        )
        val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            pck_thresholds=pck_thresholds,
        )
        scheduler.step()

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
            **val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")

        current = float(epoch_metrics[monitor])
        if _is_better(current, best_metric, mode=mode, min_delta=0.0):
            best_metric = current
            if bool(checkpoint_cfg.get("save_best", True)):
                _save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric,
                    config=cfg,
                )

        if bool(checkpoint_cfg.get("save_last", True)):
            _save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_metric=float(best_metric if best_metric is not None else current),
                config=cfg,
            )

        early_value = float(epoch_metrics[early_monitor])
        if _is_better(early_value, early_best_metric, mode=early_mode, min_delta=min_delta):
            early_best_metric = early_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_enabled and stale_epochs >= patience:
            break


if __name__ == "__main__":
    main()
