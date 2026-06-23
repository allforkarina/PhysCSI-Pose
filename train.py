from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.h36m17 import H36M17_JOINT_NAMES, SKELETON_NAME
from dataset.normalization import compute_normalization_stats, save_normalization_stats
from dataset.resampled_pose_dataset import ResampledPoseDataset
from losses.pose_losses import bone_length_loss, coordinate_l1_loss
from metrics.pose_metrics import ankle_mpjpe, joint_group_mpjpe, overall_mpjpe, per_joint_mpjpe, wrist_mpjpe
from models.baseline_csi_pose import BaselineCSIPoseModel
from models.wavelet_feature_bank import TemporalSWTFeatureBank
from models.wavelet_concat_baseline import WaveletConcatBaseline
from models.wm_wiflow import WMWiFlowPoseModel


PARAM_GROUP_PREFIXES = {
    "coarse_encoder": (
        "encoder.feature_mapper.",
        "encoder.coarse_fusion.",
        "encoder.coarse_spatial_encoder.",
        "encoder.coarse_axial_encoder.",
    ),
    "fine_encoder": (
        "encoder.feature_mapper.",
        "encoder.fine_fusion.",
        "encoder.fine_spatial_encoder.",
        "encoder.fine_axial_encoder.",
    ),
    "decoder": ("decoder.",),
    "graph_refiner": ("decoder.joint_refiner.", "graph_refiner."),
    "pose_head": ("pose_head.",),
    "stem": ("stem.",),
    "spatial_encoder": ("spatial_encoder.",),
    "axial_encoder": ("axial_encoder.",),
    "joint_decoder": ("joint_decoder.",),
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model_config = _model_config(config)
    model_type = model_config["type"]
    num_joints = int(model_config.get("num_joints", len(H36M17_JOINT_NAMES)))
    d_model = int(model_config.get("d_model", 256))
    graph_refinement = bool(model_config.get("graph_refinement", False))
    wavelet_bands = tuple(model_config.get("wavelet_bands", ("raw", "A3", "D3", "D2", "D1")))

    if model_type == "baseline":
        model = BaselineCSIPoseModel(
            num_joints=num_joints,
            d_model=d_model,
            use_graph_refiner=graph_refinement,
        )
    elif model_type == "wavelet_concat":
        model = WaveletConcatBaseline(
            num_joints=num_joints,
            d_model=d_model,
            wavelet=str(model_config.get("wavelet", "db2")),
            wavelet_bands=wavelet_bands,
            use_graph_refiner=graph_refinement,
        )
    elif model_type == "wm_wiflow":
        model = WMWiFlowPoseModel(
            num_joints=num_joints,
            d_model=d_model,
            wavelet=str(model_config.get("wavelet", "db2")),
            wavelet_bands=wavelet_bands,
            use_fine_branch=bool(model_config.get("fine_branch", True)),
            use_gate=bool(model_config.get("gate", True)),
            use_graph_refiner=graph_refinement,
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")
    apply_trainable_groups(model, config.get("trainable_groups", ("all",)))
    return model


def apply_trainable_groups(model: nn.Module, groups: Any) -> None:
    group_names = tuple(groups or ("all",))
    if "all" in group_names:
        for param in model.parameters():
            param.requires_grad = True
        return

    prefixes = []
    for group in group_names:
        prefixes.extend(PARAM_GROUP_PREFIXES.get(str(group), (f"{group}.",)))
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def collect_model_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = _model_config(config)
    metadata_config = config.get("metadata", {})
    return {
        "skeleton_name": metadata_config.get("skeleton_name", SKELETON_NAME),
        "num_joints": int(model_config.get("num_joints", len(H36M17_JOINT_NAMES))),
        "input_layout": metadata_config.get("input_layout", "antenna,subcarrier,time"),
        "model_type": model_config["type"],
    }


def validate_checkpoint_metadata(checkpoint_metadata: Mapping[str, Any], expected_metadata: Mapping[str, Any]) -> None:
    for key in ("skeleton_name", "num_joints", "input_layout"):
        if checkpoint_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"checkpoint metadata mismatch for {key}: "
                f"{checkpoint_metadata.get(key)!r} != {expected_metadata.get(key)!r}"
            )


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def extract_pose(output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if "pose" in output:
        return output["pose"]
    if "P_final" in output:
        return output["P_final"]
    raise KeyError("model output does not contain pose or P_final")


def compute_pose_loss(pred: torch.Tensor, target: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    losses = config.get("losses", {})
    l1_weight = float(losses.get("coordinate_l1", 1.0))
    bone_weight = float(losses.get("bone_length", 0.0))
    loss = pred.new_tensor(0.0)
    if l1_weight:
        loss = loss + l1_weight * coordinate_l1_loss(pred, target)
    if bone_weight:
        loss = loss + bone_weight * bone_length_loss(pred, target)
    return loss


def compute_base_pose_loss(output: torch.Tensor | Mapping[str, torch.Tensor], target: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    losses = config.get("losses", {})
    base_l1_weight = float(losses.get("base_coordinate_l1", 0.0))
    base_bone_weight = float(losses.get("base_bone_length", 0.0))
    if not isinstance(output, Mapping) or "P_base" not in output or (not base_l1_weight and not base_bone_weight):
        return target.new_tensor(0.0)
    loss = target.new_tensor(0.0)
    if base_l1_weight:
        loss = loss + base_l1_weight * coordinate_l1_loss(output["P_base"], target)
    if base_bone_weight:
        loss = loss + base_bone_weight * bone_length_loss(output["P_base"], target)
    return loss


def forward_for_training(model: nn.Module, x: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor | Mapping[str, torch.Tensor]:
    try:
        return model(x, return_intermediates=True)
    except TypeError:
        return model(x)


def training_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    x, target = _unpack_batch(batch)
    output = forward_for_training(model, x)
    pred = extract_pose(output)
    loss = compute_pose_loss(pred, target, config) + compute_base_pose_loss(output, target, config)
    return {"loss": loss, "pred": pred, "output": output}


def build_collate_fn(config: Mapping[str, Any]):
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    precompute_wavelet = bool(data_config.get("precompute_wavelet", False))
    feature_bank = None
    if precompute_wavelet:
        feature_bank = TemporalSWTFeatureBank(
            wavelet=str(model_config.get("wavelet", "db2")),
            bands=tuple(model_config.get("wavelet_bands", ("raw", "A3", "D3", "D2", "D1"))),
        )

    def collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, int]]]):
        x = torch.stack([item[0] for item in batch], dim=0)
        y = torch.stack([item[1] for item in batch], dim=0)
        meta = [item[2] for item in batch]
        if feature_bank is not None:
            x = feature_bank(x)
        return x, y, meta

    return collate


def run_training(config: Mapping[str, Any], *, resume_from: str | Path | None = None) -> dict[str, Any]:
    training_config = config.get("training", {})
    checkpoint_dir = Path(training_config.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("reproducibility", {}).get("seed", training_config.get("seed", 20260623)))
    set_random_seed(seed)
    save_reproducibility_artifacts(config, checkpoint_dir=checkpoint_dir, seed=seed)
    device = resolve_device(config)
    metadata = collect_model_metadata(config)

    train_dataset, val_dataset, test_dataset = build_datasets(config, checkpoint_dir=checkpoint_dir)
    collate_fn = build_collate_fn(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    model = build_model(config).to(device)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(training_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    start_epoch = 0
    best_metric = float("inf")
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        validate_checkpoint_metadata(checkpoint["metadata"], metadata)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        move_optimizer_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    history = []
    epochs = int(training_config.get("epochs", 1))
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device=device)
        val_metrics, diagnostics = validate(model, val_loader, device=device, return_diagnostics=True)
        val_metric = float(val_metrics["overall_mpjpe"])
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
            "best_metric": min(best_metric, val_metric),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if val_metric <= best_metric:
            best_metric = val_metric
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
        append_diagnostics(checkpoint_dir / "diagnostics.jsonl", epoch=epoch, diagnostics=diagnostics)

    return {
        "start_epoch": start_epoch,
        "epochs_completed": start_epoch + epochs,
        "best_metric": best_metric,
        "history": history,
        "checkpoint_dir": checkpoint_dir,
        "device": str(device),
        "split_counts": {
            "source_train": len(train_dataset),
            "source_val": len(val_dataset),
            "target_test": len(test_dataset),
        },
    }


def build_datasets(
    config: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
) -> tuple[ResampledPoseDataset, ResampledPoseDataset, ResampledPoseDataset]:
    data_config = config.get("data", {})
    data_root = Path(data_config.get("root", "."))
    split_index = np_load(data_root / str(data_config.get("split_index_file", "split_index.npz")))
    eval_env = str(data_config.get("eval_env"))
    train_indices = split_index[f"env_{eval_env}_source_train_frame_indices"]
    val_indices = split_index[f"env_{eval_env}_source_val_frame_indices"]
    test_indices = split_index[f"env_{eval_env}_target_test_frame_indices"]
    x_path = data_root / str(data_config.get("x_file", "X_amp_resampled.npy"))
    y_path = data_root / str(data_config.get("y_file", "Y_2d_clean.npy"))
    meta_path = data_root / str(data_config.get("meta_file", "meta.npz"))
    x = np_load(x_path, mmap_mode="r")
    stats = compute_normalization_stats(
        x,
        frame_indices=train_indices,
        chunk_size=int(data_config.get("normalization_chunk_size", 1024)),
    )
    save_normalization_stats(checkpoint_dir / "normalization_stats.npz", stats)
    train_dataset = ResampledPoseDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
        frame_indices=train_indices,
        normalization_stats=stats,
    )
    val_dataset = ResampledPoseDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
        frame_indices=val_indices,
        normalization_stats=stats,
    )
    test_dataset = ResampledPoseDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
        frame_indices=test_indices,
        normalization_stats=stats,
    )
    return train_dataset, val_dataset, test_dataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    device: torch.device | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        if device is not None:
            batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        result = training_step(model, batch, config)
        result["loss"].backward()
        optimizer.step()
        total_loss += float(result["loss"].detach())
        total_batches += 1
    if total_batches == 0:
        raise ValueError("training loader is empty")
    return total_loss / float(total_batches)


def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    return_diagnostics: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    preds = []
    targets = []
    diagnostics = []
    with torch.no_grad():
        for batch in loader:
            if device is not None:
                batch = move_to_device(batch, device)
            x, target = _unpack_batch(batch)
            output = forward_for_training(model, x)
            preds.append(extract_pose(output))
            targets.append(target)
            diagnostics.append(extract_diagnostics(output))
    if not preds:
        raise ValueError("validation loader is empty")
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    metrics = {
        "overall_mpjpe": float(overall_mpjpe(pred, target)),
        "per_joint_mpjpe": per_joint_mpjpe(pred, target).cpu(),
        "joint_group_mpjpe": {key: float(value) for key, value in joint_group_mpjpe(pred, target).items()},
        "wrist_mpjpe": float(wrist_mpjpe(pred, target)),
        "ankle_mpjpe": float(ankle_mpjpe(pred, target)),
    }
    if return_diagnostics:
        return metrics, aggregate_diagnostics(diagnostics)
    return metrics


def extract_diagnostics(output: torch.Tensor | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(output, Mapping):
        return {}
    keys = ("coarse_fusion_weights", "fine_fusion_weights", "alpha")
    return {key: output[key].detach().cpu() for key in keys if key in output and isinstance(output[key], torch.Tensor)}


def aggregate_diagnostics(items: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    keys = sorted({key for item in items for key in item})
    for key in keys:
        values = torch.cat([item[key].reshape(item[key].shape[0], -1) for item in items if key in item], dim=0)
        aggregated[key] = {
            "mean": values.mean(dim=0).tolist(),
            "std": values.std(dim=0, unbiased=False).tolist(),
        }
    return aggregated


def append_diagnostics(path: Path, *, epoch: int, diagnostics: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": epoch, "diagnostics": diagnostics}, sort_keys=True) + "\n")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_reproducibility_artifacts(config: Mapping[str, Any], *, checkpoint_dir: Path, seed: int) -> None:
    with (checkpoint_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_jsonable(config), handle, sort_keys=True)
    metadata = {
        "seed": seed,
        **git_metadata(),
    }
    (checkpoint_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(run_git(["status", "--porcelain"])),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _unpack_batch(
    batch: tuple[torch.Tensor | Mapping[str, torch.Tensor], torch.Tensor]
    | tuple[torch.Tensor | Mapping[str, torch.Tensor], torch.Tensor, Any]
    | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor | Mapping[str, torch.Tensor], torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"]
    return batch[0], batch[1]


def np_load(path: Path, *, mmap_mode: str | None = None):
    import numpy as np

    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("config must contain a model mapping")
    return model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PhysCSI-Pose model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--print-metadata", action="store_true", help="Print model metadata and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_metadata:
        print(collect_model_metadata(config))
        return
    run_training(config)


if __name__ == "__main__":
    main()
