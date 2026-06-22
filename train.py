from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from dataset.h36m17 import H36M17_JOINT_NAMES, SKELETON_NAME
from losses.pose_losses import bone_length_loss, coordinate_l1_loss
from models.baseline_csi_pose import BaselineCSIPoseModel
from models.wavelet_concat_baseline import WaveletConcatBaseline
from models.wm_wiflow import WMWiFlowPoseModel


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

    if model_type == "baseline":
        return BaselineCSIPoseModel(
            num_joints=num_joints,
            d_model=d_model,
            use_graph_refiner=graph_refinement,
        )
    if model_type == "wavelet_concat":
        return WaveletConcatBaseline(
            num_joints=num_joints,
            d_model=d_model,
            wavelet=str(model_config.get("wavelet", "db2")),
            use_graph_refiner=graph_refinement,
        )
    if model_type == "wm_wiflow":
        return WMWiFlowPoseModel(
            num_joints=num_joints,
            d_model=d_model,
            wavelet=str(model_config.get("wavelet", "db2")),
            use_fine_branch=bool(model_config.get("fine_branch", True)),
            use_graph_refiner=graph_refinement,
        )
    raise ValueError(f"unknown model type: {model_type}")


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


def training_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    x, target = _unpack_batch(batch)
    output = model(x)
    pred = extract_pose(output)
    loss = compute_pose_loss(pred, target, config)
    return {"loss": loss, "pred": pred}


def _unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, Any] | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"]
    return batch[0], batch[1]


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
    raise NotImplementedError("full epoch training loop is intentionally added after dataset split files are finalized")


if __name__ == "__main__":
    main()
