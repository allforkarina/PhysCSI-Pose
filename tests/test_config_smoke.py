from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONFIG_PATHS = (
    REPO_ROOT / "configs" / "baseline.yaml",
    REPO_ROOT / "configs" / "wavelet_concat.yaml",
    REPO_ROOT / "configs" / "wm_wiflow.yaml",
)


def test_configs_build_models_and_emit_h36m17_pose() -> None:
    from train import build_model, collect_model_metadata, extract_pose, load_config

    for path in CONFIG_PATHS:
        config = load_config(path)
        model = build_model(config)
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(1, 3, 114, 64))
        pose = extract_pose(output)
        metadata = collect_model_metadata(config)

        assert pose.shape == (1, 17, 2)
        assert metadata["skeleton_name"] == "human36m17"
        assert metadata["num_joints"] == 17
        assert metadata["input_layout"] == "antenna,subcarrier,time"


def test_configs_define_required_comparison_switches_and_logs() -> None:
    from train import load_config

    required_metrics = {"overall_mpjpe", "per_joint_mpjpe", "joint_group_mpjpe", "wrist_mpjpe", "ankle_mpjpe"}
    for path in CONFIG_PATHS:
        config = load_config(path)
        model_config = config["model"]

        assert "wavelet_bands" in model_config
        assert "fine_branch" in model_config
        assert "gate" in model_config
        assert "graph_refinement" in model_config
        assert "losses" in config
        assert "trainable_groups" in config
        assert required_metrics.issubset(set(config["logging"]["metrics"]))


def test_checkpoint_metadata_rejects_mismatched_skeleton() -> None:
    from train import collect_model_metadata, load_config, validate_checkpoint_metadata

    config = load_config(CONFIG_PATHS[0])
    metadata = collect_model_metadata(config)
    bad_metadata = copy.deepcopy(metadata)
    bad_metadata["skeleton_name"] = "openpose18"

    with pytest.raises(ValueError, match="skeleton_name"):
        validate_checkpoint_metadata(bad_metadata, metadata)


def test_graph_refinement_flag_is_trainable_for_baseline() -> None:
    from train import build_model, extract_pose, load_config

    config = load_config(CONFIG_PATHS[0])
    config["model"]["graph_refinement"] = True
    model = build_model(config)
    model.eval()

    with torch.no_grad():
        pose = extract_pose(model(torch.randn(1, 3, 114, 64)))

    assert pose.shape == (1, 17, 2)


def test_wavelet_band_config_changes_concat_model_channels() -> None:
    from train import build_model, load_config

    config = load_config(REPO_ROOT / "configs" / "wavelet_concat.yaml")
    config["model"]["wavelet_bands"] = ["raw", "D2", "D1"]
    model = build_model(config)

    assert tuple(model.feature_bank.bands) == ("raw", "D2", "D1")
    assert model.stem[0].in_channels == 9


def test_gate_config_can_disable_learned_joint_gate() -> None:
    from train import build_model, load_config

    config = load_config(REPO_ROOT / "configs" / "wm_wiflow.yaml")
    config["model"]["gate"] = False
    config["model"]["graph_refinement"] = False
    model = build_model(config)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 114, 64), return_intermediates=True)

    assert torch.equal(output["alpha"], torch.ones_like(output["alpha"]))
    assert torch.allclose(output["pose"], output["P_base"] + output["Delta_P"])


def test_trainable_groups_freeze_unlisted_parameters() -> None:
    from train import build_model, load_config

    config = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    config["trainable_groups"] = ["pose_head"]
    model = build_model(config)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert trainable
    assert all(name.startswith("pose_head.") for name in trainable)
    assert not any(name.startswith("stem.") for name in trainable)


def test_disabling_fine_branch_skips_fine_encoder_compute() -> None:
    from train import build_model, load_config

    class FailingModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("fine branch should not run")

    config = load_config(REPO_ROOT / "configs" / "wm_wiflow.yaml")
    config["model"]["fine_branch"] = False
    config["model"]["graph_refinement"] = False
    model = build_model(config)
    model.encoder.fine_spatial_encoder = FailingModule()
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 114, 64), return_intermediates=True)

    assert output["fine_tokens"] is None
    assert torch.equal(output["pose"], output["P_base"])


def test_graph_refinement_keeps_base_plus_gated_residual_formula() -> None:
    from train import build_model, load_config

    config = load_config(REPO_ROOT / "configs" / "wm_wiflow.yaml")
    config["model"]["graph_refinement"] = True
    model = build_model(config)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 114, 64), return_intermediates=True)

    assert "pre_graph_pose" not in output
    assert torch.allclose(output["pose"], output["P_base"] + output["alpha"] * output["Delta_P"])
