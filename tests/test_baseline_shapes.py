from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_baseline_csi_pose_model_shapes() -> None:
    from models.baseline_csi_pose import BaselineCSIPoseModel

    model = BaselineCSIPoseModel(num_joints=17, d_model=256)
    x = torch.randn(2, 3, 114, 64)

    outputs = model(x, return_intermediates=True)

    assert outputs["stem_features"].shape == (2, 32, 114, 64)
    assert outputs["spatial_features"].shape == (2, 128, 29, 16)
    assert outputs["encoded_features"].shape == (2, 256, 29, 16)
    assert outputs["tokens"].shape == (2, 464, 256)
    assert outputs["joint_features"].shape == (2, 17, 256)
    assert outputs["pose"].shape == (2, 17, 2)


def test_baseline_csi_pose_model_returns_pose_by_default() -> None:
    from models.baseline_csi_pose import BaselineCSIPoseModel

    model = BaselineCSIPoseModel(num_joints=17, d_model=256)
    x = torch.randn(1, 3, 114, 64)

    pose = model(x)

    assert pose.shape == (1, 17, 2)
    assert torch.isfinite(pose).all()
