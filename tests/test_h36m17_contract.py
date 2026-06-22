from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_h36m17_joint_contract() -> None:
    h36m17 = importlib.import_module("dataset.h36m17")

    assert h36m17.SKELETON_NAME == "human36m17"
    assert len(h36m17.H36M17_JOINT_NAMES) == 17
    assert h36m17.H36M17_JOINT_NAMES == (
        "pelvis",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "spine",
        "thorax",
        "neck",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    )
    assert len(h36m17.H36M17_EDGES) == 16
    assert h36m17.H36M17_JOINT_GROUPS["distal"] == (3, 6, 13, 16)
    assert h36m17.H36M17_JOINT_GROUPS["wrist"] == (13, 16)
    assert h36m17.H36M17_JOINT_GROUPS["ankle"] == (3, 6)


def test_h36m17_edges_reference_valid_joint_indices() -> None:
    h36m17 = importlib.import_module("dataset.h36m17")
    joint_indices = set(range(len(h36m17.H36M17_JOINT_NAMES)))

    assert all(left in joint_indices and right in joint_indices for left, right in h36m17.H36M17_EDGES)
    assert len(set(tuple(sorted(edge)) for edge in h36m17.H36M17_EDGES)) == len(h36m17.H36M17_EDGES)


def test_tensor_layout_contract() -> None:
    layouts = importlib.import_module("dataset.layouts")

    assert layouts.RAW_CSI_MAT_LAYOUT == "antenna,subcarrier,time10"
    assert layouts.MODEL_INPUT_LAYOUT == "sample,antenna,subcarrier,time64"
    assert layouts.POSE_TARGET_LAYOUT == "sample,joint,xy"
    assert layouts.RAW_CSI_MAT_SHAPE == (3, 114, 10)
    assert layouts.MODEL_INPUT_SHAPE == (3, 114, 64)
    assert layouts.POSE_TARGET_SHAPE == (17, 2)


def test_project_contracts_do_not_define_openpose18() -> None:
    contract_files = [
        Path("dataset/h36m17.py"),
        Path("dataset/layouts.py"),
    ]

    for path in contract_files:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        assert "OpenPose18" not in text
        assert "openpose18" not in text
        assert "OPENPOSE18" not in text
