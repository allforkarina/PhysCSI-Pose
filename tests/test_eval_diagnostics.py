from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import _compute_diagnostics


def test_compute_diagnostics_reports_per_joint_std_and_group() -> None:
    targets = np.zeros((3, 17, 2), dtype=np.float32)
    predictions = np.zeros((3, 17, 2), dtype=np.float32)
    targets[:, 16, 0] = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    predictions[:, 16, 0] = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    diagnostic = _compute_diagnostics([predictions], [targets])
    wrist_row = diagnostic["joint_rows"][16]

    assert wrist_row["joint_name"] == "right_wrist"
    assert wrist_row["joint_group"] == "distal_limb"
    assert np.isclose(wrist_row["gt_std"], np.sqrt(wrist_row["gt_var"]))
    assert np.isclose(wrist_row["pred_std"], np.sqrt(wrist_row["pred_var"]))
    assert np.isclose(wrist_row["std_ratio"], wrist_row["pred_std"] / wrist_row["gt_std"])
    assert diagnostic["overall"]["overall_std_ratio"] > 0.0
