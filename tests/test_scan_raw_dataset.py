from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_csi_frame(csi_root: Path, *, action: int, subject: int, frame: int, values: np.ndarray) -> Path:
    path = csi_root / f"A{action:02d}" / f"S{subject:02d}" / "wifi-csi" / f"frame{frame:03d}.mat"
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"CSIamp": values})
    return path


def write_gt(gt_root: Path, *, env: int, subject: int, action: int, frames: int) -> Path:
    gt_root.mkdir(parents=True, exist_ok=True)
    gt = np.zeros((frames, 17, 3), dtype=np.float32)
    for frame in range(frames):
        gt[frame, :, 0] = np.linspace(0.0, 1.0, 17) + frame
        gt[frame, :, 1] = np.linspace(1.0, 2.0, 17) + action
        gt[frame, :, 2] = 0.5
    path = gt_root / f"E{env:02d}_S{subject:02d}_A{action:02d}.npy"
    np.save(path, gt)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_scan_raw_dataset_generates_contract_and_core_outputs(tmp_path: Path) -> None:
    from scripts.scan_raw_dataset import main

    csi_root = tmp_path / "raw_csi"
    gt_root = tmp_path / "gt"
    output_root = tmp_path / "audit"
    for frame in (1, 2):
        write_csi_frame(
            csi_root,
            action=1,
            subject=1,
            frame=frame,
            values=np.full((3, 4, 5), float(frame), dtype=np.float32),
        )
    write_gt(gt_root, env=1, subject=1, action=1, frames=2)

    main(
        [
            "--csi-root",
            str(csi_root),
            "--gt-root",
            str(gt_root),
            "--output-root",
            str(output_root),
            "--target-time-steps",
            "8",
        ]
    )

    summary = json.loads((output_root / "audit_summary.json").read_text(encoding="utf-8"))
    csi_shape_histogram = json.loads((output_root / "csi_shape_histogram.json").read_text(encoding="utf-8"))
    contract = (output_root / "architecture_contract.yaml").read_text(encoding="utf-8")

    assert summary["satisfies_current_project_assumptions"] is False
    assert summary["blocking_errors"] == []
    assert csi_shape_histogram == {"3x4x5": 2}
    assert "skeleton: human36m17" in contract
    assert "input_shape: [3, 4, 8]" in contract
    assert "output_shape: [17, 2]" in contract
    assert (output_root / "sequence_inventory.csv").exists()
    assert (output_root / "sampled_pairings.csv").exists()


def test_scan_records_bad_csi_files_and_pairing_errors(tmp_path: Path) -> None:
    from scripts.scan_raw_dataset import main

    csi_root = tmp_path / "raw_csi"
    gt_root = tmp_path / "gt"
    output_root = tmp_path / "audit"
    bad_key_path = csi_root / "A01" / "S01" / "wifi-csi" / "frame001.mat"
    bad_key_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(bad_key_path, {"OtherKey": np.zeros((3, 4, 5), dtype=np.float32)})
    write_csi_frame(csi_root, action=1, subject=1, frame=2, values=np.zeros((3, 4, 4), dtype=np.float32))
    nonfinite = np.ones((3, 4, 5), dtype=np.float32)
    nonfinite[0, 0, 0] = np.nan
    nonfinite[0, 0, 1] = np.inf
    write_csi_frame(csi_root, action=1, subject=1, frame=4, values=nonfinite)
    write_gt(gt_root, env=1, subject=1, action=1, frames=3)

    main(["--csi-root", str(csi_root), "--gt-root", str(gt_root), "--output-root", str(output_root)])

    abnormal = read_csv(output_root / "abnormal_files.csv")
    missing = read_csv(output_root / "missing_frames.csv")
    pairing_errors = read_csv(output_root / "pairing_errors.csv")
    reasons = {row["reason"] for row in abnormal}

    assert {"missing_csi_key", "unexpected_csi_shape", "nonfinite_csi_values"}.issubset(reasons)
    assert any(row["missing_frame"] == "3" for row in missing)
    assert any(row["reason"] == "csi_gt_frame_count_mismatch" for row in pairing_errors)


def test_scan_flags_ambiguous_environment_mapping_and_variable_lengths(tmp_path: Path) -> None:
    from scripts.scan_raw_dataset import main

    csi_root = tmp_path / "raw_csi"
    gt_root = tmp_path / "gt"
    output_root = tmp_path / "audit"
    for frame in (1, 2, 3):
        write_csi_frame(
            csi_root,
            action=1,
            subject=1,
            frame=frame,
            values=np.ones((3, 114, 10), dtype=np.float32),
        )
    write_gt(gt_root, env=1, subject=1, action=1, frames=2)
    write_gt(gt_root, env=2, subject=1, action=1, frames=3)

    main(["--csi-root", str(csi_root), "--gt-root", str(gt_root), "--output-root", str(output_root)])

    summary = json.loads((output_root / "audit_summary.json").read_text(encoding="utf-8"))
    contract = (output_root / "architecture_contract.yaml").read_text(encoding="utf-8")

    assert "same_subject_action_across_environments_without_csi_env_path" in summary["blocking_errors"]
    assert "fixed_sequence_length: false" in contract
    assert "use_cumulative_offsets: true" in contract
