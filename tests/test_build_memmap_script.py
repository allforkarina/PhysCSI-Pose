from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from scripts.build_memmap import ensure_output_root, standardize_csi_frame


def test_standardize_csi_frame_transposes_raw_layout():
    raw = np.arange(3 * 114 * 10, dtype=np.float32).reshape(3, 114, 10)
    standardized = standardize_csi_frame(raw, Path("frame001.mat"))
    assert standardized.shape == (10, 3, 114)
    assert standardized[0, 0, 0] == raw[0, 0, 0]
    assert standardized[9, 2, 113] == raw[2, 113, 9]


def test_standardize_csi_frame_rejects_unexpected_shape():
    raw = np.zeros((10, 3, 114), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[3,114,10\]"):
        standardize_csi_frame(raw, Path("frame001.mat"))


def test_standardize_csi_frame_repairs_bad_values_and_records_stats():
    raw = np.ones((3, 114, 10), dtype=np.float32)
    raw[0, 0, 0] = np.nan
    raw[1, 1, 1] = np.inf
    raw[2, 2, 2] = -5.0
    repair_stats: dict[str, int] = {}

    standardized = standardize_csi_frame(
        raw,
        Path("A01/S01/wifi-csi/frame123.mat"),
        repair_stats=repair_stats,
    )

    assert standardized.shape == (10, 3, 114)
    assert np.isfinite(standardized).all()
    assert (standardized >= 0.0).all()
    assert standardized[0, 0, 0] == 1.0
    assert standardized[1, 1, 1] == 1.0
    assert standardized[2, 2, 2] == 1.0
    assert repair_stats["repaired_values"] == 3
    assert repair_stats["nan_values"] == 1
    assert repair_stats["inf_values"] == 1
    assert repair_stats["negative_values"] == 1
    assert repair_stats["repaired_frames"] == 1


def test_standardize_csi_frame_rejects_fully_unrepairable_frame():
    raw = np.full((3, 114, 10), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match=r"frame999\.mat.*no finite non-negative"):
        standardize_csi_frame(raw, Path("A01/S01/wifi-csi/frame999.mat"))


def test_default_config_matches_server_csi_layout():
    cfg = yaml.safe_load(Path("configs/build_memmap.yaml").read_text(encoding="utf-8"))
    csi_pattern = cfg["paths"]["csi_pattern"]
    assert (
        csi_pattern.format(action_id=1, subject_id=1, frame_id_1based=1)
        == "A01/S01/wifi-csi/frame001.mat"
    )


def test_build_script_can_run_directly_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_memmap.py", "--help"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--csi-root" in result.stdout


def test_ensure_output_root_refuses_existing_cache_without_overwrite(tmp_path):
    (tmp_path / "X_all.npy").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_root(tmp_path, overwrite=False)


def test_ensure_output_root_removes_existing_cache_with_overwrite(tmp_path):
    existing = tmp_path / "X_all.npy"
    existing.write_bytes(b"existing")
    ensure_output_root(tmp_path, overwrite=True)
    assert not existing.exists()
