from pathlib import Path

import numpy as np
import pytest

from scripts.build_memmap import ensure_output_root, standardize_csi_frame


def test_standardize_csi_frame_transposes_raw_layout():
    raw = np.arange(3 * 114 * 10, dtype=np.float32).reshape(3, 114, 10)
    standardized = standardize_csi_frame(raw, Path("frame_001.mat"))
    assert standardized.shape == (10, 3, 114)
    assert standardized[0, 0, 0] == raw[0, 0, 0]
    assert standardized[9, 2, 113] == raw[2, 113, 9]


def test_standardize_csi_frame_rejects_unexpected_shape():
    raw = np.zeros((10, 3, 114), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[3,114,10\]"):
        standardize_csi_frame(raw, Path("frame_001.mat"))


def test_ensure_output_root_refuses_existing_cache_without_overwrite(tmp_path):
    (tmp_path / "X_all.npy").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="--overwrite"):
        ensure_output_root(tmp_path, overwrite=False)


def test_ensure_output_root_removes_existing_cache_with_overwrite(tmp_path):
    existing = tmp_path / "X_all.npy"
    existing.write_bytes(b"existing")
    ensure_output_root(tmp_path, overwrite=True)
    assert not existing.exists()
