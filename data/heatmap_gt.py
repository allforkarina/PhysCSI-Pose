from __future__ import annotations

import numpy as np


def valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def extract_h36m17_xy(keypoints: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape[-2] != 17 or keypoints.shape[-1] not in (2, 3):
        raise ValueError(f"Expected Human3.6M keypoints with shape (...,17,2/3), got {keypoints.shape}")
    xy = keypoints[..., :2]
    if not np.isfinite(xy).all():
        raise ValueError("Human3.6M keypoints contain non-finite coordinates")
    return np.ascontiguousarray(xy, dtype=np.float32)
