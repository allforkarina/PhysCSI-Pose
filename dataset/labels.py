from __future__ import annotations

import numpy as np


VALID_COORD_FORMATS = {"pixel_1920x1080", "unit_norm_0_1", "target_norm_-0.8_0.8"}


def detect_gt_coord_format(xy_min: float, xy_max: float, abs_max: float) -> str:
    if abs_max > 10.0:
        return "pixel_1920x1080"
    if xy_min >= 0.0 and xy_max <= 1.0:
        return "unit_norm_0_1"
    return "target_norm_-0.8_0.8"


def normalize_gt_sequence(
    gt: np.ndarray,
    *,
    coord_format: str | None = None,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    target_min: float = -0.8,
    target_max: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if gt.shape != (297, 17, 3):
        raise ValueError(f"GT sequence must have shape [297, 17, 3], got {gt.shape}")

    gt = gt.astype(np.float32, copy=True)
    xy = gt[..., :2]
    conf = gt[..., 2]

    invalid_xy = ~np.isfinite(xy).all(axis=-1)
    zero_xy = np.all(xy == 0.0, axis=-1)
    invalid_conf = ~np.isfinite(conf)
    invalid = invalid_xy | zero_xy | invalid_conf

    xy[invalid_xy | zero_xy] = 0.0
    conf[invalid] = 0.0

    finite_xy = xy[np.isfinite(xy)]
    xy_min = float(finite_xy.min()) if finite_xy.size else 0.0
    xy_max = float(finite_xy.max()) if finite_xy.size else 0.0
    abs_max = float(np.abs(finite_xy).max()) if finite_xy.size else 0.0

    if coord_format is None:
        coord_format = detect_gt_coord_format(xy_min, xy_max, abs_max)
    elif coord_format not in VALID_COORD_FORMATS:
        raise ValueError(f"coord_format must be one of {sorted(VALID_COORD_FORMATS)}, got {coord_format!r}")

    if coord_format == "pixel_1920x1080":
        xy[..., 0] = xy[..., 0] / image_width
        xy[..., 1] = xy[..., 1] / image_height

    if coord_format in {"pixel_1920x1080", "unit_norm_0_1"}:
        span = target_max - target_min
        xy = xy * span + target_min

    xy = np.clip(xy, target_min, target_max).astype(np.float32, copy=False)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32, copy=False)
    xy[invalid] = 0.0

    stats = {
        "coord_format": coord_format,
        "xy_min_before": xy_min,
        "xy_max_before": xy_max,
        "abs_max_before": abs_max,
        "invalid_keypoints": int(invalid.sum()),
    }
    return xy, conf, stats
