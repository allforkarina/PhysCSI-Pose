from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io


CSI_RE = re.compile(
    r"^A(?P<action>\d+)[/\\]S(?P<subject>\d+)[/\\]wifi-csi[/\\]frame(?P<frame>\d+)\.mat$",
    re.IGNORECASE,
)
GT_RE = re.compile(r"^E(?P<env>\d+)_S(?P<subject>\d+)_A(?P<action>\d+)\.npy$", re.IGNORECASE)
PERCENTILES = (0.0, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0)
OUTPUT_FILENAMES = (
    "scan_summary.json",
    "sequence_index.csv",
    "csi_file_issues.csv",
    "csi_nonfinite_values.csv",
    "csi_sequence_quality.csv",
    "csi_variable_stats.csv",
    "gt_file_issues.csv",
    "gt_sequence_stats.csv",
    "gt_2d_keypoint_stats.csv",
    "gt_2d_frame_stats.csv",
    "gt_2d_outliers.csv",
    "gt_keypoint_stats.csv",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress(message: str) -> None:
    print(f"[progress] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def shape_text(shape: tuple[int, ...] | list[int] | None) -> str:
    if not shape:
        return ""
    return "x".join(str(int(v)) for v in shape)


def rel_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def limited_bad_indices(mask: np.ndarray, limit: int) -> str:
    positions = np.argwhere(mask)
    if positions.size == 0:
        return ""
    rows = [",".join(str(int(v)) for v in pos) for pos in positions[:limit]]
    return "|".join(rows)


@dataclass
class NumericSummary:
    arrays: int = 0
    total_values: int = 0
    finite_values: int = 0
    nan_values: int = 0
    posinf_values: int = 0
    neginf_values: int = 0
    negative_values: int = 0
    zero_values: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    sample_limit: int = 200_000
    sample: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray, rng: np.random.Generator) -> None:
        arr = np.asarray(values)
        if arr.size == 0:
            self.arrays += 1
            return

        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        arr = arr.astype(np.float64, copy=False).reshape(-1)

        self.arrays += 1
        self.total_values += int(arr.size)
        self.nan_values += int(np.isnan(arr).sum())
        self.posinf_values += int(np.isposinf(arr).sum())
        self.neginf_values += int(np.isneginf(arr).sum())

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return

        self.finite_values += int(finite.size)
        self.negative_values += int((finite < 0.0).sum())
        self.zero_values += int((finite == 0.0).sum())
        local_min = float(finite.min())
        local_max = float(finite.max())
        self.min_value = local_min if self.min_value is None else min(self.min_value, local_min)
        self.max_value = local_max if self.max_value is None else max(self.max_value, local_max)
        self.sum_value += float(finite.sum(dtype=np.float64))
        self.sumsq_value += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))

        keep = min(int(finite.size), min(self.sample_limit, 4096))
        if keep > 0:
            if finite.size > keep:
                idx = rng.choice(finite.size, size=keep, replace=False)
                self.sample.append(finite[idx].copy())
            else:
                self.sample.append(finite.copy())
            self._trim_sample(rng)

    def _trim_sample(self, rng: np.random.Generator) -> None:
        total = sum(int(chunk.size) for chunk in self.sample)
        if total <= self.sample_limit * 2:
            return
        merged = np.concatenate(self.sample)
        if merged.size > self.sample_limit:
            idx = rng.choice(merged.size, size=self.sample_limit, replace=False)
            merged = merged[idx]
        self.sample = [merged]

    def as_dict(self) -> dict[str, Any]:
        mean = self.sum_value / self.finite_values if self.finite_values else None
        variance = None
        std = None
        if self.finite_values:
            variance = max(self.sumsq_value / self.finite_values - float(mean) ** 2, 0.0)
            std = math.sqrt(variance)

        quantiles: dict[str, float] = {}
        sample_values = np.concatenate(self.sample) if self.sample else np.array([], dtype=np.float64)
        if sample_values.size:
            qs = np.percentile(sample_values, PERCENTILES)
            quantiles = {f"p{p:g}": float(q) for p, q in zip(PERCENTILES, qs)}

        return {
            "arrays": self.arrays,
            "total_values": self.total_values,
            "finite_values": self.finite_values,
            "nan_values": self.nan_values,
            "posinf_values": self.posinf_values,
            "neginf_values": self.neginf_values,
            "negative_values": self.negative_values,
            "zero_values": self.zero_values,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
            "std": std,
            "quantiles_from_sample": quantiles,
            "quantile_sample_values": int(sample_values.size),
        }


@dataclass
class VariableSummary:
    count: int = 0
    dtype_counts: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    complex_arrays: int = 0
    numeric: NumericSummary = field(default_factory=NumericSummary)

    def update(self, arr: np.ndarray, rng: np.random.Generator) -> None:
        self.count += 1
        self.dtype_counts[str(arr.dtype)] += 1
        self.shape_counts[shape_text(tuple(arr.shape))] += 1
        if np.iscomplexobj(arr):
            self.complex_arrays += 1
        if np.issubdtype(arr.dtype, np.number):
            self.numeric.update(arr, rng)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "dtype_counts": dict(self.dtype_counts),
            "shape_counts": dict(self.shape_counts),
            "complex_arrays": self.complex_arrays,
            **self.numeric.as_dict(),
        }


@dataclass
class CsiSequence:
    action: int
    subject: int
    rel_dir: str
    frame_ids: list[int] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    read_errors: int = 0
    nonfinite_files: int = 0
    nonfinite_values: int = 0
    zero_values: int = 0
    unexpected_shape_files: int = 0


def read_mat_variables(path: Path) -> tuple[dict[str, np.ndarray], str]:
    try:
        loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        return {k: np.asarray(v) for k, v in loaded.items() if not k.startswith("__")}, "scipy"
    except (NotImplementedError, ValueError, OSError):
        variables: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            def collect(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and not name.startswith("#"):
                    variables[name] = np.asarray(obj)

            handle.visititems(collect)
        return variables, "h5py"


def discover_csi_files(csi_root: Path, progress_every: int) -> tuple[list[Path], list[dict[str, Any]]]:
    progress(f"discovering CSI .mat files under {csi_root}")
    files: list[Path] = []
    path_issues: list[dict[str, Any]] = []
    visited_dirs = 0
    for dirpath, _, filenames in os.walk(csi_root):
        visited_dirs += 1
        if progress_every and visited_dirs % progress_every == 0:
            progress(f"discovery visited_dirs={visited_dirs} mat_files={len(files)}")
        current = Path(dirpath)
        for name in filenames:
            if not name.lower().endswith(".mat"):
                continue
            path = current / name
            files.append(path)
            rel = rel_posix(path, csi_root)
            if CSI_RE.match(rel.replace("/", os.sep)) is None and CSI_RE.match(rel) is None:
                path_issues.append({"path": rel, "issue": "unexpected_csi_path_pattern"})
    progress(f"discovered CSI mat_files={len(files)} visited_dirs={visited_dirs}")
    return sorted(files), path_issues


def discover_gt_files(gt_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    progress(f"discovering GT .npy files under {gt_root}")
    files = sorted(gt_root.glob("*.npy"))
    issues = [
        {"path": path.name, "issue": "unexpected_gt_filename_pattern"}
        for path in files
        if GT_RE.match(path.name) is None
    ]
    progress(f"discovered GT npy_files={len(files)}")
    return files, issues


def csi_match(path: Path, csi_root: Path) -> tuple[int, int, int] | None:
    rel = rel_posix(path, csi_root)
    match = CSI_RE.match(rel)
    if match is None:
        return None
    return int(match.group("action")), int(match.group("subject")), int(match.group("frame"))


def gt_match(path: Path) -> tuple[int, int, int] | None:
    match = GT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("env")), int(match.group("subject")), int(match.group("action"))


def analyze_csi(
    csi_root: Path,
    files: list[Path],
    *,
    expected_key: str,
    expected_shape: tuple[int, ...],
    expected_frames: int,
    progress_every: int,
    max_files: int | None,
    sample_limit: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, int], CsiSequence],
    list[dict[str, Any]],
]:
    variables: defaultdict[str, VariableSummary] = defaultdict(
        lambda: VariableSummary(numeric=NumericSummary(sample_limit=sample_limit))
    )
    file_issues: list[dict[str, Any]] = []
    nonfinite_rows: list[dict[str, Any]] = []
    sequences: dict[tuple[int, int], CsiSequence] = {}
    mat_reader_counts: Counter[str] = Counter()

    selected = files[:max_files] if max_files is not None else files
    start = time.time()
    for idx, path in enumerate(selected, start=1):
        parsed = csi_match(path, csi_root)
        if parsed is not None:
            action, subject, frame = parsed
            key = (subject, action)
            seq = sequences.setdefault(
                key,
                CsiSequence(
                    action=action,
                    subject=subject,
                    rel_dir=f"A{action:02d}/S{subject:02d}/wifi-csi",
                ),
            )
            seq.frame_ids.append(frame)
            seq.files.append(path)
        else:
            seq = None

        try:
            data, reader = read_mat_variables(path)
            mat_reader_counts[reader] += 1
        except Exception as exc:  # noqa: BLE001 - scanner must keep going.
            file_issues.append({"path": rel_posix(path, csi_root), "issue": "mat_read_error", "detail": repr(exc)})
            if seq is not None:
                seq.read_errors += 1
            continue

        if expected_key and expected_key not in data:
            file_issues.append(
                {
                    "path": rel_posix(path, csi_root),
                    "issue": "missing_expected_key",
                    "detail": expected_key,
                    "available_keys": "|".join(sorted(data)),
                }
            )

        for name, arr in data.items():
            variables[name].update(arr, rng)
            if name == expected_key:
                if not np.issubdtype(arr.dtype, np.number):
                    continue
                finite = np.isfinite(arr)
                zero_values = int((arr == 0).sum())
                if seq is not None:
                    seq.zero_values += zero_values
                if not bool(finite.all()):
                    bad_mask = ~finite
                    nonfinite_count = int((~finite).sum())
                    nan_count = int(np.isnan(arr).sum())
                    posinf_count = int(np.isposinf(arr).sum())
                    neginf_count = int(np.isneginf(arr).sum())
                    file_issues.append(
                        {
                            "path": rel_posix(path, csi_root),
                            "issue": "expected_key_nonfinite_values",
                            "detail": str(nonfinite_count),
                            "shape": shape_text(tuple(arr.shape)),
                        }
                    )
                    nonfinite_rows.append(
                        {
                            "path": rel_posix(path, csi_root),
                            "subject": parsed[1] if parsed else "",
                            "action": parsed[0] if parsed else "",
                            "frame": parsed[2] if parsed else "",
                            "variable": name,
                            "shape": shape_text(tuple(arr.shape)),
                            "dtype": str(arr.dtype),
                            "nonfinite_values": nonfinite_count,
                            "nan_values": nan_count,
                            "posinf_values": posinf_count,
                            "neginf_values": neginf_count,
                            "zero_values_in_frame": zero_values,
                            "first_bad_indices": limited_bad_indices(bad_mask, 16),
                        }
                    )
                    if seq is not None:
                        seq.nonfinite_files += 1
                        seq.nonfinite_values += nonfinite_count
                if expected_shape and tuple(arr.shape) != expected_shape:
                    file_issues.append(
                        {
                            "path": rel_posix(path, csi_root),
                            "issue": "unexpected_expected_key_shape",
                            "detail": f"expected={shape_text(expected_shape)} actual={shape_text(tuple(arr.shape))}",
                        }
                    )
                    if seq is not None:
                        seq.unexpected_shape_files += 1

        if progress_every and idx % progress_every == 0:
            elapsed = max(time.time() - start, 1.0e-9)
            progress(
                f"CSI read files={idx}/{len(selected)} rate={idx / elapsed:.1f}/s "
                f"issues={len(file_issues)} variables={len(variables)}"
            )

    sequence_rows = []
    frame_counts = []
    for (subject, action), seq in sorted(sequences.items()):
        frames = sorted(seq.frame_ids)
        frame_counts.append(len(frames))
        duplicate_count = len(frames) - len(set(frames))
        expected = set(range(1, expected_frames + 1)) if expected_frames else set()
        missing = sorted(expected.difference(frames)) if expected else []
        extra = sorted(set(frames).difference(expected)) if expected else []
        if duplicate_count:
            file_issues.append({"path": seq.rel_dir, "issue": "duplicate_frame_ids", "detail": str(duplicate_count)})
        if missing:
            file_issues.append(
                {
                    "path": seq.rel_dir,
                    "issue": "missing_frame_ids",
                    "detail": ",".join(str(v) for v in missing[:50]),
                    "count": len(missing),
                }
            )
        if extra:
            file_issues.append(
                {
                    "path": seq.rel_dir,
                    "issue": "unexpected_frame_ids",
                    "detail": ",".join(str(v) for v in extra[:50]),
                    "count": len(extra),
                }
            )
        sequence_rows.append(
            {
                "subject": subject,
                "action": action,
                "csi_rel_dir": seq.rel_dir,
                "csi_frame_count": len(frames),
                "first_frame": frames[0] if frames else "",
                "last_frame": frames[-1] if frames else "",
                "missing_frame_count": len(missing),
                "extra_frame_count": len(extra),
                "duplicate_frame_count": duplicate_count,
                "mat_read_errors": seq.read_errors,
                "nonfinite_files": seq.nonfinite_files,
                "nonfinite_values": seq.nonfinite_values,
                "expected_key_zero_values": seq.zero_values,
                "unexpected_shape_files": seq.unexpected_shape_files,
            }
        )

    summary = {
        "mat_files": len(files),
        "mat_files_analyzed": len(selected),
        "mat_reader_counts": dict(mat_reader_counts),
        "sequence_count": len(sequences),
        "frame_count_per_sequence": describe_numbers(frame_counts),
        "variables": {name: summary.as_dict() for name, summary in sorted(variables.items())},
    }
    return summary, sequence_rows, file_issues, sequences, nonfinite_rows


def describe_numbers(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def infer_coord_format(x_min: float | None, x_max: float | None, y_min: float | None, y_max: float | None) -> str:
    values = [v for v in (x_min, x_max, y_min, y_max) if v is not None]
    if not values:
        return "unknown"
    abs_max = max(abs(v) for v in values)
    min_value = min(values)
    max_value = max(values)
    if abs_max > 10.0:
        return "pixel_like"
    if min_value >= 0.0 and max_value <= 1.0:
        return "unit_0_1_like"
    if min_value >= -1.0 and max_value <= 1.0:
        return "normalized_centered_like"
    return "unknown_numeric_range"


def update_keypoint_stats(gt: np.ndarray, keypoint_stats: list[dict[str, NumericSummary]], rng: np.random.Generator) -> None:
    xy = gt[..., :2]
    for joint in range(min(gt.shape[1], len(keypoint_stats))):
        joint_xy = xy[:, joint, :]
        valid_xy = np.isfinite(joint_xy).all(axis=1) & ~np.all(joint_xy == 0.0, axis=1)
        keypoint_stats[joint]["x"].update(joint_xy[valid_xy, 0], rng)
        keypoint_stats[joint]["y"].update(joint_xy[valid_xy, 1], rng)


def bbox_and_motion_stats(gt: np.ndarray, bbox_stats: dict[str, NumericSummary], motion_stats: NumericSummary, rng: np.random.Generator) -> None:
    xy = gt[..., :2].astype(np.float64, copy=False)
    valid = np.isfinite(xy).all(axis=-1) & ~np.all(xy == 0.0, axis=-1)

    widths = []
    heights = []
    areas = []
    centers_x = []
    centers_y = []
    valid_counts = []
    for frame_xy, frame_valid in zip(xy, valid):
        pts = frame_xy[frame_valid]
        valid_counts.append(int(pts.shape[0]))
        if pts.shape[0] < 2:
            continue
        x_min = float(pts[:, 0].min())
        x_max = float(pts[:, 0].max())
        y_min = float(pts[:, 1].min())
        y_max = float(pts[:, 1].max())
        width = x_max - x_min
        height = y_max - y_min
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        centers_x.append((x_min + x_max) / 2.0)
        centers_y.append((y_min + y_max) / 2.0)

    bbox_stats["valid_keypoints_per_frame"].update(np.asarray(valid_counts, dtype=np.float64), rng)
    bbox_stats["width"].update(np.asarray(widths, dtype=np.float64), rng)
    bbox_stats["height"].update(np.asarray(heights, dtype=np.float64), rng)
    bbox_stats["area"].update(np.asarray(areas, dtype=np.float64), rng)
    bbox_stats["center_x"].update(np.asarray(centers_x, dtype=np.float64), rng)
    bbox_stats["center_y"].update(np.asarray(centers_y, dtype=np.float64), rng)

    if xy.shape[0] > 1:
        both_valid = valid[1:] & valid[:-1]
        deltas = xy[1:] - xy[:-1]
        speeds = np.linalg.norm(deltas, axis=-1)
        motion_stats.update(speeds[both_valid], rng)


def build_gt_2d_detail_rows(
    gt: np.ndarray,
    *,
    env: int,
    subject: int,
    action: int,
    gt_file: str,
    outlier_abs_xy: float,
    outlier_bbox_width: float,
    outlier_bbox_height: float,
    remaining_outlier_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    xy = gt[..., :2].astype(np.float64, copy=False)
    valid = np.isfinite(xy).all(axis=-1) & ~np.all(xy == 0.0, axis=-1)
    frame_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    outlier_frame_count = 0
    outlier_keypoint_count = 0

    prev_xy: np.ndarray | None = None
    prev_valid: np.ndarray | None = None
    for frame_idx in range(xy.shape[0]):
        frame_number = frame_idx + 1
        frame_xy = xy[frame_idx]
        frame_valid = valid[frame_idx]
        pts = frame_xy[frame_valid]

        x_min = x_max = y_min = y_max = width = height = area = center_x = center_y = ""
        if pts.size:
            x_min_v = float(pts[:, 0].min())
            x_max_v = float(pts[:, 0].max())
            y_min_v = float(pts[:, 1].min())
            y_max_v = float(pts[:, 1].max())
            width_v = x_max_v - x_min_v
            height_v = y_max_v - y_min_v
            x_min, x_max, y_min, y_max = x_min_v, x_max_v, y_min_v, y_max_v
            width, height, area = width_v, height_v, width_v * height_v
            center_x, center_y = (x_min_v + x_max_v) / 2.0, (y_min_v + y_max_v) / 2.0

        motion_mean = motion_max = ""
        if prev_xy is not None and prev_valid is not None:
            both_valid = frame_valid & prev_valid
            if bool(both_valid.any()):
                speeds = np.linalg.norm(frame_xy[both_valid] - prev_xy[both_valid], axis=-1)
                motion_mean = float(speeds.mean())
                motion_max = float(speeds.max())

        xy_abs_mask = np.isfinite(frame_xy).all(axis=-1) & (
            (np.abs(frame_xy[:, 0]) > outlier_abs_xy) | (np.abs(frame_xy[:, 1]) > outlier_abs_xy)
        )
        bbox_outlier = (
            isinstance(width, float)
            and isinstance(height, float)
            and (width > outlier_bbox_width or height > outlier_bbox_height)
        )
        invalid_count = int((~frame_valid).sum())
        is_outlier = bool(xy_abs_mask.any() or bbox_outlier or invalid_count)
        if is_outlier:
            outlier_frame_count += 1
        outlier_keypoint_count += int(xy_abs_mask.sum())

        frame_rows.append(
            {
                "env": env,
                "subject": subject,
                "action": action,
                "gt_file": gt_file,
                "frame": frame_number,
                "valid_2d_keypoints": int(frame_valid.sum()),
                "invalid_or_zero_2d_keypoints": invalid_count,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area": area,
                "center_x": center_x,
                "center_y": center_y,
                "motion_l2_mean_from_prev": motion_mean,
                "motion_l2_max_from_prev": motion_max,
                "is_outlier_2d": int(is_outlier),
                "outlier_reasons": "|".join(
                    reason
                    for reason, enabled in (
                        ("invalid_or_zero_xy", bool(invalid_count)),
                        ("xy_abs_outlier", bool(xy_abs_mask.any())),
                        ("bbox_outlier", bool(bbox_outlier)),
                    )
                    if enabled
                ),
            }
        )

        if remaining_outlier_rows > 0:
            for joint_idx in np.flatnonzero(xy_abs_mask):
                if len(outlier_rows) >= remaining_outlier_rows:
                    break
                outlier_rows.append(
                    {
                        "env": env,
                        "subject": subject,
                        "action": action,
                        "gt_file": gt_file,
                        "frame": frame_number,
                        "joint_index": int(joint_idx),
                        "issue": "xy_abs_outlier",
                        "x": float(frame_xy[joint_idx, 0]),
                        "y": float(frame_xy[joint_idx, 1]),
                        "bbox_width": width,
                        "bbox_height": height,
                    }
                )
            if bbox_outlier and len(outlier_rows) < remaining_outlier_rows:
                outlier_rows.append(
                    {
                        "env": env,
                        "subject": subject,
                        "action": action,
                        "gt_file": gt_file,
                        "frame": frame_number,
                        "joint_index": "",
                        "issue": "bbox_outlier",
                        "x": "",
                        "y": "",
                        "bbox_width": width,
                        "bbox_height": height,
                    }
                )

        prev_xy = frame_xy
        prev_valid = frame_valid

    return frame_rows, outlier_rows, {
        "outlier_2d_frames": outlier_frame_count,
        "outlier_2d_keypoints": outlier_keypoint_count,
    }


def analyze_gt(
    gt_root: Path,
    files: list[Path],
    *,
    expected_shape: tuple[int, ...],
    progress_every: int,
    max_files: int | None,
    sample_limit: int,
    outlier_abs_xy: float,
    outlier_bbox_width: float,
    outlier_bbox_height: float,
    max_outlier_rows: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selected = files[:max_files] if max_files is not None else files
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    issue_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    gt_by_subject_action: dict[tuple[int, int], dict[str, Any]] = {}
    coord_stats = {
        "x": NumericSummary(sample_limit=sample_limit),
        "y": NumericSummary(sample_limit=sample_limit),
    }
    discarded_last_dim_stats = NumericSummary(sample_limit=sample_limit)
    bbox_stats = {
        "valid_keypoints_per_frame": NumericSummary(sample_limit=sample_limit),
        "width": NumericSummary(sample_limit=sample_limit),
        "height": NumericSummary(sample_limit=sample_limit),
        "area": NumericSummary(sample_limit=sample_limit),
        "center_x": NumericSummary(sample_limit=sample_limit),
        "center_y": NumericSummary(sample_limit=sample_limit),
    }
    motion_stats = NumericSummary(sample_limit=sample_limit)
    keypoint_stats = [
        {
            "x": NumericSummary(sample_limit=sample_limit),
            "y": NumericSummary(sample_limit=sample_limit),
        }
        for _ in range(17)
    ]
    env_counts: Counter[int] = Counter()
    frame_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []

    start = time.time()
    for idx, path in enumerate(selected, start=1):
        parsed = gt_match(path)
        if parsed is None:
            issue_rows.append({"path": path.name, "issue": "unexpected_gt_filename_pattern"})
            continue
        env, subject, action = parsed
        env_counts[env] += 1
        try:
            gt = np.load(path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - scanner must keep going.
            issue_rows.append({"path": path.name, "issue": "gt_read_error", "detail": repr(exc)})
            continue

        shape = tuple(int(v) for v in gt.shape)
        shape_counts[shape_text(shape)] += 1
        dtype_counts[str(gt.dtype)] += 1
        if expected_shape and shape != expected_shape:
            issue_rows.append(
                {
                    "path": path.name,
                    "issue": "unexpected_gt_shape",
                    "detail": f"expected={shape_text(expected_shape)} actual={shape_text(shape)}",
                }
            )
            sequence_rows.append(
                {
                    "env": env,
                    "subject": subject,
                    "action": action,
                    "gt_file": path.name,
                    "gt_shape": shape_text(shape),
                    "gt_dtype": str(gt.dtype),
                    "gt_issue": "unexpected_shape",
                }
            )
            continue

        gt_float = gt.astype(np.float64, copy=False)
        xy = gt_float[..., :2]
        if gt_float.shape[-1] > 2:
            last_dim = gt_float[..., 2]
            discarded_last_dim_stats.update(last_dim[np.isfinite(last_dim)], rng)
        invalid_xy = ~np.isfinite(xy).all(axis=-1)
        zero_xy = np.all(xy == 0.0, axis=-1)
        valid_xy = ~(invalid_xy | zero_xy)

        coord_stats["x"].update(xy[..., 0][valid_xy], rng)
        coord_stats["y"].update(xy[..., 1][valid_xy], rng)
        update_keypoint_stats(gt_float, keypoint_stats, rng)
        bbox_and_motion_stats(gt_float, bbox_stats, motion_stats, rng)
        remaining_outliers = max(max_outlier_rows - len(outlier_rows), 0)
        file_frame_rows, file_outlier_rows, detail_counts = build_gt_2d_detail_rows(
            gt_float,
            env=env,
            subject=subject,
            action=action,
            gt_file=path.name,
            outlier_abs_xy=outlier_abs_xy,
            outlier_bbox_width=outlier_bbox_width,
            outlier_bbox_height=outlier_bbox_height,
            remaining_outlier_rows=remaining_outliers,
        )
        frame_rows.extend(file_frame_rows)
        outlier_rows.extend(file_outlier_rows)

        issue_bits = []
        if bool(invalid_xy.any()):
            issue_bits.append("nonfinite_xy")
        if int(valid_xy.sum()) == 0:
            issue_bits.append("all_xy_invalid_or_zero")
        if detail_counts["outlier_2d_frames"]:
            issue_bits.append("2d_outlier_frames")
        if issue_bits:
            issue_rows.append(
                {
                    "path": path.name,
                    "issue": "|".join(issue_bits),
                    "invalid_xy": int(invalid_xy.sum()),
                    "zero_xy": int(zero_xy.sum()),
                    "outlier_2d_frames": detail_counts["outlier_2d_frames"],
                    "outlier_2d_keypoints": detail_counts["outlier_2d_keypoints"],
                }
            )

        row = {
            "env": env,
            "subject": subject,
            "action": action,
            "gt_file": path.name,
            "gt_shape": shape_text(shape),
            "gt_dtype": str(gt.dtype),
            "target_dims": "xy_2d",
            "valid_2d_keypoints": int(valid_xy.sum()),
            "zero_xy_keypoints": int(zero_xy.sum()),
            "invalid_xy_keypoints": int(invalid_xy.sum()),
            "outlier_2d_frames": detail_counts["outlier_2d_frames"],
            "outlier_2d_keypoints": detail_counts["outlier_2d_keypoints"],
            "gt_issue": "|".join(issue_bits),
        }
        sequence_rows.append(row)
        gt_by_subject_action[(subject, action)] = row

        if progress_every and idx % progress_every == 0:
            elapsed = max(time.time() - start, 1.0e-9)
            progress(f"GT read files={idx}/{len(selected)} rate={idx / elapsed:.1f}/s issues={len(issue_rows)}")

    x = coord_stats["x"].as_dict()
    y = coord_stats["y"].as_dict()
    coord_format = infer_coord_format(x.get("min"), x.get("max"), y.get("min"), y.get("max"))
    keypoint_rows = []
    for idx, stats in enumerate(keypoint_stats):
        keypoint_rows.append(
            {
                "joint_index": idx,
                "x": json.dumps(stats["x"].as_dict(), sort_keys=True),
                "y": json.dumps(stats["y"].as_dict(), sort_keys=True),
            }
        )

    summary = {
        "files": len(files),
        "files_analyzed": len(selected),
        "target_dims": "xy_2d",
        "discarded_raw_dims": [2] if expected_shape and len(expected_shape) == 3 and expected_shape[-1] > 2 else [],
        "shape_counts": dict(shape_counts),
        "dtype_counts": dict(dtype_counts),
        "env_counts": {str(k): v for k, v in sorted(env_counts.items())},
        "coord_format_inferred": coord_format,
        "coordinate_stats": {name: stats.as_dict() for name, stats in coord_stats.items()},
        "discarded_last_dim_stats": discarded_last_dim_stats.as_dict(),
        "bbox_stats": {name: stats.as_dict() for name, stats in bbox_stats.items()},
        "motion_per_frame_l2_stats": motion_stats.as_dict(),
        "outlier_thresholds": {
            "abs_xy": outlier_abs_xy,
            "bbox_width": outlier_bbox_width,
            "bbox_height": outlier_bbox_height,
            "max_outlier_rows": max_outlier_rows,
        },
        "outlier_rows_written": len(outlier_rows),
    }
    return summary, sequence_rows, issue_rows, gt_by_subject_action, keypoint_rows, frame_rows, outlier_rows


def merge_sequence_rows(
    csi_rows: list[dict[str, Any]],
    gt_rows_by_key: dict[tuple[int, int], dict[str, Any]],
    csi_sequences: dict[tuple[int, int], CsiSequence],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    matched = 0
    missing_gt = 0
    for row in csi_rows:
        key = (int(row["subject"]), int(row["action"]))
        gt = gt_rows_by_key.get(key)
        out = dict(row)
        if gt:
            matched += 1
            out.update(
                {
                    "env": gt.get("env", ""),
                    "gt_file": gt.get("gt_file", ""),
                    "gt_shape": gt.get("gt_shape", ""),
                    "gt_dtype": gt.get("gt_dtype", ""),
                    "gt_issue": gt.get("gt_issue", ""),
                    "pair_status": "matched",
                }
            )
        else:
            missing_gt += 1
            out.update({"env": "", "gt_file": "", "gt_shape": "", "gt_dtype": "", "gt_issue": "", "pair_status": "missing_gt"})
        merged.append(out)

    csi_keys = set(csi_sequences)
    gt_only = 0
    for (subject, action), gt in sorted(gt_rows_by_key.items()):
        if (subject, action) in csi_keys:
            continue
        gt_only += 1
        merged.append(
            {
                "subject": subject,
                "action": action,
                "csi_rel_dir": "",
                "csi_frame_count": 0,
                "first_frame": "",
                "last_frame": "",
                "missing_frame_count": "",
                "extra_frame_count": "",
                "duplicate_frame_count": "",
                "mat_read_errors": "",
                "nonfinite_files": "",
                "nonfinite_values": "",
                "expected_key_zero_values": "",
                "unexpected_shape_files": "",
                "env": gt.get("env", ""),
                "gt_file": gt.get("gt_file", ""),
                "gt_shape": gt.get("gt_shape", ""),
                "gt_dtype": gt.get("gt_dtype", ""),
                "gt_issue": gt.get("gt_issue", ""),
                "pair_status": "missing_csi",
            }
        )

    return merged, {"matched_sequences": matched, "missing_gt_sequences": missing_gt, "missing_csi_sequences": gt_only}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_previous_outputs(output_root: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        path = output_root / filename
        if path.exists() and path.is_file():
            path.unlink()


def variable_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, stats in summary["variables"].items():
        rows.append(
            {
                "variable": name,
                "count": stats["count"],
                "dtype_counts": json.dumps(stats["dtype_counts"], sort_keys=True),
                "shape_counts": json.dumps(stats["shape_counts"], sort_keys=True),
                "complex_arrays": stats["complex_arrays"],
                "total_values": stats["total_values"],
                "finite_values": stats["finite_values"],
                "nan_values": stats["nan_values"],
                "posinf_values": stats["posinf_values"],
                "neginf_values": stats["neginf_values"],
                "negative_values": stats["negative_values"],
                "zero_values": stats["zero_values"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "quantiles_from_sample": json.dumps(stats["quantiles_from_sample"], sort_keys=True),
                "quantile_sample_values": stats["quantile_sample_values"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw WiFi CSI MAT files and GT pose NPY files.")
    parser.add_argument("--csi-root", default="/data/WiFiPose/dataset/dataset")
    parser.add_argument("--gt-root", default="/data/WiFiPose/dataset/ground_truth_npy")
    parser.add_argument("--output-root", default="outputs/raw_dataset_scan")
    parser.add_argument("--expected-csi-key", default="CSIamp")
    parser.add_argument("--expected-csi-shape", default="3,114,10")
    parser.add_argument("--expected-gt-shape", default="297,17,3")
    parser.add_argument("--expected-frames", type=int, default=297)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--sample-limit", type=int, default=200_000)
    parser.add_argument("--max-mat-files", type=int, default=None)
    parser.add_argument("--max-gt-files", type=int, default=None)
    parser.add_argument("--gt-outlier-abs-xy", type=float, default=2.0)
    parser.add_argument("--gt-outlier-bbox-width", type=float, default=2.0)
    parser.add_argument("--gt-outlier-bbox-height", type=float, default=2.5)
    parser.add_argument("--max-outlier-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csi_root = Path(args.csi_root)
    gt_root = Path(args.gt_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(output_root)
    rng = np.random.default_rng(args.seed)

    expected_csi_shape = parse_shape(args.expected_csi_shape)
    expected_gt_shape = parse_shape(args.expected_gt_shape)

    progress("raw dataset scan started")
    progress(f"csi_root={csi_root}")
    progress(f"gt_root={gt_root}")
    progress(f"output_root={output_root.resolve()}")

    path_summary = {
        "csi_root": str(csi_root),
        "gt_root": str(gt_root),
        "output_root": str(output_root),
        "csi_root_exists": csi_root.exists(),
        "gt_root_exists": gt_root.exists(),
    }
    if not csi_root.exists():
        raise FileNotFoundError(f"CSI root does not exist: {csi_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root does not exist: {gt_root}")

    csi_files, csi_path_issues = discover_csi_files(csi_root, args.progress_every)
    gt_files, gt_name_issues = discover_gt_files(gt_root)

    csi_summary, csi_rows, csi_file_issues, csi_sequences, csi_nonfinite_rows = analyze_csi(
        csi_root,
        csi_files,
        expected_key=args.expected_csi_key,
        expected_shape=expected_csi_shape,
        expected_frames=args.expected_frames,
        progress_every=args.progress_every,
        max_files=args.max_mat_files,
        sample_limit=args.sample_limit,
        rng=rng,
    )
    gt_summary, gt_rows, gt_file_issues, gt_by_key, keypoint_rows, gt_frame_rows, gt_outlier_rows = analyze_gt(
        gt_root,
        gt_files,
        expected_shape=expected_gt_shape,
        progress_every=args.progress_every,
        max_files=args.max_gt_files,
        sample_limit=args.sample_limit,
        outlier_abs_xy=args.gt_outlier_abs_xy,
        outlier_bbox_width=args.gt_outlier_bbox_width,
        outlier_bbox_height=args.gt_outlier_bbox_height,
        max_outlier_rows=args.max_outlier_rows,
        rng=rng,
    )

    sequence_index, pairing_summary = merge_sequence_rows(csi_rows, gt_by_key, csi_sequences)
    all_csi_issues = csi_path_issues + csi_file_issues
    all_gt_issues = gt_name_issues + gt_file_issues

    summary = {
        "generated_at_utc": now_iso(),
        "args": vars(args),
        "paths": path_summary,
        "csi": csi_summary,
        "gt": gt_summary,
        "pairing": pairing_summary,
        "issue_counts": {
            "csi": len(all_csi_issues),
            "gt": len(all_gt_issues),
        },
        "notes": [
            "CSI quantiles are estimated from a deterministic value sample to keep memory bounded.",
            "Complex MAT arrays are summarized by absolute value for numeric distribution fields.",
            "Pairing uses subject/action keys from Axx/Sxx CSI paths and E##_S##_A## GT names.",
            "GT targets are treated as 2D xy only; any raw third dimension is summarized as discarded_last_dim_stats and ignored for target diagnostics.",
        ],
    }

    (output_root / "scan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_root / "sequence_index.csv", sequence_index)
    write_csv(output_root / "csi_file_issues.csv", all_csi_issues)
    write_csv(output_root / "csi_nonfinite_values.csv", csi_nonfinite_rows)
    write_csv(output_root / "csi_sequence_quality.csv", csi_rows)
    write_csv(output_root / "gt_file_issues.csv", all_gt_issues)
    write_csv(output_root / "csi_variable_stats.csv", variable_rows(csi_summary))
    write_csv(output_root / "gt_sequence_stats.csv", gt_rows)
    write_csv(output_root / "gt_2d_keypoint_stats.csv", keypoint_rows)
    write_csv(output_root / "gt_2d_frame_stats.csv", gt_frame_rows)
    write_csv(output_root / "gt_2d_outliers.csv", gt_outlier_rows)

    progress(f"wrote scan outputs to {output_root.resolve()}")
    progress(
        "raw dataset scan finished "
        f"csi_files={csi_summary['mat_files_analyzed']} gt_files={gt_summary['files_analyzed']} "
        f"matched={pairing_summary['matched_sequences']} csi_issues={len(all_csi_issues)} gt_issues={len(all_gt_issues)}"
    )


if __name__ == "__main__":
    main()
