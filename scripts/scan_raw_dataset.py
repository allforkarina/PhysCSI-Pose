from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GT_RE = re.compile(r"^E(?P<env>\d+)_S(?P<subject>\d+)_A(?P<action>\d+)\.npy$", re.IGNORECASE)
CSI_RE = re.compile(
    r"^A(?P<action>\d+)/S(?P<subject>\d+)/wifi-csi/frame(?P<frame>\d+)\.mat$",
    re.IGNORECASE,
)
H36M17_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


@dataclass(frozen=True)
class GTRecord:
    path: Path
    env: int
    subject: int
    action: int
    shape: tuple[int, ...]
    dtype: str
    frames: int


@dataclass(frozen=True)
class CSIRecord:
    path: Path
    subject: int
    action: int
    frame: int
    shape: tuple[int, ...] | None
    dtype: str
    keys: tuple[str, ...]
    readable: bool


class StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.nan_count = 0
        self.posinf_count = 0
        self.neginf_count = 0
        self.zero_count = 0
        self.negative_count = 0
        self.samples: list[float] = []

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values)
        self.nan_count += int(np.isnan(array).sum())
        self.posinf_count += int(np.isposinf(array).sum())
        self.neginf_count += int(np.isneginf(array).sum())
        finite = array[np.isfinite(array)].astype(np.float64, copy=False)
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.sum += float(finite.sum(dtype=np.float64))
        self.sum_sq += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
        self.min = min(self.min, float(finite.min()))
        self.max = max(self.max, float(finite.max()))
        self.zero_count += int((finite == 0.0).sum())
        self.negative_count += int((finite < 0.0).sum())
        if len(self.samples) < 200_000:
            remaining = 200_000 - len(self.samples)
            self.samples.extend(finite.reshape(-1)[:remaining].tolist())

    def to_dict(self) -> dict[str, Any]:
        mean = self.sum / self.count if self.count else None
        variance = max(self.sum_sq / self.count - float(mean) ** 2, 0.0) if self.count and mean is not None else None
        quantiles = {}
        if self.samples:
            sample = np.asarray(self.samples, dtype=np.float64)
            for q in (1, 5, 50, 95, 99):
                quantiles[f"p{q}"] = float(np.percentile(sample, q))
        return {
            "finite_count": self.count,
            "nan_count": self.nan_count,
            "posinf_count": self.posinf_count,
            "neginf_count": self.neginf_count,
            "zero_count": self.zero_count,
            "negative_count": self.negative_count,
            "zero_ratio": self.zero_count / self.count if self.count else None,
            "negative_ratio": self.negative_count / self.count if self.count else None,
            "min": None if self.min == math.inf else self.min,
            "max": None if self.max == -math.inf else self.max,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "quantiles": quantiles,
        }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_root)
    logger.info("starting raw dataset audit")
    result = run_audit(
        csi_root=Path(args.csi_root),
        gt_root=Path(args.gt_root),
        output_root=output_root,
        csi_key=args.csi_key,
        target_time_steps=args.target_time_steps,
        sample_visualizations=args.sample_visualizations,
        max_files=args.max_files,
        logger=logger,
    )
    logger.info("audit complete: %s", result["audit_summary_path"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for raw WiFi CSI amplitude and Human3.6M-17 GT data.")
    parser.add_argument("--csi-root", required=True, help="Raw CSI root, e.g. /data/WiFiPose/dataset/dataset")
    parser.add_argument("--gt-root", required=True, help="Raw GT root, e.g. /data/WiFiPose/dataset/ground_truth_npy")
    parser.add_argument("--output-root", required=True, help="Directory for audit outputs.")
    parser.add_argument("--csi-key", default="CSIamp")
    parser.add_argument("--target-time-steps", type=int, default=64)
    parser.add_argument("--sample-visualizations", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=None, help="Optional debug limit for scanned CSI files.")
    return parser.parse_args(argv)


def run_audit(
    *,
    csi_root: Path,
    gt_root: Path,
    output_root: Path,
    csi_key: str,
    target_time_steps: int,
    sample_visualizations: int,
    max_files: int | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    abnormal_rows: list[dict[str, Any]] = []
    gt_records, gt_stats, gt_shape_hist = scan_gt(gt_root, abnormal_rows, logger)
    csi_records, csi_stats, csi_shape_hist = scan_csi(csi_root, csi_key, abnormal_rows, logger, max_files=max_files)
    canonical_csi_shape = most_common_shape(csi_shape_hist)
    if canonical_csi_shape is not None:
        for record in csi_records:
            if record.shape is not None and record.shape != canonical_csi_shape:
                abnormal_rows.append(
                    {
                        "path": str(record.path),
                        "kind": "csi",
                        "reason": "unexpected_csi_shape",
                        "detail": f"shape={shape_text(record.shape)} expected={shape_text(canonical_csi_shape)}",
                    }
                )

    sequence_rows, pairing_errors, missing_rows, sampled_pairings = inspect_pairings(gt_records, csi_records)
    blocking_errors = detect_blocking_errors(gt_records, sequence_rows)
    fixed_sequence_length = len({record.frames for record in gt_records}) == 1 if gt_records else False
    contract = build_contract(
        gt_records=gt_records,
        csi_records=csi_records,
        gt_stats=gt_stats,
        csi_stats=csi_stats,
        canonical_csi_shape=canonical_csi_shape,
        csi_key=csi_key,
        fixed_sequence_length=fixed_sequence_length,
        pairing_errors=pairing_errors,
        blocking_errors=blocking_errors,
        target_time_steps=target_time_steps,
    )
    summary = build_summary(
        gt_records=gt_records,
        csi_records=csi_records,
        abnormal_rows=abnormal_rows,
        pairing_errors=pairing_errors,
        blocking_errors=blocking_errors,
        contract=contract,
    )

    write_csv(output_root / "sequence_inventory.csv", sequence_rows)
    write_csv(output_root / "csi_file_inventory.csv", [csi_record_to_row(record) for record in csi_records])
    write_csv(output_root / "gt_file_inventory.csv", [gt_record_to_row(record) for record in gt_records])
    write_csv(output_root / "pairing_errors.csv", pairing_errors)
    write_csv(output_root / "missing_frames.csv", missing_rows)
    write_csv(output_root / "abnormal_files.csv", abnormal_rows)
    write_csv(output_root / "sampled_pairings.csv", sampled_pairings[:100])
    write_distribution_csv(output_root / "environment_statistics.csv", gt_records, "env")
    write_distribution_csv(output_root / "subject_statistics.csv", gt_records, "subject")
    write_distribution_csv(output_root / "action_statistics.csv", gt_records, "action")
    write_json(output_root / "audit_summary.json", summary)
    write_json(output_root / "csi_shape_histogram.json", {shape_text(k): v for k, v in csi_shape_hist.items()})
    write_json(output_root / "gt_shape_histogram.json", {shape_text(k): v for k, v in gt_shape_hist.items()})
    write_json(output_root / "csi_statistics.json", csi_stats)
    write_json(output_root / "gt_statistics.json", gt_stats)
    (output_root / "architecture_contract.yaml").write_text(to_yaml(contract), encoding="utf-8")
    write_visualizations(output_root / "visualizations", gt_records, sample_visualizations)
    return {"audit_summary_path": str(output_root / "audit_summary.json")}


def scan_gt(
    gt_root: Path,
    abnormal_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[list[GTRecord], dict[str, Any], Counter[tuple[int, ...]]]:
    records: list[GTRecord] = []
    shape_hist: Counter[tuple[int, ...]] = Counter()
    stats = StreamingStats()
    third_dim_stats = StreamingStats()
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bone_lengths: defaultdict[str, StreamingStats] = defaultdict(StreamingStats)

    for path in sorted(gt_root.glob("*.npy")):
        parsed = parse_gt_path(path)
        if parsed is None:
            abnormal_rows.append({"path": str(path), "kind": "gt", "reason": "bad_gt_filename", "detail": path.name})
            continue
        env, subject, action = parsed
        try:
            gt = np.load(path, allow_pickle=False)
        except Exception as error:  # pragma: no cover - defensive for corrupted files
            logger.warning("failed to read GT %s: %s", path, error)
            abnormal_rows.append({"path": str(path), "kind": "gt", "reason": "gt_read_error", "detail": str(error)})
            continue
        shape = tuple(int(v) for v in gt.shape)
        shape_hist[shape] += 1
        if gt.ndim != 3 or gt.shape[1] != 17 or gt.shape[2] < 2:
            abnormal_rows.append({"path": str(path), "kind": "gt", "reason": "unexpected_gt_shape", "detail": shape_text(shape)})
        if np.isnan(gt).any() or np.isinf(gt).any():
            abnormal_rows.append({"path": str(path), "kind": "gt", "reason": "nonfinite_gt_values", "detail": ""})
        xy = gt[..., :2]
        stats.update(xy)
        if gt.ndim == 3 and gt.shape[-1] >= 3:
            third_dim_stats.update(gt[..., 2])
        for frame in xy:
            width = float(np.nanmax(frame[:, 0]) - np.nanmin(frame[:, 0]))
            height = float(np.nanmax(frame[:, 1]) - np.nanmin(frame[:, 1]))
            bbox_widths.append(width)
            bbox_heights.append(height)
            for edge in H36M17_EDGES:
                length = float(np.linalg.norm(frame[edge[0]] - frame[edge[1]]))
                bone_lengths[f"{edge[0]}-{edge[1]}"].update(np.asarray([length], dtype=np.float64))
        records.append(GTRecord(path=path, env=env, subject=subject, action=action, shape=shape, dtype=str(gt.dtype), frames=int(gt.shape[0])))

    gt_stats = stats.to_dict()
    gt_stats["third_dimension"] = third_dim_stats.to_dict()
    gt_stats["bbox_width"] = summarize_values(bbox_widths)
    gt_stats["bbox_height"] = summarize_values(bbox_heights)
    gt_stats["bone_lengths"] = {edge: value.to_dict() for edge, value in bone_lengths.items()}
    return records, gt_stats, shape_hist


def scan_csi(
    csi_root: Path,
    csi_key: str,
    abnormal_rows: list[dict[str, Any]],
    logger: logging.Logger,
    *,
    max_files: int | None,
) -> tuple[list[CSIRecord], dict[str, Any], Counter[tuple[int, ...]]]:
    records: list[CSIRecord] = []
    shape_hist: Counter[tuple[int, ...]] = Counter()
    stats = StreamingStats()
    per_antenna: defaultdict[int, StreamingStats] = defaultdict(StreamingStats)
    per_subcarrier: defaultdict[int, StreamingStats] = defaultdict(StreamingStats)
    per_time: defaultdict[int, StreamingStats] = defaultdict(StreamingStats)
    full_zero_frames = 0
    full_constant_frames = 0

    paths = sorted(csi_root.glob("A*/S*/wifi-csi/frame*.mat"))
    if max_files is not None:
        paths = paths[:max_files]
    for path in paths:
        parsed = parse_csi_path(csi_root, path)
        if parsed is None:
            abnormal_rows.append({"path": str(path), "kind": "csi", "reason": "bad_csi_filename", "detail": ""})
            continue
        action, subject, frame = parsed
        try:
            keys, value = read_mat_array(path, csi_key)
        except KeyError as error:
            keys = tuple(list_mat_keys(path))
            abnormal_rows.append({"path": str(path), "kind": "csi", "reason": "missing_csi_key", "detail": str(error)})
            records.append(CSIRecord(path=path, subject=subject, action=action, frame=frame, shape=None, dtype="", keys=keys, readable=False))
            continue
        except Exception as error:  # pragma: no cover - defensive for corrupted files
            logger.warning("failed to read CSI %s: %s", path, error)
            abnormal_rows.append({"path": str(path), "kind": "csi", "reason": "csi_read_error", "detail": str(error)})
            records.append(CSIRecord(path=path, subject=subject, action=action, frame=frame, shape=None, dtype="", keys=(), readable=False))
            continue

        shape = tuple(int(v) for v in value.shape)
        shape_hist[shape] += 1
        if np.iscomplexobj(value):
            abnormal_rows.append({"path": str(path), "kind": "csi", "reason": "complex_csi_values", "detail": ""})
        if np.isnan(value).any() or np.isinf(value).any():
            abnormal_rows.append({"path": str(path), "kind": "csi", "reason": "nonfinite_csi_values", "detail": ""})
        finite = value[np.isfinite(value)]
        if finite.size and bool(np.all(finite == 0)):
            full_zero_frames += 1
        if finite.size and float(np.nanmin(finite)) == float(np.nanmax(finite)):
            full_constant_frames += 1
        stats.update(value)
        if value.ndim == 3:
            for ant in range(value.shape[0]):
                per_antenna[ant].update(value[ant])
            for sub in range(value.shape[1]):
                per_subcarrier[sub].update(value[:, sub, :])
            for tick in range(value.shape[2]):
                per_time[tick].update(value[:, :, tick])
        records.append(CSIRecord(path=path, subject=subject, action=action, frame=frame, shape=shape, dtype=str(value.dtype), keys=keys, readable=True))

    csi_stats = stats.to_dict()
    csi_stats["full_zero_frames"] = full_zero_frames
    csi_stats["full_constant_frames"] = full_constant_frames
    csi_stats["per_antenna"] = {str(k): v.to_dict() for k, v in per_antenna.items()}
    csi_stats["per_subcarrier"] = {str(k): v.to_dict() for k, v in per_subcarrier.items()}
    csi_stats["per_time"] = {str(k): v.to_dict() for k, v in per_time.items()}
    return records, csi_stats, shape_hist


def inspect_pairings(
    gt_records: list[GTRecord],
    csi_records: list[CSIRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    csi_by_key: defaultdict[tuple[int, int], list[CSIRecord]] = defaultdict(list)
    for record in csi_records:
        csi_by_key[(record.subject, record.action)].append(record)

    sequence_rows = []
    pairing_errors = []
    missing_rows = []
    sampled_pairings = []
    for gt in gt_records:
        csi = sorted(csi_by_key.get((gt.subject, gt.action), []), key=lambda item: item.frame)
        frame_numbers = [record.frame for record in csi]
        expected = set(range(1, gt.frames + 1))
        observed = set(frame_numbers)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if len(frame_numbers) != gt.frames or missing or extra:
            pairing_errors.append(
                {
                    "env": gt.env,
                    "subject": gt.subject,
                    "action": gt.action,
                    "reason": "csi_gt_frame_count_mismatch",
                    "gt_frames": gt.frames,
                    "csi_frames": len(frame_numbers),
                }
            )
        for frame in missing:
            missing_rows.append({"env": gt.env, "subject": gt.subject, "action": gt.action, "missing_frame": frame})
        for frame in extra:
            pairing_errors.append({"env": gt.env, "subject": gt.subject, "action": gt.action, "reason": "extra_csi_frame", "frame": frame})
        sequence_rows.append(
            {
                "env": gt.env,
                "subject": gt.subject,
                "action": gt.action,
                "gt_file": str(gt.path),
                "gt_frames": gt.frames,
                "csi_frames": len(frame_numbers),
                "first_csi_frame": min(frame_numbers) if frame_numbers else "",
                "last_csi_frame": max(frame_numbers) if frame_numbers else "",
                "missing_frame_count": len(missing),
                "extra_frame_count": len(extra),
            }
        )
        for index in range(min(gt.frames, 3)):
            csi_path = next((record.path for record in csi if record.frame == index + 1), "")
            sampled_pairings.append(
                {
                    "environment": gt.env,
                    "subject": gt.subject,
                    "action": gt.action,
                    "gt_file": str(gt.path),
                    "gt_frame_index": index,
                    "csi_file": str(csi_path),
                    "csi_frame_index": index + 1,
                }
            )
    return sequence_rows, pairing_errors, missing_rows, sampled_pairings


def detect_blocking_errors(gt_records: list[GTRecord], sequence_rows: list[dict[str, Any]]) -> list[str]:
    blocking: list[str] = []
    envs_by_subject_action: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for record in gt_records:
        envs_by_subject_action[(record.subject, record.action)].add(record.env)
    if any(len(envs) > 1 for envs in envs_by_subject_action.values()):
        blocking.append("same_subject_action_across_environments_without_csi_env_path")
    if any(int(row["missing_frame_count"]) > 0 for row in sequence_rows):
        blocking.append("missing_csi_frames")
    return blocking


def build_contract(
    *,
    gt_records: list[GTRecord],
    csi_records: list[CSIRecord],
    gt_stats: dict[str, Any],
    csi_stats: dict[str, Any],
    canonical_csi_shape: tuple[int, ...] | None,
    csi_key: str,
    fixed_sequence_length: bool,
    pairing_errors: list[dict[str, Any]],
    blocking_errors: list[str],
    target_time_steps: int,
) -> dict[str, Any]:
    frame_values = sorted({record.frames for record in gt_records})
    envs = sorted({record.env for record in gt_records})
    subjects = sorted({record.subject for record in gt_records})
    actions = sorted({record.action for record in gt_records})
    packets = canonical_csi_shape[2] if canonical_csi_shape and len(canonical_csi_shape) == 3 else None
    target_shape = [canonical_csi_shape[0], canonical_csi_shape[1], target_time_steps] if canonical_csi_shape and len(canonical_csi_shape) >= 2 else None
    return {
        "dataset": {
            "environments": len(envs),
            "subjects": len(subjects),
            "actions": len(actions),
            "sequences": len(gt_records),
            "frames_per_sequence": {"unique_values": frame_values},
            "fixed_sequence_length": fixed_sequence_length,
            "indexing": {"use_cumulative_offsets": not fixed_sequence_length},
        },
        "csi": {
            "file_format": "mat",
            "key": csi_key,
            "layout": "antenna,subcarrier,time" if canonical_csi_shape and len(canonical_csi_shape) == 3 else "unknown",
            "antennas": canonical_csi_shape[0] if canonical_csi_shape and len(canonical_csi_shape) == 3 else None,
            "subcarriers": canonical_csi_shape[1] if canonical_csi_shape and len(canonical_csi_shape) == 3 else None,
            "packets_per_pose": {"unique_values": sorted({record.shape[2] for record in csi_records if record.shape and len(record.shape) == 3})},
            "all_shapes_identical": len({record.shape for record in csi_records if record.shape}) <= 1,
            "nonfinite_count": csi_stats["nan_count"] + csi_stats["posinf_count"] + csi_stats["neginf_count"],
            "zero_ratio": csi_stats["zero_ratio"],
            "negative_ratio": csi_stats["negative_ratio"],
        },
        "gt": {
            "file_format": "npy",
            "layout": "frame,joint,dimension",
            "skeleton": "human36m17",
            "joints": 17,
            "dimensions": sorted({record.shape[2] for record in gt_records if len(record.shape) == 3}),
            "used_dimensions": [0, 1],
            "all_keypoints_valid": True,
            "x_range": [gt_stats["min"], gt_stats["max"]],
        },
        "alignment": {
            "mapping": "one_csi_frame_to_one_gt_frame",
            "frame_index_origin": 1,
            "all_sequences_aligned": not pairing_errors,
            "missing_pair_count": len(pairing_errors),
        },
        "recommended_preprocessing": {
            "repair_nonfinite_csi": bool(csi_stats["nan_count"] + csi_stats["posinf_count"] + csi_stats["neginf_count"]),
            "temporal_resampling": {
                "enabled": packets != target_time_steps,
                "source_length": packets,
                "target_length": target_time_steps,
                "method": "fourier",
            },
            "csi_normalization": {"method": "source_train_zscore", "statistics_shape": [1, canonical_csi_shape[0], canonical_csi_shape[1], 1] if canonical_csi_shape and len(canonical_csi_shape) == 3 else None},
            "gt": {"use_xy_only": True},
        },
        "recommended_model": {
            "input_shape": target_shape,
            "output_shape": [17, 2],
            "wavelet_axis": "time",
            "maximum_swt_level": max_swt_level(target_time_steps),
            "coarse_frequency_tokens_after_two_stride2": downsampled_length(canonical_csi_shape[1]) if canonical_csi_shape and len(canonical_csi_shape) == 3 else None,
        },
        "blocking_errors": blocking_errors,
    }


def build_summary(
    *,
    gt_records: list[GTRecord],
    csi_records: list[CSIRecord],
    abnormal_rows: list[dict[str, Any]],
    pairing_errors: list[dict[str, Any]],
    blocking_errors: list[str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected = (
        contract["csi"]["antennas"] == 3
        and contract["csi"]["subcarriers"] == 114
        and contract["csi"]["packets_per_pose"]["unique_values"] == [10]
        and contract["gt"]["joints"] == 17
        and contract["dataset"]["fixed_sequence_length"]
        and not blocking_errors
    )
    return {
        "scanned_files": len(gt_records) + len(csi_records),
        "successful_gt_files": len(gt_records),
        "successful_csi_files": sum(1 for record in csi_records if record.readable),
        "failed_files": sum(1 for record in csi_records if not record.readable),
        "abnormal_count": len(abnormal_rows),
        "pairing_error_count": len(pairing_errors),
        "satisfies_current_project_assumptions": expected,
        "blocking_errors": blocking_errors,
        "non_blocking_warnings": sorted({row["reason"] for row in abnormal_rows}),
    }


def read_mat_array(path: Path, key: str) -> tuple[tuple[str, ...], np.ndarray]:
    try:
        loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        keys = tuple(sorted(k for k in loaded if not k.startswith("__")))
        if key not in loaded:
            raise KeyError(f"{key!r} not found; available keys={keys}")
        return keys, np.asarray(loaded[key])
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            keys = tuple(sorted(handle.keys()))
            if key not in handle:
                raise KeyError(f"{key!r} not found; available keys={keys}")
            return keys, np.asarray(handle[key])


def list_mat_keys(path: Path) -> list[str]:
    try:
        loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        return sorted(k for k in loaded if not k.startswith("__"))
    except Exception:
        try:
            with h5py.File(path, "r") as handle:
                return sorted(handle.keys())
        except Exception:
            return []


def parse_gt_path(path: Path) -> tuple[int, int, int] | None:
    match = GT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("env")), int(match.group("subject")), int(match.group("action"))


def parse_csi_path(root: Path, path: Path) -> tuple[int, int, int] | None:
    rel = path.relative_to(root).as_posix()
    match = CSI_RE.match(rel)
    if match is None:
        return None
    return int(match.group("action")), int(match.group("subject")), int(match.group("frame"))


def most_common_shape(hist: Counter[tuple[int, ...]]) -> tuple[int, ...] | None:
    if not hist:
        return None
    return sorted(hist.items(), key=lambda item: (item[1], int(np.prod(item[0]))), reverse=True)[0][0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_distribution_csv(path: Path, gt_records: list[GTRecord], field: str) -> None:
    counts: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"sequences": 0, "frames": 0})
    for record in gt_records:
        key = int(getattr(record, field))
        counts[key]["sequences"] += 1
        counts[key]["frames"] += record.frames
    rows = [{field: key, **value} for key, value in sorted(counts.items())]
    write_csv(path, rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_visualizations(output_dir: Path, gt_records: list[GTRecord], count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(gt_records[:count], start=1):
        gt = np.load(record.path, allow_pickle=False)
        xy = gt[0, :, :2]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xy[:, 0], xy[:, 1], s=16)
        for left, right in H36M17_EDGES:
            ax.plot([xy[left, 0], xy[right, 0]], [xy[left, 1], xy[right, 1]], linewidth=1)
        ax.set_title(f"E{record.env:02d} S{record.subject:02d} A{record.action:02d} frame 0")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{index:02d}_E{record.env:02d}_S{record.subject:02d}_A{record.action:02d}.png")
        plt.close(fig)


def to_yaml(value: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict) or (isinstance(item, list) and any(isinstance(child, (dict, list)) for child in item)):
                lines.append(f"{spaces}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {format_yaml_scalar(item)}")
        return "\n".join(lines) + ("\n" if indent == 0 else "")
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return f"{spaces}{format_yaml_scalar(value)}"
        return "\n".join(f"{spaces}- {format_yaml_scalar(item)}" for item in value)
    return f"{spaces}{format_yaml_scalar(value)}"


def format_yaml_scalar(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(format_yaml_scalar(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return str(value)


def csi_record_to_row(record: CSIRecord) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "subject": record.subject,
        "action": record.action,
        "frame": record.frame,
        "shape": "" if record.shape is None else shape_text(record.shape),
        "dtype": record.dtype,
        "keys": ";".join(record.keys),
        "readable": record.readable,
    }


def gt_record_to_row(record: GTRecord) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "env": record.env,
        "subject": record.subject,
        "action": record.action,
        "shape": shape_text(record.shape),
        "dtype": record.dtype,
        "frames": record.frames,
    }


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std())}


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def max_swt_level(length: int) -> int:
    level = 0
    while length > 0 and length % (2 ** (level + 1)) == 0:
        level += 1
    return level


def downsampled_length(length: int) -> int:
    return int(math.ceil(int(math.ceil(length / 2)) / 2))


def configure_logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger("raw_dataset_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_root / "audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    main()
