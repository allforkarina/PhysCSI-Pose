from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import scipy.signal


GT_RE = re.compile(r"^E(?P<env>\d+)_S(?P<subject>\d+)_A(?P<action>\d+)\.npy$", re.IGNORECASE)
OUTPUT_FILES = (
    "X_amp_resampled.npy",
    "X_amp_clean.npy",
    "Y_2d_clean.npy",
    "repair_counts.npy",
    "meta.npz",
    "sequence_meta.csv",
    "frame_meta.csv",
    "splits_by_env.json",
    "split_index.npz",
    "clean_manifest.json",
)


def progress(message: str) -> None:
    print(f"[progress] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(v)) for v in shape)


def read_mat_key(path: Path, key: str) -> np.ndarray:
    try:
        loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        if key not in loaded:
            visible = sorted(k for k in loaded if not k.startswith("__"))
            raise KeyError(f"{key!r} not found in {path}; available keys={visible}")
        return np.asarray(loaded[key])
    except (NotImplementedError, ValueError, OSError):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"{key!r} not found in {path}; available keys={list(handle.keys())}")
            return np.asarray(handle[key])


def parse_gt_name(path: Path) -> tuple[int, int, int]:
    match = GT_RE.match(path.name)
    if match is None:
        raise ValueError(f"GT filename does not match E##_S##_A##.npy: {path.name}")
    return int(match.group("env")), int(match.group("subject")), int(match.group("action"))


def csi_path(csi_root: Path, subject: int, action: int, frame_id: int) -> Path:
    return csi_root / f"A{action:02d}" / f"S{subject:02d}" / "wifi-csi" / f"frame{frame_id:03d}.mat"


def discover_sequences(gt_root: Path) -> list[dict[str, Any]]:
    sequences = []
    for path in sorted(gt_root.glob("*.npy")):
        env, subject, action = parse_gt_name(path)
        sequences.append(
            {
                "sequence_id": len(sequences),
                "env": env,
                "subject": subject,
                "action": action,
                "gt_path": path,
                "gt_file": path.name,
            }
        )
    if not sequences:
        raise FileNotFoundError(f"no GT .npy files found under {gt_root}")
    return sequences


def finite_window_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean(dtype=np.float64))


def repair_csiamp_frame(
    frame: np.ndarray,
    *,
    subcarrier_radius: int = 2,
    time_radius: int = 1,
) -> tuple[np.ndarray, dict[str, int]]:
    if frame.ndim != 3:
        raise ValueError(f"CSIamp frame must be 3D [antenna, subcarrier, time], got {frame.shape}")

    cleaned = frame.astype(np.float32, copy=True)
    nonfinite = ~np.isfinite(cleaned)
    stats = {
        "nonfinite_values": int(nonfinite.sum()),
        "nan_values": int(np.isnan(cleaned).sum()),
        "posinf_values": int(np.isposinf(cleaned).sum()),
        "neginf_values": int(np.isneginf(cleaned).sum()),
        "zero_values": int((cleaned == 0.0).sum()),
        "fallback_repairs": 0,
    }
    if not bool(nonfinite.any()):
        return cleaned, stats

    ant_count, sub_count, time_count = cleaned.shape
    original = cleaned.copy()
    frame_fallback = finite_window_mean(original)
    if frame_fallback is None:
        frame_fallback = 0.0

    for ant, sub, tick in np.argwhere(nonfinite):
        sub_start = max(0, int(sub) - subcarrier_radius)
        sub_end = min(sub_count, int(sub) + subcarrier_radius + 1)
        time_start = max(0, int(tick) - time_radius)
        time_end = min(time_count, int(tick) + time_radius + 1)
        window = original[int(ant), sub_start:sub_end, time_start:time_end]
        replacement = finite_window_mean(window)
        if replacement is None:
            antenna_fallback = finite_window_mean(original[int(ant)])
            replacement = frame_fallback if antenna_fallback is None else antenna_fallback
            stats["fallback_repairs"] += 1
        cleaned[int(ant), int(sub), int(tick)] = replacement

    return cleaned, stats


def standardize_csiamp(frame: np.ndarray, expected_shape: tuple[int, ...], path: Path) -> np.ndarray:
    if tuple(frame.shape) != expected_shape:
        raise ValueError(f"{path} CSIamp shape expected {shape_text(expected_shape)}, got {shape_text(tuple(frame.shape))}")
    if len(expected_shape) != 3:
        raise ValueError(f"expected CSI shape must be 3D, got {expected_shape}")
    return frame.astype(np.float32, copy=False)


def resample_csiamp_time(frame: np.ndarray, output_time_steps: int) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"CSIamp frame must be 3D [antenna, subcarrier, time], got {frame.shape}")
    if output_time_steps <= 0:
        raise ValueError(f"output_time_steps must be positive, got {output_time_steps}")
    return scipy.signal.resample(frame, int(output_time_steps), axis=2).astype(np.float32, copy=False)


def frame_bbox_outlier(
    xy_frame: np.ndarray,
    *,
    abs_xy: float,
    bbox_width: float,
    bbox_height: float,
) -> bool:
    finite = np.isfinite(xy_frame).all(axis=-1) & ~np.all(xy_frame == 0.0, axis=-1)
    if not bool(finite.any()):
        return True
    pts = xy_frame[finite]
    if bool((np.abs(pts) > abs_xy).any()):
        return True
    width = float(pts[:, 0].max() - pts[:, 0].min())
    height = float(pts[:, 1].max() - pts[:, 1].min())
    return width > bbox_width or height > bbox_height


def estimate_gt_source_range(
    sequences: list[dict[str, Any]],
    *,
    expected_shape: tuple[int, ...],
    abs_xy: float,
    bbox_width: float,
    bbox_height: float,
) -> dict[str, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ignored_frames = 0
    for seq in sequences:
        gt = np.load(seq["gt_path"], allow_pickle=False)
        if tuple(gt.shape) != expected_shape:
            raise ValueError(f"{seq['gt_path']} GT shape expected {shape_text(expected_shape)}, got {shape_text(tuple(gt.shape))}")
        xy = gt[..., :2].astype(np.float64, copy=False)
        for frame in xy:
            if frame_bbox_outlier(frame, abs_xy=abs_xy, bbox_width=bbox_width, bbox_height=bbox_height):
                ignored_frames += 1
                continue
            valid = np.isfinite(frame).all(axis=-1) & ~np.all(frame == 0.0, axis=-1)
            if bool(valid.any()):
                xs.append(frame[valid, 0].copy())
                ys.append(frame[valid, 1].copy())

    if not xs or not ys:
        raise ValueError("unable to estimate GT source range; no valid non-outlier 2D coordinates found")

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    return {
        "x_min": float(x_all.min()),
        "x_max": float(x_all.max()),
        "y_min": float(y_all.min()),
        "y_max": float(y_all.max()),
        "ignored_outlier_frames_for_range": ignored_frames,
    }


def normalize_axis(values: np.ndarray, source_min: float, source_max: float, target_min: float, target_max: float) -> np.ndarray:
    if not source_max > source_min:
        raise ValueError(f"source_max must be greater than source_min, got {source_min}, {source_max}")
    clipped = np.clip(values, source_min, source_max)
    scaled = (clipped - source_min) / (source_max - source_min)
    return scaled * (target_max - target_min) + target_min


def normalize_gt_xy(
    gt: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    target_min: float = -0.8,
    target_max: float = 0.8,
) -> tuple[np.ndarray, dict[str, int]]:
    if gt.ndim != 3 or gt.shape[-1] < 2:
        raise ValueError(f"GT must have shape [frames, joints, dims>=2], got {gt.shape}")
    xy = gt[..., :2].astype(np.float32, copy=True)
    finite_xy = np.isfinite(xy).all(axis=-1)
    zero_xy = np.all(xy == 0.0, axis=-1)
    invalid = (~finite_xy) | zero_xy

    clipped_x = (xy[..., 0] < x_min) | (xy[..., 0] > x_max)
    clipped_y = (xy[..., 1] < y_min) | (xy[..., 1] > y_max)
    clipped_values = int((clipped_x & ~invalid).sum() + (clipped_y & ~invalid).sum())

    xy[..., 0] = normalize_axis(xy[..., 0], x_min, x_max, target_min, target_max)
    xy[..., 1] = normalize_axis(xy[..., 1], y_min, y_max, target_min, target_max)
    xy = np.clip(xy, target_min, target_max).astype(np.float32, copy=False)
    xy[invalid] = 0.0

    return xy, {
        "invalid_or_zero_keypoints": int(invalid.sum()),
        "clipped_keypoint_values": clipped_values,
        "clipped_x_values": int((clipped_x & ~invalid).sum()),
        "clipped_y_values": int((clipped_y & ~invalid).sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    existing = [output_root / name for name in OUTPUT_FILES if (output_root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output files already exist under {output_root}; pass --overwrite")
    if existing:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def resolve_output_root(output_root_arg: str | None, csi_root: Path) -> Path:
    if output_root_arg:
        return Path(output_root_arg)
    return csi_root.parent / "clean_dataset"


def build_splits(sequence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_to_sequence_ids: dict[str, list[int]] = {}
    for row in sequence_rows:
        env_to_sequence_ids.setdefault(str(row["env"]), []).append(int(row["sequence_id"]))

    env_ids = sorted(int(env) for env in env_to_sequence_ids)
    leave_one_env_out = {}
    for eval_env in env_ids:
        train_envs = [env for env in env_ids if env != eval_env]
        train_ids = [
            int(row["sequence_id"])
            for row in sequence_rows
            if int(row["env"]) in train_envs
        ]
        eval_ids = [
            int(row["sequence_id"])
            for row in sequence_rows
            if int(row["env"]) == eval_env
        ]
        leave_one_env_out[str(eval_env)] = {
            "train_envs": train_envs,
            "eval_envs": [eval_env],
            "train_sequence_ids": train_ids,
            "eval_sequence_ids": eval_ids,
        }

    return {
        "strategy": "env_id",
        "env_to_sequence_ids": {env: sorted(ids) for env, ids in env_to_sequence_ids.items()},
        "leave_one_env_out": leave_one_env_out,
    }


def write_split_index_npz(path: Path, splits: dict[str, Any], expected_frames: int) -> None:
    arrays: dict[str, np.ndarray] = {}
    for env, sequence_ids in splits["env_to_sequence_ids"].items():
        ids = np.asarray(sequence_ids, dtype=np.int32)
        arrays[f"env_{env}_sequence_ids"] = ids
        arrays[f"env_{env}_frame_indices"] = sequence_ids_to_frame_indices(ids, expected_frames)

    for env, split in splits["leave_one_env_out"].items():
        train_ids = np.asarray(split["train_sequence_ids"], dtype=np.int32)
        eval_ids = np.asarray(split["eval_sequence_ids"], dtype=np.int32)
        arrays[f"env_{env}_train_sequence_ids"] = train_ids
        arrays[f"env_{env}_eval_sequence_ids"] = eval_ids
        arrays[f"env_{env}_train_frame_indices"] = sequence_ids_to_frame_indices(train_ids, expected_frames)
        arrays[f"env_{env}_eval_frame_indices"] = sequence_ids_to_frame_indices(eval_ids, expected_frames)

    np.savez(path, **arrays)


def sequence_ids_to_frame_indices(sequence_ids: np.ndarray, expected_frames: int) -> np.ndarray:
    if sequence_ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    offsets = sequence_ids.astype(np.int64)[:, None] * int(expected_frames)
    frames = np.arange(int(expected_frames), dtype=np.int64)[None, :]
    return (offsets + frames).reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned CSIamp and 2D GT memmaps from raw WiFiPose data.")
    parser.add_argument("--csi-root", default="/data/WiFiPose/dataset/dataset")
    parser.add_argument("--gt-root", default="/data/WiFiPose/dataset/ground_truth_npy")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to <csi-root parent>/clean_dataset, e.g. /data/WiFiPose/dataset/clean_dataset.",
    )
    parser.add_argument("--csi-key", default="CSIamp")
    parser.add_argument("--ignored-mat-keys", default="CSIphase")
    parser.add_argument("--expected-csi-shape", default="3,114,10")
    parser.add_argument("--expected-gt-shape", default="297,17,3")
    parser.add_argument("--expected-frames", type=int, default=297)
    parser.add_argument("--resample-time-steps", type=int, default=64)
    parser.add_argument("--repair-subcarrier-radius", type=int, default=2)
    parser.add_argument("--repair-time-radius", type=int, default=1)
    parser.add_argument("--target-min", type=float, default=-0.8)
    parser.add_argument("--target-max", type=float, default=0.8)
    parser.add_argument("--gt-outlier-abs-xy", type=float, default=2.0)
    parser.add_argument("--gt-outlier-bbox-width", type=float, default=2.0)
    parser.add_argument("--gt-outlier-bbox-height", type=float, default=2.5)
    parser.add_argument("--gt-source-x-min", type=float, default=None)
    parser.add_argument("--gt-source-x-max", type=float, default=None)
    parser.add_argument("--gt-source-y-min", type=float, default=None)
    parser.add_argument("--gt-source-y-max", type=float, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csi_root = Path(args.csi_root)
    gt_root = Path(args.gt_root)
    output_root = resolve_output_root(args.output_root, csi_root)
    expected_csi_shape = parse_shape(args.expected_csi_shape)
    expected_gt_shape = parse_shape(args.expected_gt_shape)
    ignored_keys = [key.strip() for key in args.ignored_mat_keys.split(",") if key.strip()]

    if not csi_root.exists():
        raise FileNotFoundError(f"CSI root does not exist: {csi_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root does not exist: {gt_root}")

    ensure_output_root(output_root, args.overwrite)
    sequences = discover_sequences(gt_root)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise ValueError("no sequences selected")

    progress(f"selected sequences={len(sequences)}")
    range_overrides = [args.gt_source_x_min, args.gt_source_x_max, args.gt_source_y_min, args.gt_source_y_max]
    if any(value is not None for value in range_overrides):
        if not all(value is not None for value in range_overrides):
            raise ValueError("provide all GT source range overrides or none")
        gt_range = {
            "x_min": float(args.gt_source_x_min),
            "x_max": float(args.gt_source_x_max),
            "y_min": float(args.gt_source_y_min),
            "y_max": float(args.gt_source_y_max),
            "ignored_outlier_frames_for_range": None,
        }
    else:
        progress("estimating GT 2D source range from non-outlier frames")
        gt_range = estimate_gt_source_range(
            sequences,
            expected_shape=expected_gt_shape,
            abs_xy=args.gt_outlier_abs_xy,
            bbox_width=args.gt_outlier_bbox_width,
            bbox_height=args.gt_outlier_bbox_height,
        )
    progress(f"GT source range={gt_range}")

    frame_shape = (expected_csi_shape[0], expected_csi_shape[1], args.resample_time_steps)
    total_frames = len(sequences) * args.expected_frames
    x_all = np.lib.format.open_memmap(
        output_root / "X_amp_resampled.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, *frame_shape),
    )
    y_all = np.lib.format.open_memmap(
        output_root / "Y_2d_clean.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, expected_gt_shape[1], 2),
    )
    repair_counts = np.lib.format.open_memmap(
        output_root / "repair_counts.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(total_frames,),
    )

    env_arr = np.zeros(total_frames, dtype=np.int16)
    subject_arr = np.zeros(total_frames, dtype=np.int16)
    action_arr = np.zeros(total_frames, dtype=np.int16)
    frame_arr = np.zeros(total_frames, dtype=np.int16)
    sequence_id_arr = np.zeros(total_frames, dtype=np.int32)

    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    totals = {
        "frames": 0,
        "csi_nonfinite_values_repaired": 0,
        "csi_zero_values_kept": 0,
        "csi_fallback_repairs": 0,
        "gt_invalid_or_zero_keypoints": 0,
        "gt_clipped_keypoint_values": 0,
    }

    start = time.time()
    for seq_index, seq in enumerate(sequences):
        gt = np.load(seq["gt_path"], allow_pickle=False)
        if tuple(gt.shape) != expected_gt_shape:
            raise ValueError(f"{seq['gt_path']} GT shape expected {shape_text(expected_gt_shape)}, got {shape_text(tuple(gt.shape))}")
        y_seq, gt_stats = normalize_gt_xy(
            gt,
            x_min=gt_range["x_min"],
            x_max=gt_range["x_max"],
            y_min=gt_range["y_min"],
            y_max=gt_range["y_max"],
            target_min=args.target_min,
            target_max=args.target_max,
        )

        seq_nonfinite = 0
        seq_zero = 0
        seq_fallback = 0
        for frame_zero in range(args.expected_frames):
            frame_id = frame_zero + 1
            out_index = seq_index * args.expected_frames + frame_zero
            path = csi_path(csi_root, int(seq["subject"]), int(seq["action"]), frame_id)
            raw = read_mat_key(path, args.csi_key)
            repaired, repair_stats = repair_csiamp_frame(
                raw,
                subcarrier_radius=args.repair_subcarrier_radius,
                time_radius=args.repair_time_radius,
            )
            standardized = standardize_csiamp(repaired, expected_csi_shape, path)
            x_all[out_index] = resample_csiamp_time(standardized, args.resample_time_steps)
            y_all[out_index] = y_seq[frame_zero]
            repair_counts[out_index] = min(repair_stats["nonfinite_values"], np.iinfo(np.uint16).max)

            env_arr[out_index] = int(seq["env"])
            subject_arr[out_index] = int(seq["subject"])
            action_arr[out_index] = int(seq["action"])
            frame_arr[out_index] = frame_id
            sequence_id_arr[out_index] = int(seq["sequence_id"])

            seq_nonfinite += repair_stats["nonfinite_values"]
            seq_zero += repair_stats["zero_values"]
            seq_fallback += repair_stats["fallback_repairs"]
            frame_rows.append(
                {
                    "frame_index": out_index,
                    "sequence_id": int(seq["sequence_id"]),
                    "env": int(seq["env"]),
                    "subject": int(seq["subject"]),
                    "action": int(seq["action"]),
                    "frame": frame_id,
                    "csi_nonfinite_repaired": repair_stats["nonfinite_values"],
                    "csi_zero_values_kept": repair_stats["zero_values"],
                    "gt_clipped_keypoint_values": "",
                }
            )

        sequence_rows.append(
            {
                "sequence_id": int(seq["sequence_id"]),
                "env": int(seq["env"]),
                "subject": int(seq["subject"]),
                "action": int(seq["action"]),
                "gt_file": seq["gt_file"],
                "frame_count": args.expected_frames,
                "csi_nonfinite_values_repaired": seq_nonfinite,
                "csi_zero_values_kept": seq_zero,
                "csi_fallback_repairs": seq_fallback,
                **gt_stats,
            }
        )
        totals["frames"] += args.expected_frames
        totals["csi_nonfinite_values_repaired"] += seq_nonfinite
        totals["csi_zero_values_kept"] += seq_zero
        totals["csi_fallback_repairs"] += seq_fallback
        totals["gt_invalid_or_zero_keypoints"] += gt_stats["invalid_or_zero_keypoints"]
        totals["gt_clipped_keypoint_values"] += gt_stats["clipped_keypoint_values"]

        if args.progress_every and (seq_index + 1) % args.progress_every == 0:
            elapsed = max(time.time() - start, 1.0e-9)
            progress(f"cleaned sequences={seq_index + 1}/{len(sequences)} rate={(seq_index + 1) / elapsed:.2f}/s")

    x_all.flush()
    y_all.flush()
    repair_counts.flush()

    np.savez(
        output_root / "meta.npz",
        env=env_arr,
        subject=subject_arr,
        action=action_arr,
        frame=frame_arr,
        sequence_id=sequence_id_arr,
    )
    write_csv(output_root / "sequence_meta.csv", sequence_rows)
    write_csv(output_root / "frame_meta.csv", frame_rows)
    splits = build_splits(sequence_rows)
    (output_root / "splits_by_env.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    write_split_index_npz(output_root / "split_index.npz", splits, args.expected_frames)
    manifest = {
        "generated_at_utc": now_iso(),
        "csi_root": str(csi_root),
        "gt_root": str(gt_root),
        "output_root": str(output_root),
        "csi_key": args.csi_key,
        "ignored_mat_keys": ignored_keys,
        "raw_csi_shape": [int(v) for v in expected_csi_shape],
        "resampled_csi_shape": [int(v) for v in frame_shape],
        "resample_method": "scipy.signal.resample",
        "x_shape": [int(v) for v in x_all.shape],
        "y_shape": [int(v) for v in y_all.shape],
        "storage_layout": {
            "sample_axis": "frame",
            "x_layout": "sample,antenna,subcarrier,time",
            "y_layout": "frame,joint,xy",
            "meta_layout": "frame_aligned_arrays",
            "split_index_layout": "sequence_ids_and_frame_indices_by_env",
        },
        "training_io": {
            "x_file": "X_amp_resampled.npy",
            "y_file": "Y_2d_clean.npy",
            "meta_file": "meta.npz",
            "split_index_file": "split_index.npz",
            "recommended_loading": "np.load(..., mmap_mode='r') for X/Y and np.load for meta/splits",
        },
        "target_dims": "xy_2d",
        "target_range": [args.target_min, args.target_max],
        "gt_source_range": gt_range,
        "repair_filter": {
            "type": "same_antenna_window_mean",
            "subcarrier_radius": args.repair_subcarrier_radius,
            "time_radius": args.repair_time_radius,
            "zeros_are_valid_values": True,
        },
        "env_split_strategy": "leave_one_env_out",
        "totals": totals,
    }
    (output_root / "clean_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    progress(f"wrote cleaned dataset to {output_root.resolve()}")


if __name__ == "__main__":
    main()
