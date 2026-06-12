from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import scipy.io
import torch
import yaml

from dataset.features import build_amplitude_features
from dataset.labels import detect_gt_coord_format, normalize_gt_sequence
from dataset.meta import build_meta_arrays, env_id_from_subject, global_index
from scripts.scan_gt_stats import scan_gt_stats


OUTPUT_FILES = ["X_all.npy", "Y_all.npy", "Conf_all.npy", "meta.npz", "meta_build.json"]
CSI_REPAIR_STAT_KEYS = (
    "repaired_frames",
    "repaired_values",
    "nan_values",
    "inf_values",
    "negative_values",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build_memmap.yaml")
    parser.add_argument("--csi-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def read_mat_key(path: Path, key: str) -> np.ndarray:
    try:
        data = scipy.io.loadmat(path)
        if key not in data:
            raise KeyError(f"{key!r} not found in {path}")
        return np.asarray(data[key])
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"{key!r} not found in {path}")
            return np.asarray(handle[key])


def _accumulate_repair_stats(total: dict[str, int] | None, update: dict[str, int]) -> None:
    if total is None:
        return
    for key in CSI_REPAIR_STAT_KEYS:
        total[key] = int(total.get(key, 0)) + int(update.get(key, 0))


def _repair_csi_frame_values(raw: np.ndarray, path: Path) -> tuple[np.ndarray, dict[str, int]]:
    raw = raw.astype(np.float32, copy=False)
    finite_mask = np.isfinite(raw)
    negative_mask = finite_mask & (raw < 0.0)
    bad_mask = (~finite_mask) | negative_mask
    stats = {
        "repaired_frames": int(bad_mask.any()),
        "repaired_values": int(bad_mask.sum()),
        "nan_values": int(np.isnan(raw).sum()),
        "inf_values": int(np.isinf(raw).sum()),
        "negative_values": int(negative_mask.sum()),
    }
    if not bad_mask.any():
        return raw, stats

    valid_mask = finite_mask & (raw >= 0.0)
    if not valid_mask.any():
        raise ValueError(f"{path} raw CSIamp has no finite non-negative values to repair from")

    values_for_median = np.where(valid_mask, raw, np.nan)
    with np.errstate(all="ignore"):
        local_median = np.nanmedian(values_for_median, axis=2)
        frame_median = float(np.nanmedian(values_for_median))
    local_median = np.where(np.isfinite(local_median), local_median, frame_median).astype(np.float32)

    repaired = raw.copy()
    repair_values = np.broadcast_to(local_median[:, :, None], raw.shape)
    repaired[bad_mask] = repair_values[bad_mask]
    return repaired, stats


def standardize_csi_frame(raw: np.ndarray, path: Path, repair_stats: dict[str, int] | None = None) -> np.ndarray:
    if raw.shape != (3, 114, 10):
        raise ValueError(f"{path} raw CSIamp shape must be [3,114,10], got {raw.shape}")
    repaired, stats = _repair_csi_frame_values(raw, path)
    _accumulate_repair_stats(repair_stats, stats)
    return np.transpose(repaired, (2, 0, 1)).astype(np.float32, copy=False)


def ensure_output_root(root: Path, overwrite: bool) -> None:
    existing = [name for name in OUTPUT_FILES if (root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output files already exist: {existing}; pass --overwrite to rebuild")
    if existing and overwrite:
        for name in existing:
            path = root / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)


def build_memmap(cfg: dict[str, Any], csi_root: Path, gt_root: Path, output_root: Path, device: torch.device) -> None:
    dataset_cfg = cfg["dataset"]
    num_subjects = dataset_cfg["num_envs"] * dataset_cfg["subjects_per_env"]
    num_actions = dataset_cfg["num_actions"]
    num_frames = dataset_cfg["num_frames"]
    total = num_subjects * num_actions * num_frames

    gt_stats = scan_gt_stats(
        gt_root,
        cfg["paths"]["gt_pattern"],
        num_subjects=num_subjects,
        num_actions=num_actions,
        subjects_per_env=dataset_cfg["subjects_per_env"],
    )
    if gt_stats["missing_files"]:
        raise FileNotFoundError(f"missing GT files: {gt_stats['missing_files'][:5]}")
    coord_format = detect_gt_coord_format(
        float(gt_stats["xy_min"]),
        float(gt_stats["xy_max"]),
        float(gt_stats["abs_max"]),
    )

    x_all = np.lib.format.open_memmap(
        output_root / "X_all.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, 12, 10, 114),
    )
    y_all = np.lib.format.open_memmap(
        output_root / "Y_all.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, 17, 2),
    )
    conf_all = np.lib.format.open_memmap(
        output_root / "Conf_all.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, 17),
    )

    meta = build_meta_arrays(num_subjects=num_subjects, num_actions=num_actions, num_frames=num_frames)
    np.savez(output_root / "meta.npz", **meta)

    csi_pattern = cfg["paths"]["csi_pattern"]
    gt_pattern = cfg["paths"]["gt_pattern"]
    csi_key = cfg["mat_keys"]["csi_key"]
    feature_cfg = cfg["feature"]
    sequence_count = 0
    invalid_keypoints = 0
    csi_repair_stats = {key: 0 for key in CSI_REPAIR_STAT_KEYS}

    for subject_id in range(1, num_subjects + 1):
        env_id = env_id_from_subject(subject_id, subjects_per_env=dataset_cfg["subjects_per_env"])
        for action_id in range(1, num_actions + 1):
            frames = []
            for frame_zero_based in range(num_frames):
                frame_id_1based = frame_zero_based + 1
                csi_path = csi_root / csi_pattern.format(
                    action_id=action_id,
                    subject_id=subject_id,
                    frame_id_1based=frame_id_1based,
                )
                raw = read_mat_key(csi_path, csi_key)
                frames.append(standardize_csi_frame(raw, csi_path, repair_stats=csi_repair_stats))
            csi_seq = np.stack(frames, axis=0)

            gt_path = gt_root / gt_pattern.format(env_id=env_id, subject_id=subject_id, action_id=action_id)
            gt_seq = np.load(gt_path)
            y_seq, conf_seq, label_stats = normalize_gt_sequence(gt_seq, coord_format=coord_format)
            invalid_keypoints += int(label_stats["invalid_keypoints"])

            x_seq_t = build_amplitude_features(
                torch.as_tensor(csi_seq, dtype=torch.float32, device=device),
                **feature_cfg,
            )
            x_seq = x_seq_t.detach().cpu().numpy()
            if x_seq.shape != (297, 12, 10, 114):
                raise ValueError(f"feature shape mismatch for S{subject_id:02d} A{action_id:02d}: {x_seq.shape}")

            start = global_index(subject_id, action_id, 0, num_actions=num_actions, num_frames=num_frames)
            end = start + num_frames
            x_all[start:end] = x_seq
            y_all[start:end] = y_seq
            conf_all[start:end] = conf_seq
            sequence_count += 1

    x_all.flush()
    y_all.flush()
    conf_all.flush()

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "total_samples": total,
        "sequence_count": sequence_count,
        "gt_stats": gt_stats,
        "gt_coord_format_detected": coord_format,
        "invalid_keypoints": invalid_keypoints,
        "csi_repair_stats": csi_repair_stats,
        "config": cfg,
    }
    (output_root / "meta_build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = select_device(args.device)
    output_root = Path(args.output_root)
    ensure_output_root(output_root, overwrite=args.overwrite)
    build_memmap(cfg, Path(args.csi_root), Path(args.gt_root), output_root, device)


if __name__ == "__main__":
    main()
