"""
Build memory-mapped .npy files from WiFiPose CSI and Human3.6M-17 GT files.

Pre-computes 3 normalization variants and stores each as a single .npy file.
Training loader uses np.load(path, mmap_mode='r') for zero-copy OS-cached access.

Input:
    /data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/
        wifi-csi/frame*.mat   ← CSIamp (3, 114, 10) float64
        rgb/frame*.npy        ← optional Human3.6M-17 xy keypoints (17, 2) float32

Output:
    /data/WiFiPose/dataset/mmfi_pose_v3/
        csi_gminmax.npy  ← global_minmax normalized (N, 64, 3, 114) float32
        csi_gzscore.npy  ← global_zscore normalized (N, 64, 3, 114) float32
        csi_zscore.npy   ← per-sample zscore normalized (N, 64, 3, 114) float32
        ground_truth.npy ← Human3.6M-17 raw xy coordinates (N, 17, 2) float32
        meta.npz         ← environment, sample, action, frame_idx
        stats.json       ← normalization statistics

Usage:
    python scripts/build_memmap.py \
        --src /data/WiFiPose/dataset/dataset \
        --dst /data/WiFiPose/dataset/mmfi_pose_v3 \
        --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample


TIME_PACKETS = 64
RX_ANTENNAS = 3
SUBCARRIERS = 114

def derive_env(subject: str) -> str:
    num = int(subject.lstrip("S"))
    return f"env{(num - 1) // 10 + 1}"


def sanitize_csi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.median(x[finite])) if finite.any() else 0.0
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def preprocess_csi_one_frame(csi_amp: np.ndarray) -> np.ndarray:
    csi_amp = sanitize_csi(np.asarray(csi_amp, dtype=np.float32))
    csi_amp = sanitize_csi(resample(csi_amp, TIME_PACKETS, axis=-1))
    return np.transpose(csi_amp, (2, 0, 1)).astype(np.float32, copy=False)


def extract_h36m17_xy(gt_data: np.ndarray, *, source: str) -> np.ndarray:
    gt_data = np.asarray(gt_data, dtype=np.float32)
    if gt_data.ndim == 3 and gt_data.shape[1] == 17 and gt_data.shape[2] in (2, 3):
        xy = gt_data[:, :, :2]
    elif gt_data.ndim == 2 and gt_data.shape == (17, 2):
        xy = gt_data[None, :, :]
    else:
        raise ValueError(
            f"{source} must have shape [frames,17,2], [frames,17,3], or [17,2], got {gt_data.shape}"
        )
    if not np.isfinite(xy).all():
        raise ValueError(f"{source} contains non-finite GT coordinates")
    return np.ascontiguousarray(xy, dtype=np.float32)


def iter_trials(src_root: Path, require_rgb: bool = True) -> list[Path]:
    trials: list[Path] = []
    for action_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("A")):
        for subj_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("S")):
            has_wifi = (subj_dir / "wifi-csi").is_dir()
            has_rgb = (subj_dir / "rgb").is_dir()
            if has_wifi and (require_rgb or (not require_rgb)):
                trials.append(subj_dir)
            elif has_wifi and require_rgb and not has_rgb:
                pass
    return trials


def process_trial(
    trial_dir: Path,
    pose_min: float,
    pose_max: float,
    gt_dir: Path | None = None,
) -> dict | None:
    action = trial_dir.parent.name
    subject = trial_dir.name
    environment = derive_env(subject)
    wifi_dir = trial_dir / "wifi-csi"

    mat_paths = sorted(wifi_dir.glob("frame*.mat"))
    if not mat_paths:
        return None

    n_frames = len(mat_paths)
    csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
    frame_idx = np.zeros(n_frames, dtype=np.int64)

    for i, mp in enumerate(mat_paths):
        mat = sio.loadmat(str(mp))
        csi_frames[i] = preprocess_csi_one_frame(np.asarray(mat["CSIamp"], dtype=np.float32))
        frame_idx[i] = int(mp.stem.replace("frame", ""))

    if gt_dir is not None:
        env_num = int(environment.replace("env", ""))
        gt_filename = f"E{env_num:02d}_{subject}_{action}.npy"
        gt_path = gt_dir / gt_filename
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file for {action}/{subject}: {gt_path}")

        gt_xy = extract_h36m17_xy(np.load(str(gt_path)), source=gt_filename)
        gt_n = gt_xy.shape[0]
        if gt_n != n_frames:
            raise ValueError(
                f"CSI/GT frame count mismatch for {action}/{subject}: "
                f"{len(mat_paths)} CSI frames vs {gt_n} GT frames ({gt_filename})"
            )
        keypoints = gt_xy
    else:
        rgb_dir = trial_dir / "rgb"
        npy_paths = sorted(rgb_dir.glob("frame*.npy"))
        npy_stems = {p.stem: p for p in npy_paths}
        mat_stems = {p.stem: p for p in mat_paths}
        common = sorted(set(mat_stems) & set(npy_stems))
        if not common:
            return None
        n_frames = len(common)
        csi_frames = np.empty((n_frames, TIME_PACKETS, RX_ANTENNAS, SUBCARRIERS), dtype=np.float32)
        keypoints = np.zeros((n_frames, 17, 2), dtype=np.float32)
        frame_idx = np.zeros(n_frames, dtype=np.int64)
        for i, stem in enumerate(common):
            mat = sio.loadmat(str(mat_stems[stem]))
            csi_frames[i] = preprocess_csi_one_frame(np.asarray(mat["CSIamp"], dtype=np.float32))
            keypoints[i] = extract_h36m17_xy(np.load(str(npy_stems[stem])), source=npy_stems[stem].name)[0]
            frame_idx[i] = int(stem.replace("frame", ""))

    return {
        "csi": csi_frames,
        "keypoints": keypoints,
        "environment": environment,
        "sample": subject,
        "action": action,
        "frame_idx": frame_idx,
    }


def _worker(args):
    trial_dir, pose_min, pose_max, gt_dir = args
    try:
        result = process_trial(Path(trial_dir), pose_min, pose_max, gt_dir)
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, result, None
    except Exception:
        label = f"{Path(trial_dir).parent.name}/{Path(trial_dir).name}"
        return label, None, traceback.format_exc()


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def main():
    parser = argparse.ArgumentParser(description="Build memmap .npy files from MM-Fi dataset")
    parser.add_argument("--src", default="/data/WiFiPose/dataset/dataset")
    parser.add_argument("--dst", default="/data/WiFiPose/dataset/mmfi_pose_v3")
    parser.add_argument("--train-subjects", nargs="+",
                        default=["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"])
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gt-dir", default=None,
                        help="Directory containing E{env}_S{subject}_A{action}.npy GT files. "
                             "When provided, uses these real GT files instead of rgb/*.npy.")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    train_set = set(args.train_subjects)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    trials = iter_trials(src_root, require_rgb=(gt_dir is None))
    print(f"Found {len(trials)} trials")
    if gt_dir:
        print(f"Using GT dir: {gt_dir}")

    all_data: list[dict] = []
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    if args.workers <= 1:
        for i, trial in enumerate(trials):
            label = f"{trial.parent.name}/{trial.name}"
            result = process_trial(trial, args.pose_min, args.pose_max, gt_dir)
            if result:
                all_data.append(result)
                print(f"  [{i+1}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
            else:
                print(f"  [{i+1}/{len(trials)}] {label} SKIP")
    else:
        tasks = [(str(t), args.pose_min, args.pose_max, gt_dir) for t in trials]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                label, result, err = fut.result()
                if err:
                    failures.append((label, err))
                    print(f"  [{done}/{len(trials)}] {label} FAIL")
                elif result:
                    all_data.append(result)
                    print(f"  [{done}/{len(trials)}] {label} ok N={result['csi'].shape[0]}")
                else:
                    print(f"  [{done}/{len(trials)}] {label} SKIP")

    dt = time.time() - t0
    print(f"\nPhase 1: {len(all_data)} ok, {len(failures)} fail ({dt:.1f}s)")
    if failures:
        print("ERROR: At least one trial failed; aborting before writing memmap files.")
        for label, err in failures:
            print(f"\n--- {label} ---")
            print(err.rstrip())
        sys.exit(1)
    if not all_data:
        print("ERROR: No trials processed")
        sys.exit(1)

    print("Concatenating...")
    all_csi_raw  = np.concatenate([d["csi"] for d in all_data], axis=0).astype(np.float32)
    all_keypoints = np.concatenate([d["keypoints"] for d in all_data], axis=0).astype(np.float32)
    all_envs     = np.array([e for d in all_data for e in [d["environment"]] * d["csi"].shape[0]])
    all_subjects = np.array([s for d in all_data for s in [d["sample"]]      * d["csi"].shape[0]])
    all_actions  = np.array([a for d in all_data for a in [d["action"]]      * d["csi"].shape[0]])
    all_fidx     = np.concatenate([d["frame_idx"] for d in all_data])

    n_total = all_csi_raw.shape[0]
    n_train = int(np.isin(all_subjects.astype(str), list(train_set)).sum())
    print(f"Total: {n_total}, train: {n_train} ({n_train/n_total*100:.1f}%)")

    print("Computing normalization statistics (train set only)...")
    train_mask = np.isin(all_subjects.astype(str), list(train_set))
    train_csi = all_csi_raw[train_mask]
    amp_min  = float(train_csi.min())
    amp_max  = float(train_csi.max())
    amp_mean = float(train_csi.mean())
    amp_std  = float(train_csi.std())
    print(f"  min={amp_min:.4f}  max={amp_max:.4f}  mean={amp_mean:.4f}  std={amp_std:.4f}")

    print("  global_minmax...")
    csi_gminmax = safe_div(all_csi_raw - amp_min, amp_max - amp_min).astype(np.float32)

    print("  global_zscore...")
    csi_gzscore = safe_div(all_csi_raw - amp_mean, amp_std).astype(np.float32)

    print("  zscore...")
    csi_zscore = np.empty_like(all_csi_raw, dtype=np.float32)
    for i in range(n_total):
        x = all_csi_raw[i]
        m, s = float(x.mean()), float(x.std())
        csi_zscore[i] = safe_div(x - m, s).astype(np.float32)

    del all_csi_raw

    print("Saving .npy files...")
    t_save = time.time()

    np.save(str(dst_root / "csi_gminmax.npy"), csi_gminmax)
    np.save(str(dst_root / "csi_gzscore.npy"), csi_gzscore)
    np.save(str(dst_root / "csi_zscore.npy"),  csi_zscore)
    np.save(str(dst_root / "ground_truth.npy"), all_keypoints)
    np.savez(str(dst_root / "meta.npz"),
             environment=all_envs, sample=all_subjects,
             action=all_actions, frame_idx=all_fidx)

    stats = {
        "amplitude_train_min":  amp_min,
        "amplitude_train_max":  amp_max,
        "amplitude_train_mean": amp_mean,
        "amplitude_train_std":  amp_std,
        "gt_coordinate_space": "human36m_raw_xy",
        "gt_normalization": "none",
        "gt_shape": [17, 2],
        "time_packets": TIME_PACKETS,
        "rx_antennas": RX_ANTENNAS,
        "subcarriers": SUBCARRIERS,
        "total_frames": n_total,
        "train_frames": n_train,
    }
    with open(dst_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    total_mb = sum(
        f.stat().st_size for f in dst_root.glob("*.npy") if f.is_file()
    ) / (1024 * 1024)
    print(f"Done in {time.time()-t_save:.0f}s — {total_mb:.0f} MB total .npy files")


if __name__ == "__main__":
    main()
