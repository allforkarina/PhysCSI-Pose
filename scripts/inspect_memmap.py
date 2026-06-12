from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)

    x = np.load(root / "X_all.npy", mmap_mode="r")
    y = np.load(root / "Y_all.npy", mmap_mode="r")
    conf = np.load(root / "Conf_all.npy", mmap_mode="r")
    meta = np.load(root / "meta.npz")
    manifest = json.loads((root / "meta_build.json").read_text(encoding="utf-8"))

    print(f"X_all: shape={x.shape}, dtype={x.dtype}")
    print(f"Y_all: shape={y.shape}, dtype={y.dtype}")
    print(f"Conf_all: shape={conf.shape}, dtype={conf.dtype}")
    for key in meta.files:
        print(f"meta[{key}]: shape={meta[key].shape}, dtype={meta[key].dtype}")
    print(json.dumps(manifest, indent=2))

    # ── Y_all quality checks ──────────────────────────────────
    print("\n── Y_all quality ──")
    print(f"  min:      {float(y.min()):.6f}")
    print(f"  max:      {float(y.max()):.6f}")
    print(f"  mean:     {float(y.mean()):.6f}")
    print(f"  std:      {float(y.std()):.6f}")

    clip_low = (y == -0.8).any(axis=-1)  # per-keypoint: either x or y clipped
    clip_high = (y == 0.8).any(axis=-1)
    n_total_kp = y.shape[0] * y.shape[1]
    print(f"  any clipped to -0.8: {clip_low.sum():,d} / {n_total_kp} ({100*clip_low.sum()/n_total_kp:.3f}%)")
    print(f"  any clipped to  0.8: {clip_high.sum():,d} / {n_total_kp} ({100*clip_high.sum()/n_total_kp:.3f}%)")

    print("  per-joint clip ratio (-0.8 | 0.8):")
    joint_names = [
        "0:pelvis", "1:r_hip", "2:r_knee", "3:r_ankle", "4:l_hip",
        "5:l_knee", "6:l_ankle", "7:neck", "8:head", "9:r_shoulder",
        "10:r_elbow", "11:r_wrist", "12:l_shoulder", "13:l_elbow",
        "14:l_wrist", "15:spine_mid", "16:spine_low",
    ]
    for j in range(y.shape[1]):
        c_low = float((y[:, j] == -0.8).any(axis=-1).mean())
        c_high = float((y[:, j] == 0.8).any(axis=-1).mean())
        name = joint_names[j] if j < len(joint_names) else f"{j}:?"
        print(f"    {name:18s} low={c_low:.4f}  high={c_high:.4f}")

    print("\n── Conf_all quality ──")
    print(f"  min:      {float(conf.min()):.6f}")
    print(f"  max:      {float(conf.max()):.6f}")
    print(f"  mean:     {float(conf.mean()):.6f}")
    zero_conf = (conf == 0.0)
    n_total_conf = conf.shape[0] * conf.shape[1]
    print(f"  conf == 0: {zero_conf.sum():,d} / {n_total_conf} ({100*zero_conf.sum()/n_total_conf:.3f}%)")
    print("\n── X_all quality ──")
    print(f"  any NaN in X: {bool(np.any(np.isnan(x)))}")
    print(f"  any Inf in X: {bool(np.any(np.isinf(x)))}")


if __name__ == "__main__":
    main()
