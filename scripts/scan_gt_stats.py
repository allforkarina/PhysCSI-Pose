from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def scan_gt_stats(
    gt_root: Path,
    pattern: str,
    *,
    num_subjects: int = 40,
    num_actions: int = 27,
    subjects_per_env: int = 10,
) -> dict[str, Any]:
    values: list[np.ndarray] = []
    missing_files: list[str] = []

    for subject_id in range(1, num_subjects + 1):
        env_id = (subject_id - 1) // subjects_per_env + 1
        for action_id in range(1, num_actions + 1):
            path = gt_root / pattern.format(env_id=env_id, subject_id=subject_id, action_id=action_id)
            if not path.exists():
                missing_files.append(str(path))
                continue
            gt = np.load(path)
            if gt.shape != (297, 17, 3):
                raise ValueError(f"{path} has shape {gt.shape}, expected [297,17,3]")
            xy = gt[..., :2]
            values.append(xy[np.isfinite(xy)])

    merged = np.concatenate(values) if values else np.array([0.0], dtype=np.float32)
    return {
        "xy_min": float(merged.min()),
        "xy_max": float(merged.max()),
        "abs_max": float(np.abs(merged).max()),
        "missing_files": missing_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build_memmap.yaml")
    parser.add_argument("--gt-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stats = scan_gt_stats(
        Path(args.gt_root),
        cfg["paths"]["gt_pattern"],
        num_subjects=cfg["dataset"]["num_envs"] * cfg["dataset"]["subjects_per_env"],
        num_actions=cfg["dataset"]["num_actions"],
        subjects_per_env=cfg["dataset"]["subjects_per_env"],
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
