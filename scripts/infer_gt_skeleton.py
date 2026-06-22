from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np


class CandidateSkeleton(NamedTuple):
    name: str
    joint_names: list[str]
    edges: list[tuple[int, int]]


class FrameCandidate(NamedTuple):
    path: Path
    gt_file: str
    frame_index: int
    gt_shape: tuple[int, ...]
    valid_joints: int


class SampledFrame(NamedTuple):
    path: Path
    gt_file: str
    frame_index: int
    gt_shape: tuple[int, ...]
    xy: np.ndarray
    valid_joints: int


HUMAN36M_17 = CandidateSkeleton(
    name="human36m_17",
    joint_names=[
        "pelvis",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "spine",
        "thorax",
        "neck_or_nose",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ],
    edges=[
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
    ],
)

OPENPOSE_COCO_17 = CandidateSkeleton(
    name="openpose_coco_17",
    joint_names=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    edges=[
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ],
)

CANDIDATE_SKELETONS = {
    HUMAN36M_17.name: HUMAN36M_17,
    OPENPOSE_COCO_17.name: OPENPOSE_COCO_17,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(v)) for v in shape)


def valid_xy_mask(xy: np.ndarray) -> np.ndarray:
    return np.isfinite(xy).all(axis=-1) & ~np.all(xy == 0.0, axis=-1)


def discover_frame_candidates(gt_root: Path, min_valid_joints: int) -> list[FrameCandidate]:
    gt_files = sorted(gt_root.glob("*.npy"))
    if not gt_files:
        raise FileNotFoundError(f"no GT .npy files found under {gt_root}")

    candidates: list[FrameCandidate] = []
    for path in gt_files:
        gt = np.load(path, mmap_mode="r", allow_pickle=False)
        if gt.ndim != 3 or gt.shape[-1] < 2:
            continue
        gt_shape = tuple(int(v) for v in gt.shape)
        xy = gt[..., :2]
        frame_valid_counts = valid_xy_mask(xy).sum(axis=1)
        for frame_index, valid_joints in enumerate(frame_valid_counts):
            valid_count = int(valid_joints)
            if valid_count >= min_valid_joints:
                candidates.append(
                    FrameCandidate(
                        path=path,
                        gt_file=path.name,
                        frame_index=int(frame_index),
                        gt_shape=gt_shape,
                        valid_joints=valid_count,
                    )
                )
    if not candidates:
        raise ValueError(f"no GT frames with at least {min_valid_joints} valid joints found under {gt_root}")
    return candidates


def sample_gt_frames(gt_root: Path, num_frames: int, seed: int, min_valid_joints: int = 6) -> list[SampledFrame]:
    candidates = discover_frame_candidates(gt_root, min_valid_joints=min_valid_joints)
    rng = np.random.default_rng(seed)
    sample_size = min(int(num_frames), len(candidates))
    selected_indices = rng.choice(len(candidates), size=sample_size, replace=False)

    samples: list[SampledFrame] = []
    for index in selected_indices:
        candidate = candidates[int(index)]
        gt = np.load(candidate.path, mmap_mode="r", allow_pickle=False)
        xy = np.asarray(gt[candidate.frame_index, :, :2], dtype=np.float64)
        samples.append(
            SampledFrame(
                path=candidate.path,
                gt_file=candidate.gt_file,
                frame_index=candidate.frame_index,
                gt_shape=candidate.gt_shape,
                xy=xy,
                valid_joints=candidate.valid_joints,
            )
        )
    return samples


def skeletons_to_jsonable(candidate_skeletons: dict[str, CandidateSkeleton]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "joint_names": skeleton.joint_names,
            "edges": [[int(a), int(b)] for a, b in skeleton.edges],
        }
        for name, skeleton in candidate_skeletons.items()
    }


def plot_points(ax: object, sample: SampledFrame, skeleton: CandidateSkeleton | None, title: str, image_y_axis: bool) -> None:
    valid = valid_xy_mask(sample.xy)
    points = sample.xy[valid]
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=28, c="#1f77b4")

    for joint_index, (x_value, y_value) in enumerate(sample.xy):
        if not valid[joint_index]:
            continue
        ax.text(float(x_value), float(y_value), str(joint_index), fontsize=8, color="#111111")

    if skeleton is not None:
        for left, right in skeleton.edges:
            if left >= sample.xy.shape[0] or right >= sample.xy.shape[0]:
                continue
            if not (valid[left] and valid[right]):
                continue
            xs = [float(sample.xy[left, 0]), float(sample.xy[right, 0])]
            ys = [float(sample.xy[left, 1]), float(sample.xy[right, 1])]
            ax.plot(xs, ys, linewidth=1.6, color="#d62728")

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if image_y_axis:
        ax.invert_yaxis()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_visualizations(
    samples: list[SampledFrame],
    output_root: Path,
    *,
    candidate_skeletons: dict[str, CandidateSkeleton] | None = None,
    image_y_axis: bool = True,
    dpi: int = 140,
) -> list[dict[str, object]]:
    if candidate_skeletons is None:
        candidate_skeletons = CANDIDATE_SKELETONS

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for GT skeleton visualization; install requirements.txt") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "frames"
    image_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sample_number, sample in enumerate(samples, start=1):
        column_count = 1 + len(candidate_skeletons)
        fig, axes = plt.subplots(1, column_count, figsize=(5.2 * column_count, 5.2), squeeze=False)
        fig.suptitle(f"{sample.gt_file} frame={sample.frame_index} shape={shape_text(sample.gt_shape)}", fontsize=12)

        plot_points(axes[0][0], sample, None, "scatter_only", image_y_axis=image_y_axis)
        for axis, skeleton in zip(axes[0][1:], candidate_skeletons.values()):
            plot_points(axis, sample, skeleton, skeleton.name, image_y_axis=image_y_axis)

        fig.tight_layout()
        image_name = f"sample_{sample_number:02d}_{sample.gt_file.removesuffix('.npy')}_frame{sample.frame_index:03d}.png"
        image_path = image_root / image_name
        fig.savefig(image_path, dpi=dpi)
        plt.close(fig)

        valid = valid_xy_mask(sample.xy)
        valid_xy = sample.xy[valid]
        row = {
            "sample_index": sample_number,
            "gt_file": sample.gt_file,
            "frame_index": sample.frame_index,
            "gt_shape": shape_text(sample.gt_shape),
            "valid_joints": sample.valid_joints,
            "image": image_path.relative_to(output_root).as_posix(),
            "x_min": float(valid_xy[:, 0].min()) if valid_xy.size else "",
            "x_max": float(valid_xy[:, 0].max()) if valid_xy.size else "",
            "y_min": float(valid_xy[:, 1].min()) if valid_xy.size else "",
            "y_max": float(valid_xy[:, 1].max()) if valid_xy.size else "",
        }
        rows.append(row)

    write_csv(output_root / "sampled_frames.csv", rows)
    (output_root / "sampled_frames.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "candidate_skeletons.json").write_text(
        json.dumps(skeletons_to_jsonable(candidate_skeletons), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": now_iso(),
        "sample_count": len(samples),
        "image_y_axis": image_y_axis,
        "candidate_skeletons": list(candidate_skeletons),
        "notes": [
            "Each PNG contains scatter_only plus candidate skeleton overlays for manual GT joint-order inspection.",
            "Joint labels are raw GT indices; red lines are candidate skeleton edges.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample GT frames and visualize candidate 17-joint skeleton orders.")
    parser.add_argument("--gt-root", default="/data/WiFiPose/dataset/ground_truth_npy")
    parser.add_argument("--output-root", default="outputs/gt_skeleton_debug")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--min-valid-joints", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--no-image-y-axis",
        action="store_true",
        help="Do not invert the y axis. By default, plots use image-style y-down coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    output_root = Path(args.output_root)
    samples = sample_gt_frames(
        gt_root,
        num_frames=args.num_frames,
        seed=args.seed,
        min_valid_joints=args.min_valid_joints,
    )
    rows = write_visualizations(
        samples,
        output_root,
        image_y_axis=not args.no_image_y_axis,
        dpi=args.dpi,
    )
    print(f"[done] wrote {len(rows)} GT skeleton debug images to {output_root.resolve()}", flush=True)
    print(f"[done] metadata: {(output_root / 'sampled_frames.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
