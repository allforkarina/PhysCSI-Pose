from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer_gt_skeleton.py"
SPEC = importlib.util.spec_from_file_location("infer_gt_skeleton", MODULE_PATH)
infer_gt_skeleton = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(infer_gt_skeleton)


def write_gt(path: Path, frames: int, joints: int = 17) -> None:
    data = np.zeros((frames, joints, 3), dtype=np.float32)
    for frame in range(frames):
        for joint in range(joints):
            data[frame, joint, 0] = float(joint + 1)
            data[frame, joint, 1] = float(frame + 1 + joint / 10.0)
            data[frame, joint, 2] = 1.0
    np.save(path, data)


def sample_keys(samples: list[object]) -> list[tuple[str, int]]:
    return [(sample.gt_file, sample.frame_index) for sample in samples]


def test_sample_gt_frames_is_reproducible_and_unique(tmp_path: Path) -> None:
    write_gt(tmp_path / "E01_S01_A01.npy", frames=4)
    write_gt(tmp_path / "E01_S01_A02.npy", frames=4)

    first = infer_gt_skeleton.sample_gt_frames(tmp_path, num_frames=5, seed=123, min_valid_joints=3)
    second = infer_gt_skeleton.sample_gt_frames(tmp_path, num_frames=5, seed=123, min_valid_joints=3)

    assert len(first) == 5
    assert sample_keys(first) == sample_keys(second)
    assert len(set(sample_keys(first))) == 5
    assert all(sample.xy.shape == (17, 2) for sample in first)
    assert all(sample.valid_joints == 17 for sample in first)


def test_write_visualizations_outputs_images_and_metadata(tmp_path: Path) -> None:
    write_gt(tmp_path / "E01_S01_A01.npy", frames=1)
    samples = infer_gt_skeleton.sample_gt_frames(tmp_path, num_frames=1, seed=123, min_valid_joints=3)
    output_root = tmp_path / "outputs"

    rows = infer_gt_skeleton.write_visualizations(
        samples,
        output_root,
        candidate_skeletons={
            "tiny_candidate": infer_gt_skeleton.CandidateSkeleton(
                name="tiny_candidate",
                joint_names=["j0", "j1", "j2"],
                edges=[(0, 1), (1, 2)],
            )
        },
        image_y_axis=True,
        dpi=50,
    )

    assert len(rows) == 1
    image_path = output_root / rows[0]["image"]
    assert image_path.exists()
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with (output_root / "sampled_frames.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["gt_file"] == "E01_S01_A01.npy"
    assert csv_rows[0]["frame_index"] == "0"

    metadata = json.loads((output_root / "sampled_frames.json").read_text(encoding="utf-8"))
    assert metadata[0]["valid_joints"] == 17

    skeletons = json.loads((output_root / "candidate_skeletons.json").read_text(encoding="utf-8"))
    assert skeletons["tiny_candidate"]["edges"] == [[0, 1], [1, 2]]
