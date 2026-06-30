from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_memmap_dataset(data_dir: Path, frames: int = 20) -> None:
    data_dir.mkdir()
    csi = np.zeros((frames, 64, 3, 114), dtype=np.float32)
    keypoints = np.zeros((frames, 17, 2), dtype=np.float32)
    for name in ("csi_gminmax.npy", "csi_gzscore.npy", "csi_zscore.npy"):
        np.save(data_dir / name, csi)
    np.save(data_dir / "ground_truth.npy", keypoints)
    np.savez(
        data_dir / "meta.npz",
        environment=np.array(["env1"] * frames),
        sample=np.array(["S01"] * frames),
        action=np.array(["A01"] * frames),
        frame_idx=np.arange(1, frames + 1),
    )


def _write_subject_env_memmap_dataset(
    data_dir: Path,
    *,
    envs: tuple[str, ...] = ("env1",),
    subjects: tuple[str, ...] = ("S01", "S02", "S03", "S04", "S05"),
    frames_per_subject: int = 3,
) -> None:
    data_dir.mkdir()
    environments: list[str] = []
    samples: list[str] = []
    actions: list[str] = []
    frame_idx: list[int] = []
    for env in envs:
        for subject in subjects:
            for frame in range(1, frames_per_subject + 1):
                environments.append(env)
                samples.append(subject)
                actions.append("A01")
                frame_idx.append(frame)

    frames = len(samples)
    csi = np.zeros((frames, 64, 3, 114), dtype=np.float32)
    keypoints = np.zeros((frames, 17, 2), dtype=np.float32)
    for name in ("csi_gminmax.npy", "csi_gzscore.npy", "csi_zscore.npy"):
        np.save(data_dir / name, csi)
    np.save(data_dir / "ground_truth.npy", keypoints)
    np.savez(
        data_dir / "meta.npz",
        environment=np.array(environments),
        sample=np.array(samples),
        action=np.array(actions),
        frame_idx=np.array(frame_idx),
    )


def test_fixed_frame_split_is_disjoint_complete_and_70_10_20_per_subject(
    tmp_path: Path,
) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    subjects = ("S01", "S02", "S03")
    _write_subject_env_memmap_dataset(
        data_dir,
        envs=("env1",),
        subjects=subjects,
        frames_per_subject=10,
    )

    train = MemmapDataset(data_dir, split="train", seed=123)
    val = MemmapDataset(data_dir, split="val", seed=123)
    test = MemmapDataset(data_dir, split="test", seed=123)
    all_data = MemmapDataset(data_dir, split="all", seed=123)

    train_indices = set(train.indices.tolist())
    val_indices = set(val.indices.tolist())
    test_indices = set(test.indices.tolist())

    assert train_indices
    assert val_indices
    assert test_indices
    assert val_indices != test_indices
    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)
    assert train_indices | val_indices | test_indices == set(all_data.indices.tolist())
    assert len(train) == 21
    assert len(val) == 3
    assert len(test) == 6

    for dataset in (train, val, test):
        assert {str(dataset._samples[idx]) for idx in dataset.indices} == set(subjects)


def test_environment_filter_happens_before_frame_split(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_subject_env_memmap_dataset(
        data_dir,
        envs=("env1", "env2"),
        frames_per_subject=10,
    )

    train = MemmapDataset(data_dir, split="train", envs=("env1",), seed=123)
    val = MemmapDataset(data_dir, split="val", envs=("env1",), seed=123)
    test = MemmapDataset(data_dir, split="test", envs=("env1",), seed=123)

    def envs_for(dataset: MemmapDataset) -> set[str]:
        return {str(dataset._envs[idx]) for idx in dataset.indices}

    assert envs_for(train) == {"env1"}
    assert envs_for(val) == {"env1"}
    assert envs_for(test) == {"env1"}


def test_fixed_frame_split_is_seeded(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_subject_env_memmap_dataset(
        data_dir,
        envs=("env1",),
        subjects=("S01", "S02"),
        frames_per_subject=10,
    )
    first = MemmapDataset(data_dir, split="train", envs=("env1",), seed=7)
    repeated = MemmapDataset(data_dir, split="train", envs=("env1",), seed=7)
    changed = MemmapDataset(data_dir, split="train", envs=("env1",), seed=8)

    assert np.array_equal(first.indices, repeated.indices)
    assert not np.array_equal(first.indices, changed.indices)


def test_dataset_and_loader_expose_only_fixed_frame_split_api() -> None:
    from data.memmap_dataset import MemmapDataset
    from dataloader import create_memmap_data_loader

    dataset_params = inspect.signature(MemmapDataset).parameters
    loader_params = inspect.signature(create_memmap_data_loader).parameters
    for removed in ("split_strategy", "train_subjects", "val_subjects", "test_subjects"):
        assert removed not in dataset_params
        assert removed not in loader_params


def test_create_few_shot_data_loader_returns_only_train_loader_and_indices(tmp_path: Path) -> None:
    from dataloader import create_few_shot_data_loader

    data_dir = tmp_path / "memmap"
    _write_memmap_dataset(data_dir, frames=20)

    result = create_few_shot_data_loader(
        data_dir=data_dir,
        target_envs=("env1",),
        few_shot_subjects=1,
        few_shot_frames=3,
        batch_size=2,
        num_workers=0,
        seed=123,
    )

    assert len(result) == 2
    train_loader, train_indices = result
    assert len(train_indices) == 3
    assert len(train_loader.dataset) == 3
