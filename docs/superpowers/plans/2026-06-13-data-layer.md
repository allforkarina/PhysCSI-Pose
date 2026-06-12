# Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data feature construction, label normalization, full-dataset memmap cache, and source-only Dataset infrastructure for PhysCSI-Pose.

**Architecture:** Keep the data layer split into small modules: feature math in `dataset/features.py`, label cleanup in `dataset/labels.py`, neutral metadata mapping in `dataset/meta.py`, split rules in `dataset/splits.py`, and mmap reading in `dataset/memmap_dataset.py`. Build scripts orchestrate file IO and cache writing; core modules stay testable with synthetic tensors and arrays.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, h5py, PyYAML, pytest.

---

## File Structure

Create these files during implementation:

```text
scripts/scan_gt_stats.py
scripts/build_memmap.py
scripts/inspect_memmap.py
dataset/__init__.py
dataset/features.py
dataset/labels.py
dataset/meta.py
dataset/splits.py
dataset/memmap_dataset.py
tests/test_features.py
tests/test_labels.py
tests/test_meta.py
tests/test_dataset.py
configs/build_memmap.yaml
requirements.txt
README.md
```

Do not create an extra nested `WiFiPose/` directory. Do not add generated data files to Git.

## Task 1: Dependencies, Config, And README

**Files:**
- Create: `requirements.txt`
- Create: `configs/build_memmap.yaml`
- Create: `README.md`

- [ ] **Step 1: Create dependency file**

Create `requirements.txt`:

```text
numpy
scipy
h5py
pyyaml
pytest
torch
```

- [ ] **Step 2: Create build config**

Create `configs/build_memmap.yaml`:

```yaml
dataset:
  num_envs: 4
  subjects_per_env: 10
  num_actions: 27
  num_frames: 297
  num_packets: 10
  num_rx: 3
  num_subcarriers: 114
  num_joints: 17

feature:
  eps_log: 1.0e-6
  eps_mad: 1.0e-6
  subcarrier_smooth_kernel: 15
  subcarrier_padding_mode: reflect

gt:
  image_width: 1920
  image_height: 1080
  target_min: -0.8
  target_max: 0.8

mat_keys:
  csi_key: CSIamp

paths:
  csi_pattern: "A{action_id:02d}/S{subject_id:02d}/frame_{frame_id_1based:03d}.mat"
  gt_pattern: "E{env_id:02d}_S{subject_id:02d}_A{action_id:02d}.npy"
```

- [ ] **Step 3: Create README**

Create `README.md`:

````markdown
# PhysCSI-Pose

Data-layer implementation for WiFi CSI based human pose recognition.

The first version builds amplitude-only CSI features once and stores them as mmap-readable `.npy` arrays. Model networks, training loops, inference, and evaluation are intentionally not part of this phase.

The cache always stores all four feature groups as 12 channels. Ablation studies select feature groups at Dataset read time:

```python
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm", "f_sub"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train")
```

The default uses all feature groups: `["l_norm", "d_center", "f_sub", "c_ant"]`.

## Data Build

Generated files must live outside Git, for example:

```text
/data/WiFiPose/dataset/memmap/
  X_all.npy
  Y_all.npy
  Conf_all.npy
  meta.npz
  meta_build.json
```

Build command:

```bash
python scripts/build_memmap.py \
  --config configs/build_memmap.yaml \
  --csi-root /path/to/csi_root \
  --gt-root /path/to/gt_root \
  --output-root /data/WiFiPose/dataset/memmap \
  --device auto
```

If the server already has CUDA PyTorch installed, do not overwrite it with a CPU-only wheel.
````

- [ ] **Step 4: Verify files exist**

Run:

```bash
rg --files
```

Expected: the three files above appear, and no generated data files appear.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/build_memmap.yaml README.md
git commit -m "docs: add data build configuration"
git push
```

## Task 2: Metadata And Source-Only Split Rules

**Files:**
- Create: `dataset/__init__.py`
- Create: `dataset/meta.py`
- Create: `dataset/splits.py`
- Create: `tests/test_meta.py`

- [ ] **Step 1: Write failing metadata tests**

Create `tests/test_meta.py`:

```python
import numpy as np
import pytest

from dataset.meta import (
    env_id_from_subject,
    global_index,
    sequence_id,
    build_meta_arrays,
)
from dataset.splits import source_only_subjects


def test_env_sequence_and_global_index_reference_points():
    assert env_id_from_subject(1) == 1
    assert env_id_from_subject(10) == 1
    assert env_id_from_subject(11) == 2
    assert env_id_from_subject(40) == 4

    assert sequence_id(1, 1) == 0
    assert sequence_id(1, 2) == 1
    assert sequence_id(40, 27) == 1079

    assert global_index(1, 1, 0) == 0
    assert global_index(1, 1, 296) == 296
    assert global_index(1, 2, 0) == 297
    assert global_index(40, 27, 296) == 320759


def test_build_meta_arrays_shapes_and_dtypes():
    meta = build_meta_arrays(num_subjects=40, num_actions=27, num_frames=297)
    assert meta["global_idx"].shape == (320760,)
    assert meta["env_id"].dtype == np.uint8
    assert meta["subject_id"].dtype == np.uint8
    assert meta["action_id"].dtype == np.uint8
    assert meta["frame_id"].dtype == np.uint16
    assert meta["seq_id"].dtype == np.uint16
    assert meta["global_idx"][0] == 0
    assert meta["env_id"][0] == 1
    assert meta["subject_id"][320759] == 40
    assert meta["action_id"][320759] == 27
    assert meta["frame_id"][320759] == 296
    assert meta["seq_id"][320759] == 1079


def test_source_only_split_subjects():
    assert source_only_subjects(env_id=1, split="train") == [1, 2, 3, 4, 5, 6, 7]
    assert source_only_subjects(env_id=1, split="val") == [8, 9]
    assert source_only_subjects(env_id=1, split="test") == [10]
    assert source_only_subjects(env_id=2, split="train") == [11, 12, 13, 14, 15, 16, 17]
    assert source_only_subjects(env_id=4, split="test") == [40]


def test_invalid_split_inputs_raise():
    with pytest.raises(ValueError, match="env_id"):
        source_only_subjects(env_id=0, split="train")
    with pytest.raises(ValueError, match="split"):
        source_only_subjects(env_id=1, split="dev")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_meta.py -v
```

Expected: FAIL because `dataset.meta` and `dataset.splits` do not exist.

- [ ] **Step 3: Implement metadata helpers**

Create `dataset/__init__.py`:

```python
"""Dataset utilities for PhysCSI-Pose."""
```

Create `dataset/meta.py`:

```python
from __future__ import annotations

import numpy as np


def env_id_from_subject(subject_id: int, subjects_per_env: int = 10) -> int:
    if subject_id < 1:
        raise ValueError(f"subject_id must be positive, got {subject_id}")
    return (subject_id - 1) // subjects_per_env + 1


def sequence_id(subject_id: int, action_id: int, num_actions: int = 27) -> int:
    if subject_id < 1:
        raise ValueError(f"subject_id must be positive, got {subject_id}")
    if not 1 <= action_id <= num_actions:
        raise ValueError(f"action_id must be in 1..{num_actions}, got {action_id}")
    return (subject_id - 1) * num_actions + (action_id - 1)


def global_index(subject_id: int, action_id: int, frame_id: int, num_actions: int = 27, num_frames: int = 297) -> int:
    if not 0 <= frame_id < num_frames:
        raise ValueError(f"frame_id must be in 0..{num_frames - 1}, got {frame_id}")
    return sequence_id(subject_id, action_id, num_actions=num_actions) * num_frames + frame_id


def build_meta_arrays(num_subjects: int = 40, num_actions: int = 27, num_frames: int = 297, subjects_per_env: int = 10) -> dict[str, np.ndarray]:
    total = num_subjects * num_actions * num_frames
    global_idx = np.empty(total, dtype=np.int64)
    env_id = np.empty(total, dtype=np.uint8)
    subject_id_arr = np.empty(total, dtype=np.uint8)
    action_id_arr = np.empty(total, dtype=np.uint8)
    frame_id_arr = np.empty(total, dtype=np.uint16)
    seq_id_arr = np.empty(total, dtype=np.uint16)

    cursor = 0
    for subject_id in range(1, num_subjects + 1):
        env = env_id_from_subject(subject_id, subjects_per_env=subjects_per_env)
        for action_id in range(1, num_actions + 1):
            seq = sequence_id(subject_id, action_id, num_actions=num_actions)
            for frame_id in range(num_frames):
                idx = seq * num_frames + frame_id
                global_idx[cursor] = idx
                env_id[cursor] = env
                subject_id_arr[cursor] = subject_id
                action_id_arr[cursor] = action_id
                frame_id_arr[cursor] = frame_id
                seq_id_arr[cursor] = seq
                cursor += 1

    return {
        "global_idx": global_idx,
        "env_id": env_id,
        "subject_id": subject_id_arr,
        "action_id": action_id_arr,
        "frame_id": frame_id_arr,
        "seq_id": seq_id_arr,
    }
```

Create `dataset/splits.py`:

```python
from __future__ import annotations


VALID_SPLITS = {"train", "val", "test"}


def source_only_subjects(env_id: int, split: str, subjects_per_env: int = 10) -> list[int]:
    if not 1 <= env_id <= 4:
        raise ValueError(f"env_id must be in 1..4, got {env_id}")
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    start = (env_id - 1) * subjects_per_env + 1
    subjects = list(range(start, start + subjects_per_env))
    if split == "train":
        return subjects[:7]
    if split == "val":
        return subjects[7:9]
    return subjects[9:]
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_meta.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset/__init__.py dataset/meta.py dataset/splits.py tests/test_meta.py
git commit -m "feat: add metadata and source-only splits"
git push
```

## Task 3: Amplitude Feature Construction

**Files:**
- Create: `dataset/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing feature tests**

Create `tests/test_features.py`:

```python
import pytest
import torch

from dataset.features import FEATURE_CHANNELS, build_amplitude_features, selected_feature_channels


def make_csi_sequence():
    base = torch.arange(297 * 10 * 3 * 114, dtype=torch.float32).reshape(297, 10, 3, 114)
    return base.remainder(37).add(1.0)


def test_feature_output_shape_and_device():
    csi = make_csi_sequence()
    x = build_amplitude_features(csi)
    assert x.shape == (297, 12, 10, 114)
    assert x.dtype == torch.float32
    assert x.device == csi.device


def test_d_center_has_zero_short_time_mean():
    x = build_amplitude_features(make_csi_sequence())
    d_center = x[:, 3:6].permute(0, 2, 1, 3)
    assert torch.allclose(d_center.mean(dim=1), torch.zeros_like(d_center.mean(dim=1)), atol=1e-5)


def test_c_ant_has_zero_rx_mean():
    x = build_amplitude_features(make_csi_sequence())
    c_ant = x[:, 9:12].permute(0, 2, 1, 3)
    assert torch.allclose(c_ant.mean(dim=2), torch.zeros_like(c_ant.mean(dim=2)), atol=1e-5)


def test_channel_order_uses_feature_blocks_then_rx():
    csi = make_csi_sequence()
    outputs = build_amplitude_features(csi, return_components=True)
    x = outputs.x
    assert torch.allclose(x[:, 0], outputs.l_norm[:, :, 0, :])
    assert torch.allclose(x[:, 1], outputs.l_norm[:, :, 1, :])
    assert torch.allclose(x[:, 2], outputs.l_norm[:, :, 2, :])
    assert torch.allclose(x[:, 3], outputs.d_center[:, :, 0, :])
    assert torch.allclose(x[:, 6], outputs.f_sub[:, :, 0, :])
    assert torch.allclose(x[:, 9], outputs.c_ant[:, :, 0, :])


def test_feature_channel_mapping_for_ablation():
    assert FEATURE_CHANNELS == {
        "l_norm": (0, 1, 2),
        "d_center": (3, 4, 5),
        "f_sub": (6, 7, 8),
        "c_ant": (9, 10, 11),
    }
    assert selected_feature_channels(["l_norm"]) == [0, 1, 2]
    assert selected_feature_channels(["f_sub", "c_ant"]) == [6, 7, 8, 9, 10, 11]


def test_invalid_feature_selection_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        selected_feature_channels(["raw_amp"])


def test_invalid_shape_raises():
    bad = torch.ones(297, 3, 10, 114)
    try:
        build_amplitude_features(bad)
    except ValueError as exc:
        assert "[297, 10, 3, 114]" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_features.py -v
```

Expected: FAIL because `dataset.features` does not exist.

- [ ] **Step 3: Implement feature construction**

Create `dataset/features.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


FEATURE_CHANNELS: dict[str, tuple[int, int, int]] = {
    "l_norm": (0, 1, 2),
    "d_center": (3, 4, 5),
    "f_sub": (6, 7, 8),
    "c_ant": (9, 10, 11),
}
DEFAULT_FEATURES: tuple[str, ...] = ("l_norm", "d_center", "f_sub", "c_ant")


@dataclass(frozen=True)
class FeatureComponents:
    x: torch.Tensor
    l_norm: torch.Tensor
    d_center: torch.Tensor
    f_sub: torch.Tensor
    c_ant: torch.Tensor


def selected_feature_channels(features: list[str] | tuple[str, ...] | None = None) -> list[int]:
    names = DEFAULT_FEATURES if features is None else tuple(features)
    if not names:
        raise ValueError("features must contain at least one feature name")

    channels: list[int] = []
    for name in names:
        if name not in FEATURE_CHANNELS:
            raise ValueError(f"unknown feature {name!r}; expected one of {list(FEATURE_CHANNELS)}")
        channels.extend(FEATURE_CHANNELS[name])
    return channels


def _validate_csi_sequence(csiamp: torch.Tensor) -> None:
    if tuple(csiamp.shape) != (297, 10, 3, 114):
        raise ValueError(f"CSIamp sequence must have shape [297, 10, 3, 114], got {tuple(csiamp.shape)}")
    if not torch.isfinite(csiamp).all():
        raise ValueError("CSIamp sequence must contain only finite values")
    if torch.any(csiamp < 0):
        raise ValueError("CSIamp sequence must be non-negative")


def _subcarrier_smooth(l_norm: torch.Tensor, kernel_size: int, padding_mode: str) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if padding_mode != "reflect":
        raise ValueError(f"only reflect padding is supported, got {padding_mode!r}")

    f_count, t_count, rx_count, sub_count = l_norm.shape
    flat = l_norm.reshape(f_count * t_count * rx_count, 1, sub_count)
    padded = F.pad(flat, (kernel_size // 2, kernel_size // 2), mode=padding_mode)
    smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return smoothed.reshape(f_count, t_count, rx_count, sub_count)


def build_amplitude_features(
    csiamp: torch.Tensor,
    *,
    eps_log: float = 1.0e-6,
    eps_mad: float = 1.0e-6,
    subcarrier_smooth_kernel: int = 15,
    subcarrier_padding_mode: str = "reflect",
    return_components: bool = False,
) -> torch.Tensor | FeatureComponents:
    csiamp = csiamp.to(dtype=torch.float32)
    _validate_csi_sequence(csiamp)

    l_value = torch.log(csiamp + eps_log)
    bg = torch.median(l_value.reshape(-1, 3, 114), dim=0).values
    centered = l_value - bg.view(1, 1, 3, 114)
    mad = torch.median(torch.abs(centered).reshape(-1, 3, 114), dim=0).values
    l_norm = centered / (mad.view(1, 1, 3, 114) + eps_mad)

    d_center = l_norm - l_norm.mean(dim=1, keepdim=True)
    smooth_sub = _subcarrier_smooth(l_norm, kernel_size=subcarrier_smooth_kernel, padding_mode=subcarrier_padding_mode)
    f_sub = l_norm - smooth_sub
    c_ant = l_norm - l_norm.mean(dim=2, keepdim=True)

    blocks = [l_norm, d_center, f_sub, c_ant]
    x = torch.cat([block.permute(0, 2, 1, 3) for block in blocks], dim=1).contiguous()

    if return_components:
        return FeatureComponents(x=x, l_norm=l_norm, d_center=d_center, f_sub=f_sub, c_ant=c_ant)
    return x
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_features.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset/features.py tests/test_features.py
git commit -m "feat: add CSI amplitude feature construction"
git push
```

## Task 4: Label Cleaning And Coordinate Normalization

**Files:**
- Create: `dataset/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write failing label tests**

Create `tests/test_labels.py`:

```python
import numpy as np

from dataset.labels import detect_gt_coord_format, normalize_gt_sequence


def test_pixel_coordinates_map_to_target_range():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = 1920.0
    gt[..., 1] = 1080.0
    gt[..., 2] = 1.0
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "pixel_1920x1080"
    assert np.allclose(y[..., 0], 0.8)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 1.0)


def test_unit_coordinates_map_to_target_range():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = 0.5
    gt[..., 1] = 1.0
    gt[..., 2] = 0.7
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "unit_norm_0_1"
    assert np.allclose(y[..., 0], 0.0)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 0.7)


def test_target_norm_clamps_only():
    gt = np.zeros((297, 17, 3), dtype=np.float32)
    gt[..., 0] = -0.9
    gt[..., 1] = 0.9
    gt[..., 2] = 2.0
    y, conf, stats = normalize_gt_sequence(gt)
    assert stats["coord_format"] == "target_norm_-0.8_0.8"
    assert np.allclose(y[..., 0], -0.8)
    assert np.allclose(y[..., 1], 0.8)
    assert np.allclose(conf, 1.0)


def test_invalid_xy_and_conf_are_zeroed():
    gt = np.ones((297, 17, 3), dtype=np.float32)
    gt[0, 0, 0] = np.nan
    gt[0, 0, 1] = 5.0
    gt[0, 1, :2] = 0.0
    gt[0, 2, 2] = np.inf
    y, conf, stats = normalize_gt_sequence(gt)
    assert np.allclose(y[0, 0], [0.0, 0.0])
    assert conf[0, 0] == 0.0
    assert np.allclose(y[0, 1], [0.0, 0.0])
    assert conf[0, 1] == 0.0
    assert conf[0, 2] == 0.0
    assert stats["invalid_keypoints"] >= 3


def test_detect_gt_coord_format():
    assert detect_gt_coord_format(-1.0, 100.0, 100.0) == "pixel_1920x1080"
    assert detect_gt_coord_format(0.0, 1.0, 1.0) == "unit_norm_0_1"
    assert detect_gt_coord_format(-0.7, 0.7, 0.7) == "target_norm_-0.8_0.8"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_labels.py -v
```

Expected: FAIL because `dataset.labels` does not exist.

- [ ] **Step 3: Implement label helpers**

Create `dataset/labels.py`:

```python
from __future__ import annotations

import numpy as np


def detect_gt_coord_format(xy_min: float, xy_max: float, abs_max: float) -> str:
    if abs_max > 10.0:
        return "pixel_1920x1080"
    if xy_min >= 0.0 and xy_max <= 1.0:
        return "unit_norm_0_1"
    return "target_norm_-0.8_0.8"


def normalize_gt_sequence(
    gt: np.ndarray,
    *,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    target_min: float = -0.8,
    target_max: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if gt.shape != (297, 17, 3):
        raise ValueError(f"GT sequence must have shape [297, 17, 3], got {gt.shape}")

    gt = gt.astype(np.float32, copy=True)
    xy = gt[..., :2]
    conf = gt[..., 2]

    invalid_xy = ~np.isfinite(xy).all(axis=-1)
    zero_xy = np.all(xy == 0.0, axis=-1)
    invalid_conf = ~np.isfinite(conf)
    invalid = invalid_xy | zero_xy | invalid_conf

    xy[invalid_xy | zero_xy] = 0.0
    conf[invalid] = 0.0

    finite_xy = xy[np.isfinite(xy)]
    xy_min = float(finite_xy.min()) if finite_xy.size else 0.0
    xy_max = float(finite_xy.max()) if finite_xy.size else 0.0
    abs_max = float(np.abs(finite_xy).max()) if finite_xy.size else 0.0
    coord_format = detect_gt_coord_format(xy_min, xy_max, abs_max)

    if coord_format == "pixel_1920x1080":
        xy[..., 0] = xy[..., 0] / image_width
        xy[..., 1] = xy[..., 1] / image_height

    if coord_format in {"pixel_1920x1080", "unit_norm_0_1"}:
        span = target_max - target_min
        xy = xy * span + target_min

    xy = np.clip(xy, target_min, target_max).astype(np.float32, copy=False)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32, copy=False)
    xy[invalid] = 0.0

    stats = {
        "coord_format": coord_format,
        "xy_min_before": xy_min,
        "xy_max_before": xy_max,
        "abs_max_before": abs_max,
        "invalid_keypoints": int(invalid.sum()),
    }
    return xy, conf, stats
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_labels.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset/labels.py tests/test_labels.py
git commit -m "feat: add pose label normalization"
git push
```

## Task 5: Memmap Dataset Reader

**Files:**
- Create: `dataset/memmap_dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing Dataset tests**

Create `tests/test_dataset.py`:

```python
import numpy as np
import pytest

from dataset.memmap_dataset import MemmapPoseDataset
from dataset.meta import build_meta_arrays


def write_fake_cache(root):
    n = 40 * 27 * 297
    x = np.zeros((n, 12, 10, 114), dtype=np.float32)
    for channel in range(12):
        x[:, channel, :, :] = float(channel)
    np.save(root / "X_all.npy", x)
    np.save(root / "Y_all.npy", np.zeros((n, 17, 2), dtype=np.float32))
    np.save(root / "Conf_all.npy", np.ones((n, 17), dtype=np.float32))
    np.savez(root / "meta.npz", **build_meta_arrays())


def test_source_only_env01_train_filters_subjects(tmp_path):
    write_fake_cache(tmp_path)
    ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train")
    assert len(ds) == 7 * 27 * 297
    item = ds[0]
    assert item["x"].shape == (12, 10, 114)
    assert item["y"].shape == (17, 2)
    assert item["conf"].shape == (17,)
    assert item["meta"]["subject_id"] in {1, 2, 3, 4, 5, 6, 7}


def test_feature_selection_returns_requested_channel_groups(tmp_path):
    write_fake_cache(tmp_path)
    l_norm_ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["l_norm"])
    l_norm_item = l_norm_ds[0]
    assert l_norm_item["x"].shape == (3, 10, 114)
    assert np.all(l_norm_item["x"][:, 0, 0] == np.array([0.0, 1.0, 2.0], dtype=np.float32))

    combo_ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["f_sub", "c_ant"])
    combo_item = combo_ds[0]
    assert combo_item["x"].shape == (6, 10, 114)
    assert np.all(combo_item["x"][:, 0, 0] == np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32))


def test_invalid_feature_selection_raises(tmp_path):
    write_fake_cache(tmp_path)
    with pytest.raises(ValueError, match="unknown feature"):
        MemmapPoseDataset(tmp_path, protocol="source_only", env_id=1, split="train", features=["raw_amp"])


def test_source_only_env02_test_filters_subjects(tmp_path):
    write_fake_cache(tmp_path)
    ds = MemmapPoseDataset(tmp_path, protocol="source_only", env_id=2, split="test")
    assert len(ds) == 1 * 27 * 297
    subjects = {ds[i]["meta"]["subject_id"] for i in [0, len(ds) - 1]}
    assert subjects == {20}


def test_finetune_protocol_is_not_implemented(tmp_path):
    write_fake_cache(tmp_path)
    with pytest.raises(NotImplementedError, match="finetune"):
        MemmapPoseDataset(tmp_path, protocol="finetune", env_id=2, split="train")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_dataset.py -v
```

Expected: FAIL because `dataset.memmap_dataset` does not exist.

- [ ] **Step 3: Implement Dataset**

Create `dataset/memmap_dataset.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dataset.features import selected_feature_channels
from dataset.splits import source_only_subjects


class MemmapPoseDataset:
    def __init__(
        self,
        root: str | Path,
        *,
        protocol: str,
        env_id: int,
        split: str,
        features: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if protocol == "finetune":
            raise NotImplementedError("finetune protocol is not implemented yet")
        if protocol != "source_only":
            raise ValueError(f"protocol must be 'source_only', got {protocol!r}")

        self.root = Path(root)
        self.feature_channels = selected_feature_channels(features)
        self.x_all = np.load(self.root / "X_all.npy", mmap_mode="r")
        self.y_all = np.load(self.root / "Y_all.npy", mmap_mode="r")
        self.conf_all = np.load(self.root / "Conf_all.npy", mmap_mode="r")
        self.meta = np.load(self.root / "meta.npz")

        subjects = np.array(source_only_subjects(env_id=env_id, split=split), dtype=np.uint8)
        mask = (self.meta["env_id"] == env_id) & np.isin(self.meta["subject_id"], subjects)
        self.indices = self.meta["global_idx"][mask].astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        global_idx = int(self.indices[item])
        meta_item = {
            "global_idx": global_idx,
            "env_id": int(self.meta["env_id"][global_idx]),
            "subject_id": int(self.meta["subject_id"][global_idx]),
            "action_id": int(self.meta["action_id"][global_idx]),
            "frame_id": int(self.meta["frame_id"][global_idx]),
            "seq_id": int(self.meta["seq_id"][global_idx]),
        }
        return {
            "x": self.x_all[global_idx, self.feature_channels],
            "y": self.y_all[global_idx],
            "conf": self.conf_all[global_idx],
            "meta": meta_item,
        }
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
pytest tests/test_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dataset/memmap_dataset.py tests/test_dataset.py
git commit -m "feat: add memmap pose dataset"
git push
```

## Task 6: Build And Inspect Scripts

**Files:**
- Create: `scripts/scan_gt_stats.py`
- Create: `scripts/build_memmap.py`
- Create: `scripts/inspect_memmap.py`

- [ ] **Step 1: Create GT stats scanner**

Create `scripts/scan_gt_stats.py` with this behavior:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/build_memmap.yaml")
    parser.add_argument("--gt-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pattern = cfg["paths"]["gt_pattern"]
    gt_root = Path(args.gt_root)
    values = []
    invalid_files = []
    for subject_id in range(1, 41):
        env_id = (subject_id - 1) // 10 + 1
        for action_id in range(1, 28):
            path = gt_root / pattern.format(env_id=env_id, subject_id=subject_id, action_id=action_id)
            if not path.exists():
                invalid_files.append(str(path))
                continue
            gt = np.load(path)
            if gt.shape != (297, 17, 3):
                raise ValueError(f"{path} has shape {gt.shape}, expected [297,17,3]")
            xy = gt[..., :2]
            values.append(xy[np.isfinite(xy)])

    merged = np.concatenate(values) if values else np.array([0.0], dtype=np.float32)
    stats = {
        "xy_min": float(merged.min()),
        "xy_max": float(merged.max()),
        "abs_max": float(np.abs(merged).max()),
        "missing_files": invalid_files,
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create memmap inspector**

Create `scripts/inspect_memmap.py`:

```python
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create build script**

Create `scripts/build_memmap.py` with these functions and main flow:

```python
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch
import yaml

from dataset.features import build_amplitude_features
from dataset.labels import normalize_gt_sequence
from dataset.meta import build_meta_arrays, env_id_from_subject, global_index


OUTPUT_FILES = ["X_all.npy", "Y_all.npy", "Conf_all.npy", "meta.npz", "meta_build.json"]


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
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"{key!r} not found in {path}")
            return np.asarray(handle[key])


def standardize_csi_frame(raw: np.ndarray, path: Path) -> np.ndarray:
    if raw.shape != (3, 114, 10):
        raise ValueError(f"{path} raw CSIamp shape must be [3,114,10], got {raw.shape}")
    return np.transpose(raw, (2, 0, 1)).astype(np.float32, copy=False)


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


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = select_device(args.device)
    csi_root = Path(args.csi_root)
    gt_root = Path(args.gt_root)
    output_root = Path(args.output_root)
    ensure_output_root(output_root, overwrite=args.overwrite)

    num_subjects = cfg["dataset"]["num_envs"] * cfg["dataset"]["subjects_per_env"]
    num_actions = cfg["dataset"]["num_actions"]
    num_frames = cfg["dataset"]["num_frames"]
    total = num_subjects * num_actions * num_frames

    x_all = np.lib.format.open_memmap(output_root / "X_all.npy", mode="w+", dtype=np.float32, shape=(total, 12, 10, 114))
    y_all = np.lib.format.open_memmap(output_root / "Y_all.npy", mode="w+", dtype=np.float32, shape=(total, 17, 2))
    conf_all = np.lib.format.open_memmap(output_root / "Conf_all.npy", mode="w+", dtype=np.float32, shape=(total, 17))

    meta = build_meta_arrays(num_subjects=num_subjects, num_actions=num_actions, num_frames=num_frames)
    np.savez(output_root / "meta.npz", **meta)

    csi_pattern = cfg["paths"]["csi_pattern"]
    gt_pattern = cfg["paths"]["gt_pattern"]
    csi_key = cfg["mat_keys"]["csi_key"]
    feature_cfg = cfg["feature"]
    sequence_count = 0
    invalid_keypoints = 0
    coord_formats: dict[str, int] = {}

    for subject_id in range(1, num_subjects + 1):
        env_id = env_id_from_subject(subject_id, subjects_per_env=cfg["dataset"]["subjects_per_env"])
        for action_id in range(1, num_actions + 1):
            frames = []
            for frame_zero_based in range(num_frames):
                frame_id_1based = frame_zero_based + 1
                csi_path = csi_root / csi_pattern.format(action_id=action_id, subject_id=subject_id, frame_id_1based=frame_id_1based)
                raw = read_mat_key(csi_path, csi_key)
                frames.append(standardize_csi_frame(raw, csi_path))
            csi_seq = np.stack(frames, axis=0)

            gt_path = gt_root / gt_pattern.format(env_id=env_id, subject_id=subject_id, action_id=action_id)
            gt_seq = np.load(gt_path)
            y_seq, conf_seq, label_stats = normalize_gt_sequence(gt_seq)
            invalid_keypoints += int(label_stats["invalid_keypoints"])
            coord_format = str(label_stats["coord_format"])
            coord_formats[coord_format] = coord_formats.get(coord_format, 0) + 1

            x_seq_t = build_amplitude_features(torch.as_tensor(csi_seq, dtype=torch.float32, device=device), **feature_cfg)
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
        "invalid_keypoints": invalid_keypoints,
        "coord_formats": coord_formats,
        "config": cfg,
    }
    (output_root / "meta_build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest tests/test_features.py tests/test_labels.py tests/test_meta.py tests/test_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/scan_gt_stats.py scripts/build_memmap.py scripts/inspect_memmap.py
git commit -m "feat: add memmap build scripts"
git push
```

## Task 7: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run all tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify Git-tracked files do not include generated data**

Run:

```bash
git ls-files
```

Expected: tracked files are source, tests, config, requirements, README, and docs only. No `X_all.npy`, `Y_all.npy`, `Conf_all.npy`, `meta.npz`, or `meta_build.json`.

- [ ] **Step 3: Verify status**

Run:

```bash
git -c core.excludesFile= status --short --branch
```

Expected: branch is clean and tracks `origin/main`.

## Plan Self-Review

Spec coverage:

- Amplitude feature definitions are covered in Task 3.
- Feature group encoding and ablation channel selection are covered in Tasks 3 and 5.
- GT cleanup and coordinate normalization are covered in Task 4.
- Neutral metadata and source-only split rules are covered in Task 2.
- Memmap Dataset reading is covered in Task 5.
- Build, scan, and inspect scripts are covered in Task 6.
- Generated data exclusion is covered in Task 7.

Red-flag scan:

- No task contains deferred behavior, unresolved decisions, or unspecified file paths.
- `finetune` is intentionally specified as `NotImplementedError` for the first version.

Type consistency:

- `X_all.npy` is float32 `[320760,12,10,114]`.
- `Y_all.npy` is float32 `[320760,17,2]`.
- `Conf_all.npy` is float32 `[320760,17]`.
- `meta.npz` fields and dtypes match the design.
