# Train Eval Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable training and evaluation pipeline for PhysCSI-Pose using fixed-length temporal windows from the existing memmap dataset.

**Architecture:** Keep the neutral memmap data layer unchanged and add a small `engine/` package for window indexing, loss, metrics, and loop utilities. Add root-level `train.py` and `eval.py` as protocol-agnostic entry points driven by YAML plus CLI overrides, and add `PhysCSIPoseNet` as the single model wrapper consumed by the loops.

**Tech Stack:** Python, PyTorch, NumPy memmap, PyYAML, pytest, existing `dataset/` and `models/` modules.

---

## File Structure

- Create: `engine/__init__.py`
  - Exports public engine helpers.
- Create: `engine/window_dataset.py`
  - Builds or loads cached temporal window indices.
  - Reads existing `X_all.npy`, `Y_all.npy`, `Conf_all.npy`, and `meta.npz`.
  - Returns fixed-shape windows: `x [L,C,10,114]`, `y [L,17,2]`, `conf [L,17]`, and metadata.
- Create: `engine/losses.py`
  - Implements confidence-masked SmoothL1 coordinate loss.
- Create: `engine/metrics.py`
  - Implements masked MPJPE, PCK at configured thresholds, per-joint PCK, prediction joint std, GT joint std, and overlapping-window mean aggregation.
- Create: `engine/loops.py`
  - Implements `train_one_epoch` and `evaluate_one_epoch`.
- Create: `models/physcsi_pose.py`
  - Wraps `AmpFeatureMixEncoder`, `PoseAwareTokenProjection`, `TemporalLiteTransformer`, and `PoseHeatmapDecoder`.
- Modify: `models/amp_feature_mix_encoder.py`
  - Add configurable `input_channels`; default remains 12.
- Modify: `models/__init__.py`
  - Export `PhysCSIPoseNet`.
- Create: `configs/train.yaml`
  - Default train configuration.
- Create: `configs/eval.yaml`
  - Default eval configuration.
- Create: `train.py`
  - Generic training entry point.
- Create: `eval.py`
  - Generic evaluation entry point.
- Modify: `.gitignore`
  - Keep `runs/` and `outputs/` ignored. They are already ignored; verify no change is needed.
- Create or modify tests:
  - `tests/test_window_dataset.py`
  - `tests/test_losses.py`
  - `tests/test_metrics.py`
  - `tests/test_physcsi_pose.py`
  - `tests/test_train_eval_config.py`
  - Update `tests/test_amp_feature_mix_encoder.py`

---

## Task 1: Config Defaults And CLI Merge Contract

**Files:**
- Create: `configs/train.yaml`
- Create: `configs/eval.yaml`
- Create: `tests/test_train_eval_config.py`
- Later consumed by: `train.py`, `eval.py`

- [ ] **Step 1: Write tests for config loading and override precedence**

Create `tests/test_train_eval_config.py` with tests that drive a small helper API to be implemented in `train.py`:

```python
from pathlib import Path

import yaml

from train import deep_update, load_config_with_overrides


def test_deep_update_preserves_unspecified_nested_values():
    base = {"train": {"epochs": 100, "batch_size": 32}, "model": {"token_dim": 128}}
    override = {"train": {"batch_size": 64}}

    merged = deep_update(base, override)

    assert merged["train"]["epochs"] == 100
    assert merged["train"]["batch_size"] == 64
    assert merged["model"]["token_dim"] == 128


def test_cli_overrides_win_over_yaml(tmp_path: Path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"experiment": {"env_id": 1}, "train": {"batch_size": 32}}),
        encoding="utf-8",
    )

    cfg = load_config_with_overrides(
        cfg_path,
        overrides={"experiment.env_id": 3, "train.batch_size": 16},
    )

    assert cfg["experiment"]["env_id"] == 3
    assert cfg["train"]["batch_size"] == 16
```

- [ ] **Step 2: Run tests and verify they fail because helpers do not exist**

Run:

```bash
python -m pytest tests/test_train_eval_config.py -v
```

Expected: FAIL with import errors for `deep_update` or `load_config_with_overrides`.

- [ ] **Step 3: Add config files**

Create `configs/train.yaml`:

```yaml
experiment:
  protocol: source_only
  env_id: 1
  run_name: null
  output_dir: runs
  seed: 42
  device: auto

data:
  memmap_root: /data/WiFiPose/dataset/memmap
  feature_selection:
    names: [L_norm, D_center, F_sub, C_ant]
  window:
    length: 4
    stride: 1
    rebuild_index: false
  splits:
    train: train
    val: val

model:
  input_channels: auto
  token_dim: 128
  num_joints: 17
  encoder:
    out_channels: 128
  token_projection:
    num_attention_maps: 4
    dropout: 0.1
  temporal:
    min_window_length: 4
    max_window_length: 8
    num_layers: 2
    num_heads: 4
    ffn_expansion: 2
    attention_dropout: 0.1
    ffn_dropout: 0.1
    residual_dropout: 0.1
  decoder:
    heatmap_size: 64
    coord_min: -0.8
    coord_max: 0.8
    joint_hidden_dim: 128
    decoder_channels: 64
    seed_size: 8
    dropout: 0.1

train:
  epochs: 100
  batch_size: 32
  num_workers: 4
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  grad_clip_norm: 1.0
  amp: true

scheduler:
  name: cosine
  min_lr: 1.0e-5
  warmup_epochs: 10

checkpoint:
  monitor: val_loss
  mode: min
  save_best: true
  save_last: true

early_stopping:
  enabled: true
  monitor: val_loss
  mode: min
  patience: 20
  min_delta: 0.0

metrics:
  pck_thresholds: [0.05, 0.10, 0.20, 0.50]
```

Create `configs/eval.yaml`:

```yaml
experiment:
  protocol: source_only
  env_id: 1
  split: test
  eval_name: null
  output_dir: outputs
  device: auto
  save_predictions: false

data:
  memmap_root: /data/WiFiPose/dataset/memmap
  feature_selection:
    names: [L_norm, D_center, F_sub, C_ant]
  window:
    length: 4
    stride: 1
    rebuild_index: false

model:
  input_channels: auto
  token_dim: 128
  num_joints: 17
  encoder:
    out_channels: 128
  token_projection:
    num_attention_maps: 4
    dropout: 0.1
  temporal:
    min_window_length: 4
    max_window_length: 8
    num_layers: 2
    num_heads: 4
    ffn_expansion: 2
    attention_dropout: 0.1
    ffn_dropout: 0.1
    residual_dropout: 0.1
  decoder:
    heatmap_size: 64
    coord_min: -0.8
    coord_max: 0.8
    joint_hidden_dim: 128
    decoder_channels: 64
    seed_size: 8
    dropout: 0.1

eval:
  batch_size: 32
  num_workers: 4
  amp: true

metrics:
  pck_thresholds: [0.05, 0.10, 0.20, 0.50]
```

- [ ] **Step 4: Implement config helpers in `train.py`**

Add only pure helper functions first:

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def load_config_with_overrides(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_by_dotted_key(config, key, value)
    return config
```

- [ ] **Step 5: Run config tests**

Run:

```bash
python -m pytest tests/test_train_eval_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add configs/train.yaml configs/eval.yaml train.py tests/test_train_eval_config.py
git commit -m "feat: add train eval config defaults"
```

---

## Task 2: Configurable Encoder And End-To-End Model Wrapper

**Files:**
- Modify: `models/amp_feature_mix_encoder.py`
- Create: `models/physcsi_pose.py`
- Modify: `models/__init__.py`
- Modify: `tests/test_amp_feature_mix_encoder.py`
- Create: `tests/test_physcsi_pose.py`

- [ ] **Step 1: Write failing tests for variable input channels**

Append to `tests/test_amp_feature_mix_encoder.py`:

```python
def test_encoder_accepts_configurable_input_channels():
    encoder = AmpFeatureMixEncoder(input_channels=6)
    x = torch.randn(2, 6, 10, 114)

    z = encoder(x)

    assert z.shape == (2, 128, 10, 29)


def test_configurable_encoder_guard_reports_expected_channels():
    encoder = AmpFeatureMixEncoder(input_channels=6)
    x = torch.randn(2, 12, 10, 114)

    with pytest.raises(AssertionError, match="expected 6 input channels"):
        encoder(x)
```

- [ ] **Step 2: Write failing tests for `PhysCSIPoseNet`**

Create `tests/test_physcsi_pose.py`:

```python
from __future__ import annotations

import torch

from models import PhysCSIPoseNet


def test_physcsi_pose_net_outputs_window_coordinates():
    model = PhysCSIPoseNet(input_channels=12, token_dim=128, num_joints=17)
    x = torch.randn(2, 4, 12, 10, 114)

    pred = model(x)

    assert pred.shape == (2, 4, 17, 2)
    assert torch.isfinite(pred).all()
    assert pred.min() >= -0.80001
    assert pred.max() <= 0.80001


def test_physcsi_pose_net_accepts_feature_ablation_channels():
    model = PhysCSIPoseNet(input_channels=6, token_dim=128, num_joints=17)
    x = torch.randn(2, 4, 6, 10, 114)

    pred = model(x)

    assert pred.shape == (2, 4, 17, 2)


def test_physcsi_pose_net_returns_auxiliary_outputs():
    model = PhysCSIPoseNet(input_channels=12, token_dim=128, num_joints=17)
    x = torch.randn(1, 4, 12, 10, 114)

    pred, aux = model(x, return_aux=True)

    assert pred.shape == (1, 4, 17, 2)
    assert aux["encoder_maps"].shape == (1, 4, 128, 10, 29)
    assert aux["tokens"].shape == (1, 4, 128)
    assert aux["temporal_tokens"].shape == (1, 4, 128)
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_amp_feature_mix_encoder.py tests/test_physcsi_pose.py -v
```

Expected: FAIL because `AmpFeatureMixEncoder(input_channels=...)` and `PhysCSIPoseNet` do not exist yet.

- [ ] **Step 4: Modify `AmpFeatureMixEncoder`**

Implement:

```python
def __init__(self, input_channels: int = 12) -> None:
    super().__init__()
    assert input_channels >= 1, f"expected input_channels >= 1, got {input_channels}"
    self.input_channels = input_channels
    self.stage0 = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=1),
        nn.GroupNorm(num_groups=8, num_channels=32),
        nn.GELU(),
    )
```

Update the channel guard:

```python
assert x.shape[1] == self.input_channels, (
    f"expected {self.input_channels} input channels, got {x.shape[1]}"
)
```

- [ ] **Step 5: Implement `PhysCSIPoseNet`**

Create `models/physcsi_pose.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn

from models.amp_feature_mix_encoder import AmpFeatureMixEncoder
from models.pose_aware_token_projection import PoseAwareTokenProjection
from models.pose_heatmap_decoder import PoseHeatmapDecoder
from models.temporal_lite_transformer import TemporalLiteTransformer


class PhysCSIPoseNet(nn.Module):
    """End-to-end PhysCSI-Pose model for temporal window inputs.

    Input:  [B,L,C,10,114]
    Output: [B,L,17,2]
    """

    def __init__(
        self,
        input_channels: int = 12,
        token_dim: int = 128,
        num_joints: int = 17,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_max_window_length: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.token_dim = token_dim
        self.num_joints = num_joints

        self.encoder = AmpFeatureMixEncoder(input_channels=input_channels)
        self.token_projection = PoseAwareTokenProjection(
            in_channels=128,
            token_dim=token_dim,
            num_attention_maps=4,
            dropout=dropout,
        )
        self.temporal = TemporalLiteTransformer(
            input_dim=token_dim,
            min_window_length=4,
            max_window_length=temporal_max_window_length,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            ffn_expansion=2,
            attention_dropout=dropout,
            ffn_dropout=dropout,
            residual_dropout=dropout,
        )
        self.decoder = PoseHeatmapDecoder(
            input_dim=token_dim,
            num_joints=num_joints,
            heatmap_size=64,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.ndim == 5, f"expected 5D input [B,L,C,10,114], got ndim={x.ndim}"
        b, l, c, t, s = x.shape
        assert c == self.input_channels, f"expected {self.input_channels} channels, got {c}"
        assert t == 10, f"expected 10 packets, got {t}"
        assert s == 114, f"expected 114 subcarriers, got {s}"

        x_flat = x.reshape(b * l, c, t, s)
        encoder_maps = self.encoder(x_flat).reshape(b, l, 128, 10, 29)
        tokens = self.token_projection(encoder_maps)
        temporal_tokens = self.temporal(tokens)
        pred = self.decoder(temporal_tokens)

        if return_aux:
            return pred, {
                "encoder_maps": encoder_maps,
                "tokens": tokens,
                "temporal_tokens": temporal_tokens,
            }
        return pred
```

Update `models/__init__.py`:

```python
from models.physcsi_pose import PhysCSIPoseNet

__all__ = [
    "AmpFeatureMixEncoder",
    "PoseAwareTokenProjection",
    "PoseHeatmapDecoder",
    "TemporalLiteTransformer",
    "PhysCSIPoseNet",
]
```

- [ ] **Step 6: Run model tests**

Run:

```bash
python -m pytest tests/test_amp_feature_mix_encoder.py tests/test_physcsi_pose.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add models/amp_feature_mix_encoder.py models/physcsi_pose.py models/__init__.py tests/test_amp_feature_mix_encoder.py tests/test_physcsi_pose.py
git commit -m "feat: add end to end PhysCSI pose model"
```

---

## Task 3: Temporal Window Dataset And Window Index Cache

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/window_dataset.py`
- Create: `tests/test_window_dataset.py`

- [ ] **Step 1: Write failing tests for fixed windows and cache reuse**

Create `tests/test_window_dataset.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.window_dataset import WindowMemmapPoseDataset, build_or_load_window_index


def write_sequence_cache(root: Path) -> None:
    n = 10
    x = np.zeros((n, 12, 10, 114), dtype=np.float32)
    for i in range(n):
        x[i] = float(i)
    np.save(root / "X_all.npy", x)
    np.save(root / "Y_all.npy", np.zeros((n, 17, 2), dtype=np.float32))
    np.save(root / "Conf_all.npy", np.ones((n, 17), dtype=np.float32))
    np.savez(
        root / "meta.npz",
        global_idx=np.arange(n, dtype=np.int64),
        env_id=np.ones(n, dtype=np.uint8),
        subject_id=np.ones(n, dtype=np.uint8),
        action_id=np.ones(n, dtype=np.uint8),
        frame_id=np.arange(n, dtype=np.uint16),
        seq_id=np.zeros(n, dtype=np.uint16),
    )


def test_build_window_index_counts_contiguous_windows(tmp_path: Path):
    write_sequence_cache(tmp_path)
    index_path = tmp_path / "window_index" / "train.npz"

    index = build_or_load_window_index(
        memmap_root=tmp_path,
        index_path=index_path,
        protocol="source_only",
        env_id=1,
        split="train",
        window_length=4,
        stride=1,
        rebuild=True,
    )

    assert index["start_global_idx"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert index["seq_id"].tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert index_path.exists()


def test_window_dataset_returns_fixed_shapes_and_feature_selection(tmp_path: Path):
    write_sequence_cache(tmp_path)
    ds = WindowMemmapPoseDataset(
        memmap_root=tmp_path,
        index_path=tmp_path / "window_index" / "train.npz",
        protocol="source_only",
        env_id=1,
        split="train",
        window_length=4,
        stride=1,
        features=["l_norm", "f_sub"],
        rebuild_index=True,
    )

    item = ds[0]

    assert item["x"].shape == (4, 6, 10, 114)
    assert item["y"].shape == (4, 17, 2)
    assert item["conf"].shape == (4, 17)
    assert item["global_idx"].tolist() == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_window_dataset.py -v
```

Expected: FAIL because `engine.window_dataset` does not exist.

- [ ] **Step 3: Implement window index builder**

Create `engine/window_dataset.py` with these public functions/classes:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dataset.features import selected_feature_channels
from dataset.splits import source_only_subjects


def _eligible_frame_indices(meta: np.lib.npyio.NpzFile, protocol: str, env_id: int, split: str) -> np.ndarray:
    if protocol != "source_only":
        raise NotImplementedError(f"protocol {protocol!r} is not implemented")
    subjects = np.array(source_only_subjects(env_id=env_id, split=split), dtype=np.uint8)
    mask = (meta["env_id"] == env_id) & np.isin(meta["subject_id"], subjects)
    return meta["global_idx"][mask].astype(np.int64)


def _build_window_index_from_meta(
    meta: np.lib.npyio.NpzFile,
    eligible_indices: np.ndarray,
    window_length: int,
    stride: int,
) -> dict[str, np.ndarray]:
    eligible = set(int(i) for i in eligible_indices.tolist())
    starts: list[int] = []
    seq_ids: list[int] = []
    start_frames: list[int] = []
    for seq_id in np.unique(meta["seq_id"][eligible_indices]):
        seq_mask = meta["seq_id"] == seq_id
        seq_indices = meta["global_idx"][seq_mask].astype(np.int64)
        seq_indices = np.array([idx for idx in seq_indices.tolist() if int(idx) in eligible], dtype=np.int64)
        seq_indices.sort()
        for offset in range(0, len(seq_indices) - window_length + 1, stride):
            window = seq_indices[offset : offset + window_length]
            frame_ids = meta["frame_id"][window]
            if np.all(np.diff(frame_ids) == 1):
                starts.append(int(window[0]))
                seq_ids.append(int(seq_id))
                start_frames.append(int(frame_ids[0]))
    return {
        "start_global_idx": np.asarray(starts, dtype=np.int64),
        "seq_id": np.asarray(seq_ids, dtype=np.int64),
        "start_frame": np.asarray(start_frames, dtype=np.int64),
        "window_length": np.asarray(window_length, dtype=np.int64),
        "stride": np.asarray(stride, dtype=np.int64),
    }


def build_or_load_window_index(
    memmap_root: str | Path,
    index_path: str | Path,
    *,
    protocol: str,
    env_id: int,
    split: str,
    window_length: int,
    stride: int,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    index_path = Path(index_path)
    if index_path.exists() and not rebuild:
        loaded = np.load(index_path)
        return {key: loaded[key] for key in loaded.files}

    meta = np.load(Path(memmap_root) / "meta.npz")
    eligible = _eligible_frame_indices(meta, protocol=protocol, env_id=env_id, split=split)
    index = _build_window_index_from_meta(meta, eligible, window_length=window_length, stride=stride)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path, **index)
    return index
```

- [ ] **Step 4: Implement window dataset**

Append to `engine/window_dataset.py`:

```python
class WindowMemmapPoseDataset:
    def __init__(
        self,
        memmap_root: str | Path,
        index_path: str | Path,
        *,
        protocol: str,
        env_id: int,
        split: str,
        window_length: int,
        stride: int,
        features: list[str] | tuple[str, ...] | None = None,
        rebuild_index: bool = False,
    ) -> None:
        self.memmap_root = Path(memmap_root)
        self.window_length = window_length
        self.feature_channels = selected_feature_channels(features)
        self.x_all = np.load(self.memmap_root / "X_all.npy", mmap_mode="r")
        self.y_all = np.load(self.memmap_root / "Y_all.npy", mmap_mode="r")
        self.conf_all = np.load(self.memmap_root / "Conf_all.npy", mmap_mode="r")
        self.meta = np.load(self.memmap_root / "meta.npz")
        self.index = build_or_load_window_index(
            self.memmap_root,
            index_path,
            protocol=protocol,
            env_id=env_id,
            split=split,
            window_length=window_length,
            stride=stride,
            rebuild=rebuild_index,
        )

    def __len__(self) -> int:
        return int(self.index["start_global_idx"].shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        start = int(self.index["start_global_idx"][item])
        global_idx = np.arange(start, start + self.window_length, dtype=np.int64)
        x = self.x_all[global_idx][:, self.feature_channels]
        return {
            "x": x.astype(np.float32, copy=False),
            "y": self.y_all[global_idx].astype(np.float32, copy=False),
            "conf": self.conf_all[global_idx].astype(np.float32, copy=False),
            "global_idx": global_idx,
            "seq_id": int(self.index["seq_id"][item]),
            "start_frame": int(self.index["start_frame"][item]),
        }
```

Create `engine/__init__.py`:

```python
from engine.window_dataset import WindowMemmapPoseDataset, build_or_load_window_index

__all__ = ["WindowMemmapPoseDataset", "build_or_load_window_index"]
```

- [ ] **Step 5: Run window dataset tests**

Run:

```bash
python -m pytest tests/test_window_dataset.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add engine/__init__.py engine/window_dataset.py tests/test_window_dataset.py
git commit -m "feat: add temporal window memmap dataset"
```

---

## Task 4: Masked Losses And Evaluation Metrics

**Files:**
- Create: `engine/losses.py`
- Create: `engine/metrics.py`
- Create: `tests/test_losses.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write loss tests**

Create `tests/test_losses.py`:

```python
from __future__ import annotations

import torch

from engine.losses import masked_smooth_l1_loss


def test_masked_smooth_l1_ignores_conf_zero_points():
    pred = torch.tensor([[[[0.5, 0.5], [0.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.8, 0.8]]]])
    conf = torch.tensor([[[1.0, 0.0]]])

    loss = masked_smooth_l1_loss(pred, target, conf, beta=1.0)

    assert torch.allclose(loss, torch.tensor(0.125))


def test_masked_smooth_l1_all_invalid_returns_zero_with_grad():
    pred = torch.randn(2, 4, 17, 2, requires_grad=True)
    target = torch.randn(2, 4, 17, 2)
    conf = torch.zeros(2, 4, 17)

    loss = masked_smooth_l1_loss(pred, target, conf)
    loss.backward()

    assert torch.allclose(loss.detach(), torch.tensor(0.0))
    assert pred.grad is not None
```

- [ ] **Step 2: Write metrics tests**

Create `tests/test_metrics.py`:

```python
from __future__ import annotations

import torch

from engine.metrics import aggregate_window_predictions_mean, compute_pose_metrics


def test_compute_pose_metrics_masks_invalid_points():
    pred = torch.tensor([[[[0.0, 0.0], [0.5, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    conf = torch.tensor([[[1.0, 0.0]]])

    metrics = compute_pose_metrics(pred, target, conf, pck_thresholds=(0.05, 0.10))

    assert metrics["mpjpe_norm"] == 0.0
    assert metrics["pck_0.05"] == 1.0
    assert metrics["pck_0.10"] == 1.0


def test_compute_pose_metrics_detects_prediction_std():
    pred = torch.zeros(3, 1, 17, 2)
    pred[1, :, :, 0] = 0.2
    pred[2, :, :, 0] = 0.4
    target = torch.zeros_like(pred)
    conf = torch.ones(3, 1, 17)

    metrics = compute_pose_metrics(pred, target, conf, pck_thresholds=(0.5,))

    assert metrics["mean_joint_std"] > 0.0
    assert metrics["gt_mean_joint_std"] == 0.0


def test_aggregate_window_predictions_mean_averages_overlaps():
    pred = torch.tensor(
        [
            [[[0.0, 0.0]], [[2.0, 0.0]]],
            [[[4.0, 0.0]], [[6.0, 0.0]]],
        ]
    )
    global_idx = torch.tensor([[0, 1], [1, 2]])

    agg_pred, agg_idx = aggregate_window_predictions_mean(pred, global_idx)

    assert agg_idx.tolist() == [0, 1, 2]
    assert torch.allclose(agg_pred[:, 0, 0], torch.tensor([0.0, 3.0, 6.0]))
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_losses.py tests/test_metrics.py -v
```

Expected: FAIL because `engine.losses` and `engine.metrics` do not exist.

- [ ] **Step 4: Implement masked loss**

Create `engine/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smooth_l1_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    conf: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    assert pred_xy.shape == target_xy.shape, f"pred and target shape mismatch: {pred_xy.shape} vs {target_xy.shape}"
    assert pred_xy.shape[:-1] == conf.shape, f"conf shape {conf.shape} does not match pred {pred_xy.shape}"
    valid = (conf > 0).to(dtype=pred_xy.dtype)
    loss = F.smooth_l1_loss(pred_xy, target_xy, reduction="none", beta=beta)
    loss = loss * valid.unsqueeze(-1)
    denom = valid.sum() * pred_xy.shape[-1]
    return loss.sum() / (denom + eps)
```

- [ ] **Step 5: Implement metrics**

Create `engine/metrics.py` with these public functions:

```python
from __future__ import annotations

import torch


def aggregate_window_predictions_mean(
    pred_xy: torch.Tensor,
    global_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_pred = pred_xy.reshape(-1, pred_xy.shape[-2], pred_xy.shape[-1])
    flat_idx = global_idx.reshape(-1).to(device=pred_xy.device)
    unique_idx = torch.unique(flat_idx, sorted=True)
    out = torch.zeros(unique_idx.numel(), pred_xy.shape[-2], pred_xy.shape[-1], device=pred_xy.device)
    count = torch.zeros(unique_idx.numel(), 1, 1, device=pred_xy.device)
    inverse = torch.searchsorted(unique_idx, flat_idx)
    out.index_add_(0, inverse, flat_pred)
    count.index_add_(0, inverse, torch.ones_like(count[inverse]))
    return out / count.clamp_min(1.0), unique_idx


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def compute_pose_metrics(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    conf: torch.Tensor,
    *,
    pck_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
) -> dict[str, float | list[float]]:
    assert pred_xy.shape == target_xy.shape
    assert pred_xy.shape[:-1] == conf.shape
    valid = conf > 0
    dist = torch.linalg.norm(pred_xy - target_xy, dim=-1)
    denom = valid.sum().clamp_min(1)
    masked_dist = dist[valid]
    metrics: dict[str, float | list[float]] = {
        "mpjpe_norm": _safe_float(masked_dist.sum() / denom),
    }
    for threshold in pck_thresholds:
        correct = ((dist <= threshold) & valid).sum()
        key = f"pck_{threshold:.2f}"
        metrics[key] = _safe_float(correct / denom)
        per_joint = []
        for joint_id in range(pred_xy.shape[-2]):
            joint_valid = valid[..., joint_id]
            joint_denom = joint_valid.sum().clamp_min(1)
            joint_correct = ((dist[..., joint_id] <= threshold) & joint_valid).sum()
            per_joint.append(_safe_float(joint_correct / joint_denom))
        metrics[f"per_joint_{key}"] = per_joint

    pred_flat = pred_xy.reshape(-1, pred_xy.shape[-2], pred_xy.shape[-1])
    target_flat = target_xy.reshape(-1, target_xy.shape[-2], target_xy.shape[-1])
    pred_joint_std = pred_flat.std(dim=0, unbiased=False)
    gt_joint_std = target_flat.std(dim=0, unbiased=False)
    pred_joint_std_l2 = torch.linalg.norm(pred_joint_std, dim=-1)
    gt_joint_std_l2 = torch.linalg.norm(gt_joint_std, dim=-1)
    metrics["mean_joint_std"] = _safe_float(pred_joint_std_l2.mean())
    metrics["min_joint_std"] = _safe_float(pred_joint_std_l2.min())
    metrics["per_joint_std"] = [float(v) for v in pred_joint_std_l2.detach().cpu().tolist()]
    metrics["gt_mean_joint_std"] = _safe_float(gt_joint_std_l2.mean())
    metrics["gt_min_joint_std"] = _safe_float(gt_joint_std_l2.min())
    metrics["gt_per_joint_std"] = [float(v) for v in gt_joint_std_l2.detach().cpu().tolist()]
    return metrics
```

- [ ] **Step 6: Run metrics and loss tests**

Run:

```bash
python -m pytest tests/test_losses.py tests/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add engine/losses.py engine/metrics.py tests/test_losses.py tests/test_metrics.py
git commit -m "feat: add masked pose loss and metrics"
```

---

## Task 5: Train And Evaluate Loop Utilities

**Files:**
- Create: `engine/loops.py`
- Modify: `engine/__init__.py`
- Create: `tests/test_loops.py`

- [ ] **Step 1: Write tests using a tiny model and synthetic loader**

Create `tests/test_loops.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from engine.loops import evaluate_one_epoch, train_one_epoch


class TinyWindowDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.ones(4, 3, 10, 114),
            "y": torch.zeros(4, 17, 2),
            "conf": torch.ones(4, 17),
            "global_idx": torch.arange(idx * 4, idx * 4 + 4),
        }


class TinyPoseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(x.shape[0], x.shape[1], 17, 2)


def test_train_one_epoch_updates_model():
    model = TinyPoseModel()
    loader = DataLoader(TinyWindowDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = model.bias.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
        grad_clip_norm=1.0,
    )

    assert metrics["train_loss"] > 0
    assert not torch.allclose(model.bias.detach(), before)


def test_evaluate_one_epoch_returns_metrics():
    model = TinyPoseModel()
    loader = DataLoader(TinyWindowDataset(), batch_size=2)

    metrics = evaluate_one_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        amp_enabled=False,
        pck_thresholds=(0.05, 0.10),
    )

    assert "val_loss" in metrics
    assert "mpjpe_norm" in metrics
    assert "pck_0.05" in metrics
    assert "mean_joint_std" in metrics
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_loops.py -v
```

Expected: FAIL because `engine.loops` does not exist.

- [ ] **Step 3: Implement loop helpers**

Create `engine/loops.py`:

```python
from __future__ import annotations

from typing import Iterable

import torch

from engine.losses import masked_smooth_l1_loss
from engine.metrics import compute_pose_metrics


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(batch["x"])
            loss = masked_smooth_l1_loss(pred, batch["y"], batch["conf"])
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return {"train_loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    amp_enabled: bool,
    pck_thresholds: tuple[float, ...],
    loss_prefix: str = "val",
) -> dict[str, float | list[float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds = []
    targets = []
    confs = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred = model(batch["x"])
            loss = masked_smooth_l1_loss(pred, batch["y"], batch["conf"])
        batch_size = int(batch["x"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        preds.append(pred.detach().cpu())
        targets.append(batch["y"].detach().cpu())
        confs.append(batch["conf"].detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    conf_all = torch.cat(confs, dim=0)
    metrics = compute_pose_metrics(pred_all, target_all, conf_all, pck_thresholds=pck_thresholds)
    metrics[f"{loss_prefix}_loss"] = total_loss / max(total_items, 1)
    return metrics
```

Update `engine/__init__.py`:

```python
from engine.loops import evaluate_one_epoch, train_one_epoch
from engine.window_dataset import WindowMemmapPoseDataset, build_or_load_window_index

__all__ = [
    "WindowMemmapPoseDataset",
    "build_or_load_window_index",
    "train_one_epoch",
    "evaluate_one_epoch",
]
```

- [ ] **Step 4: Run loop tests**

Run:

```bash
python -m pytest tests/test_loops.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add engine/loops.py engine/__init__.py tests/test_loops.py
git commit -m "feat: add train eval loop helpers"
```

---

## Task 6: Root-Level `train.py`

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_script_smoke.py`

- [ ] **Step 1: Write smoke test for run directory creation without real data**

Create `tests/test_train_script_smoke.py`:

```python
from __future__ import annotations

from pathlib import Path

from train import build_run_dir, make_run_name, resolve_device


def test_make_run_name_includes_protocol_env_and_window():
    name = make_run_name(protocol="source_only", env_id=3, window_length=4)

    assert name.startswith("source_only_env03_L4_")


def test_build_run_dir_creates_expected_subdirectories(tmp_path: Path):
    run_dir = build_run_dir(tmp_path, run_name="exp001")

    assert run_dir == tmp_path / "exp001"
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "window_index").is_dir()


def test_resolve_device_auto_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    device = resolve_device("auto")

    assert device.type == "cpu"
```

- [ ] **Step 2: Run smoke tests and verify they fail**

Run:

```bash
python -m pytest tests/test_train_script_smoke.py -v
```

Expected: FAIL because `build_run_dir`, `make_run_name`, or `resolve_device` do not exist.

- [ ] **Step 3: Implement training helpers and main**

Extend `train.py` with:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.features import selected_feature_channels
from engine.loops import evaluate_one_epoch, train_one_epoch
from engine.window_dataset import WindowMemmapPoseDataset
from models import PhysCSIPoseNet
```

Add helpers:

```python
def make_run_name(protocol: str, env_id: int, window_length: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{protocol}_env{env_id:02d}_L{window_length}_{stamp}"


def build_run_dir(output_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(output_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "window_index").mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
```

Main implementation requirements:

```text
1. Parse --config, --env-id, --protocol, --run-name, --device, --batch-size, --epochs.
2. Load YAML and apply CLI overrides using existing helper.
3. Seed Python, NumPy, and PyTorch with config experiment.seed.
4. Select feature channels using dataset.features.selected_feature_channels.
5. Set model input_channels to len(selected_channels).
6. Build train and val WindowMemmapPoseDataset with index files:
   runs/<run_name>/window_index/train.npz
   runs/<run_name>/window_index/val.npz
7. Build DataLoaders with configured batch size, workers, pin_memory when CUDA.
8. Build PhysCSIPoseNet and move it to device.
9. Use AdamW, cosine schedule with 10 warmup epochs, AMP when CUDA and config train.amp is true.
10. Each epoch writes one JSON object to runs/<run_name>/metrics.jsonl.
11. Save best checkpoint by val_loss to checkpoints/best.pt.
12. Save last checkpoint each epoch to checkpoints/last.pt.
13. Stop when early_stopping patience is reached.
14. Save resolved config to runs/<run_name>/config_resolved.yaml.
```

Checkpoint dictionary format:

```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_metric": best_metric,
    "config": cfg,
}
```

- [ ] **Step 4: Run smoke tests**

Run:

```bash
python -m pytest tests/test_train_script_smoke.py tests/test_train_eval_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add train.py tests/test_train_script_smoke.py
git commit -m "feat: add configurable training entrypoint"
```

---

## Task 7: Root-Level `eval.py`

**Files:**
- Create: `eval.py`
- Create: `tests/test_eval_script_smoke.py`

- [ ] **Step 1: Write smoke tests for eval naming and output directory**

Create `tests/test_eval_script_smoke.py`:

```python
from __future__ import annotations

from pathlib import Path

from eval import build_eval_dir, make_eval_name


def test_make_eval_name_uses_checkpoint_parent_and_split():
    checkpoint = Path("runs/exp001/checkpoints/best.pt")

    name = make_eval_name(checkpoint=checkpoint, split="test")

    assert name.startswith("exp001_test_")


def test_build_eval_dir_creates_directory(tmp_path: Path):
    eval_dir = build_eval_dir(tmp_path, eval_name="exp001_test")

    assert eval_dir == tmp_path / "exp001_test"
    assert eval_dir.is_dir()
```

- [ ] **Step 2: Run smoke tests and verify they fail**

Run:

```bash
python -m pytest tests/test_eval_script_smoke.py -v
```

Expected: FAIL because `eval.py` does not exist.

- [ ] **Step 3: Implement eval helpers and main**

Create `eval.py` with helpers:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.features import selected_feature_channels
from engine.loops import evaluate_one_epoch
from engine.window_dataset import WindowMemmapPoseDataset
from models import PhysCSIPoseNet
from train import load_config_with_overrides, resolve_device


def make_eval_name(checkpoint: Path, split: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = checkpoint.parents[1].name
    return f"{run_name}_{split}_{stamp}"


def build_eval_dir(output_dir: str | Path, eval_name: str) -> Path:
    eval_dir = Path(output_dir) / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir
```

Main implementation requirements:

```text
1. Parse --config, --checkpoint, --env-id, --split, --eval-name, --device, --save-predictions.
2. Load YAML and CLI overrides.
3. Build output directory outputs/<eval_name>/.
4. Save resolved config to outputs/<eval_name>/config_resolved.yaml.
5. Build test WindowMemmapPoseDataset using outputs/<eval_name>/window_index/<split>.npz or checkpoint run window_index when available.
6. Build model with input_channels from selected features.
7. Load checkpoint["model"].
8. Run evaluate_one_epoch with loss_prefix set to split.
9. Save metrics JSON to outputs/<eval_name>/metrics.json.
10. Only when --save-predictions is set, run prediction collection and save predictions.npz.
```

For first implementation, prediction saving can use:

```python
np.savez(
    eval_dir / "predictions.npz",
    pred_xy=pred_xy.astype("float32"),
    global_idx=global_idx.astype("int64"),
)
```

- [ ] **Step 4: Run eval smoke tests**

Run:

```bash
python -m pytest tests/test_eval_script_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add eval.py tests/test_eval_script_smoke.py
git commit -m "feat: add configurable evaluation entrypoint"
```

---

## Task 8: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `.gitignore` only if `runs/` or `outputs/` are missing

- [ ] **Step 1: Update README training section**

Add concise usage examples:

```bash
python train.py --config configs/train.yaml --env-id 1 --protocol source_only --device auto
python train.py --config configs/train.yaml --env-id 3 --protocol source_only --run-name env03_baseline
python eval.py --config configs/eval.yaml --checkpoint runs/env03_baseline/checkpoints/best.pt --env-id 3 --split test
python eval.py --config configs/eval.yaml --checkpoint runs/env03_baseline/checkpoints/best.pt --env-id 3 --split test --save-predictions
```

Document outputs:

```text
runs/<run_name>/checkpoints/best.pt
runs/<run_name>/checkpoints/last.pt
runs/<run_name>/metrics.jsonl
runs/<run_name>/config_resolved.yaml
outputs/<eval_name>/metrics.json
```

- [ ] **Step 2: Update AGENTS project status**

Record:

```text
Training and evaluation code exists, but local verification still uses synthetic tests only.
Real training/eval should be run on the Linux server with memmap data under /data/WiFiPose/dataset/memmap.
runs/ and outputs/ are generated artifacts and must not be pushed.
```

- [ ] **Step 3: Run full local test suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS for all tests.

- [ ] **Step 4: Check git status only includes intended source files**

Run:

```bash
git status --short
```

Expected: only source, config, docs, and tests from this plan are modified or added. Existing unrelated untracked files such as `.reasonix/` and older untracked plan/spec files must remain untouched.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md AGENTS.md .gitignore
git commit -m "docs: document train eval workflow"
```

- [ ] **Step 6: Push all commits**

Run:

```bash
git push
```

Expected: push to `origin/main` succeeds.

---

## Self-Review

- Spec coverage: The plan covers configurable root-level `train.py` and `eval.py`, `runs/` and `outputs/`, source-only split by `env_id`, fixed temporal windows, cached window indices, feature selection for ablations, end-to-end model wrapper, masked SmoothL1 loss, MPJPE, four PCK thresholds, per-joint metrics, joint std, overlapping-window mean aggregation, checkpointing, early stopping, and synthetic local tests.
- Marker scan: No unresolved draft markers are present. The plan uses exact file paths, exact command lines, concrete test snippets, and concrete default YAML values.
- Type consistency: Dataset samples use `x [L,C,10,114]`, DataLoader batches use `x [B,L,C,10,114]`, model output uses `pred [B,L,17,2]`, and loss/metrics use `conf [B,L,17]`.
