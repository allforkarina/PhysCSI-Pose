# CSI Domain Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first explicit CSI domain-alignment training path for Wi_Posev2 using axial feature CORAL during cross-domain few-shot finetuning.

**Architecture:** Keep the existing supervised source-only and few-shot finetune paths unchanged. Add a narrow `finetune_align` mode that trains on source labeled batches and target few-shot labeled batches while minimizing CORAL between source and target axial encoder features. Expose only the D1/D4 ablation controls needed for the first experiment: `--align-loss`, `--align-layer`, and `--align-weight`.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing NPY memmap dataloaders.

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `models/wiflow_model.py` | Modify | Add `encode_features()` and keep `forward()` backward-compatible. |
| `train.py` | Modify | Add CORAL loss, alignment config fields, `finetune_align` loop, CLI options, and checkpoint config storage. |
| `tests/test_wiflow_model.py` | Create | Verify feature extraction shape and `forward()` compatibility. |
| `tests/test_domain_alignment.py` | Create | Verify CORAL behavior, config parsing, and aligned epoch logging with synthetic loaders. |
| `AGENTS.md` | Modify | Document the new cross-domain alignment training command and physical ablation purpose. |

---

### Task 1: Add Feature Extraction Contract

**Files:**
- Create: `tests/test_wiflow_model.py`
- Modify: `models/wiflow_model.py`

- [ ] **Step 1: Write the failing feature interface tests**

Create `tests/test_wiflow_model.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WiFlowModel


def test_encode_features_returns_axial_feature_map() -> None:
    model = WiFlowModel()
    x = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        features = model.encode_features(x)

    assert features.shape == (2, 256, 29, 16)


def test_forward_decodes_encoded_features() -> None:
    model = WiFlowModel()
    x = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        prediction = model(x)
        decoded = model.decode_features(model.encode_features(x))

    assert prediction.shape == (2, 18, 2)
    assert torch.allclose(prediction, decoded)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_wiflow_model.py -v
```

Expected: `test_encode_features_returns_axial_feature_map` fails because `WiFlowModel` has no `encode_features`.

- [ ] **Step 3: Implement minimal feature interface**

Modify `models/wiflow_model.py`:

```python
    def encode_features(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError("WiFlowModel expects input shaped [B, 3, 114, 64]")
        x = self.spatial_encoder(x)
        return self.axial_encoder(x)

    def decode_features(self, x: torch.Tensor):
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        return self.decode_features(self.encode_features(x))
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_wiflow_model.py -v
```

Expected: both tests pass.

---

### Task 2: Add CORAL Alignment Loss

**Files:**
- Create: `tests/test_domain_alignment.py`
- Modify: `train.py`

- [ ] **Step 1: Write failing CORAL tests**

Create the first part of `tests/test_domain_alignment.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import ALIGN_LAYERS, ALIGN_LOSSES, coral_loss, compute_alignment_loss


def test_coral_loss_is_zero_for_matching_features() -> None:
    feature = torch.randn(4, 3, 2, 2)

    loss = coral_loss(feature, feature.clone())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_coral_loss_increases_for_shifted_covariance() -> None:
    source = torch.randn(4, 3, 2, 2)
    target = source.clone()
    target[:, 0] = target[:, 0] * 3.0

    loss = coral_loss(source, target)

    assert loss > 0


def test_compute_alignment_loss_supports_none_and_coral() -> None:
    source = torch.randn(4, 3, 2, 2)
    target = torch.randn(4, 3, 2, 2)

    assert compute_alignment_loss(source, target, "none").item() == 0.0
    assert compute_alignment_loss(source, target, "coral").item() >= 0.0
    assert ALIGN_LOSSES == ("none", "coral")
    assert ALIGN_LAYERS == ("axial",)
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py -v
```

Expected: import fails because `coral_loss`, `compute_alignment_loss`, `ALIGN_LOSSES`, and `ALIGN_LAYERS` do not exist.

- [ ] **Step 3: Implement minimal CORAL support**

Modify `train.py` near constants and loss helpers:

```python
ALIGN_LOSSES: tuple[str, ...] = ("none", "coral")
ALIGN_LAYERS: tuple[str, ...] = ("axial",)


def _flatten_alignment_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim < 2:
        raise ValueError("Alignment feature must include batch and feature dimensions")
    if feature.ndim == 2:
        return feature
    reduce_dims = tuple(range(2, feature.ndim))
    return feature.mean(dim=reduce_dims)


def coral_loss(source_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
    source = _flatten_alignment_feature(source_feature)
    target = _flatten_alignment_feature(target_feature)
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_zeros(())

    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    source_cov = source.t().matmul(source) / (source.shape[0] - 1)
    target_cov = target.t().matmul(target) / (target.shape[0] - 1)
    feature_dim = source.shape[1]
    return (source_cov - target_cov).pow(2).sum() / (4.0 * feature_dim * feature_dim)


def compute_alignment_loss(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    align_loss: str,
) -> torch.Tensor:
    if align_loss == "none":
        return source_feature.new_zeros(())
    if align_loss == "coral":
        return coral_loss(source_feature, target_feature)
    raise ValueError(f"Unknown align_loss: {align_loss}. Valid options: {ALIGN_LOSSES}")
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py -v
```

Expected: CORAL tests pass.

---

### Task 3: Add `finetune_align` Epoch Logic

**Files:**
- Modify: `tests/test_domain_alignment.py`
- Modify: `train.py`

- [ ] **Step 1: Write failing aligned epoch test**

Append to `tests/test_domain_alignment.py`:

```python
from torch import nn

from train import TrainConfig, run_alignment_finetune_epoch


class TinyAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(2, 2, bias=False)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x).view(x.shape[0], 1, 2)


def _tiny_batch(values: torch.Tensor) -> dict:
    return {
        "csi_amplitude": values,
        "keypoints": torch.zeros(values.shape[0], 1, 2),
    }


def test_run_alignment_finetune_epoch_logs_pose_and_alignment_terms() -> None:
    model = TinyAlignmentModel()
    source_loader = [_tiny_batch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))]
    target_loader = [_tiny_batch(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))]
    config = TrainConfig(
        dataset_root="unused",
        mode="finetune_align",
        align_loss="coral",
        align_weight=0.25,
        bone_loss_weight=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_alignment_finetune_epoch(
        model,
        source_loader,
        target_loader,
        config,
        torch.device("cpu"),
        optimizer,
    )

    assert metrics["loss"] >= metrics["source_loss"]
    assert metrics["loss"] >= metrics["target_loss"]
    assert metrics["align_loss"] > 0.0
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py::test_run_alignment_finetune_epoch_logs_pose_and_alignment_terms -v
```

Expected: import fails because `run_alignment_finetune_epoch` does not exist.

- [ ] **Step 3: Implement aligned epoch helper**

Modify `train.py`:

```python
def run_alignment_finetune_epoch(
    model: nn.Module,
    source_loader: Iterable[Mapping[str, torch.Tensor]],
    target_loader: Iterable[Mapping[str, torch.Tensor]],
    criterion_config: TrainConfig,
    device: torch.device,
    optimizer: AdamW,
    scheduler: LRScheduler | None = None,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    sample_count = 0

    for source_batch, target_batch in zip(source_loader, target_loader):
        source_input, source_target = prepare_model_input(source_batch, device)
        target_input, target_target = prepare_model_input(target_batch, device)

        optimizer.zero_grad(set_to_none=True)
        source_feature = model.encode_features(source_input)
        target_feature = model.encode_features(target_input)
        source_prediction = model.decode_features(source_feature)
        target_prediction = model.decode_features(target_feature)
        source_losses = compute_losses(source_prediction, source_target, criterion_config.bone_loss_weight)
        target_losses = compute_losses(target_prediction, target_target, criterion_config.bone_loss_weight)
        align = compute_alignment_loss(source_feature, target_feature, criterion_config.align_loss)
        loss = source_losses["loss"] + target_losses["loss"] + criterion_config.align_weight * align
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=criterion_config.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = target_target.shape[0]
        sample_count += batch_size
        values = {
            "loss": loss,
            "source_loss": source_losses["loss"],
            "target_loss": target_losses["loss"],
            "align_loss": align,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return average_meter_totals(totals, sample_count)
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py -v
```

Expected: alignment tests pass.

---

### Task 4: Add `finetune_align` Training Path and CLI

**Files:**
- Modify: `tests/test_domain_alignment.py`
- Modify: `train.py`

- [ ] **Step 1: Write failing CLI/config test**

Append to `tests/test_domain_alignment.py`:

```python
from unittest.mock import patch

from train import parse_args


def test_parse_args_accepts_finetune_align_options() -> None:
    argv = [
        "train.py",
        "--mode", "finetune_align",
        "--dataset-root", "data/mmfi_pose",
        "--source-envs", "env1",
        "--target-envs", "env2",
        "--finetune-from", "outputs/source/best_val_mpjpe.pth",
        "--align-loss", "coral",
        "--align-layer", "axial",
        "--align-weight", "0.1",
    ]

    with patch("sys.argv", argv):
        args = parse_args()

    assert args.mode == "finetune_align"
    assert args.align_loss == "coral"
    assert args.align_layer == "axial"
    assert args.align_weight == 0.1
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py::test_parse_args_accepts_finetune_align_options -v
```

Expected: parser rejects `finetune_align` or alignment arguments.

- [ ] **Step 3: Implement config and training path**

Modify `TrainConfig`:

```python
    align_loss: str = "none"
    align_layer: str = "axial"
    align_weight: float = 0.0
```

Add `_run_finetune_align()` that:

1. Requires `--source-envs`, `--target-envs`, and `--finetune-from`.
2. Creates a source train loader with `create_memmap_data_loader(split="train", envs=source_envs)`.
3. Creates target few-shot loaders with `create_few_shot_data_loader(...)`.
4. Loads the checkpoint into `WiFlowModel`.
5. Applies `--trainable-groups` or deprecated `--freeze-tier`.
6. Uses `steps_per_epoch=min(len(source_loader), len(target_train_loader))`.
7. Logs `train_loss`, `source_loss`, `target_loss`, `align_loss`, `align_weight`, and `current_lr`.
8. Saves `best_train_loss.pth`, epoch checkpoints, and `few_shot_train_indices.npy`.

Modify `run_training()`:

```python
    elif config.mode == "finetune_align":
        _run_finetune_align(config, device, output_dir)
```

Modify parser:

```python
    parser.add_argument("--mode", required=True, choices=("source_only", "finetune", "finetune_align"))
    parser.add_argument("--align-loss", default="none", choices=ALIGN_LOSSES)
    parser.add_argument("--align-layer", default="axial", choices=ALIGN_LAYERS)
    parser.add_argument("--align-weight", type=float, default=0.0)
```

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_domain_alignment.py -v
```

Expected: all domain alignment tests pass.

---

### Task 5: Update Repository Documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the alignment command**

Add a short section after the trainable-group finetune ablations:

```powershell
# Explicit CSI domain alignment: keep pose supervision while aligning source/target axial CSI features
python train.py --mode finetune_align --dataset-root data\mmfi_pose --source-envs env1 --target-envs env2 --output-dir outputs\ft_align_coral_w01 --finetune-from outputs\source_baseline\best_val_mpjpe.pth --trainable-groups encoder --align-loss coral --align-layer axial --align-weight 0.1 --epochs 30
```

Document that `--align-loss` supports `none` and `coral`, `--align-layer` supports `axial`, and `--align-weight` is swept for D4.

- [ ] **Step 2: Verify docs mention the physical claim**

Run:

```powershell
Select-String -Path AGENTS.md -Pattern "domain alignment","CORAL","axial CSI features"
```

Expected: all three phrases are present.

---

### Task 6: Full Verification and Commit

**Files:** all modified files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
conda activate WiFiPose; pytest tests\test_wiflow_model.py tests\test_domain_alignment.py tests\test_trainable_groups.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
conda activate WiFiPose; pytest
```

Expected: all tests pass.

- [ ] **Step 3: Check git status**

Run:

```powershell
git status --short
```

Expected: only intended files are modified, plus existing untracked local tool directories.

- [ ] **Step 4: Commit and push**

Run:

```powershell
git add docs/superpowers/plans/2026-06-07-csi-domain-alignment.md models/wiflow_model.py train.py tests/test_wiflow_model.py tests/test_domain_alignment.py AGENTS.md
git commit -m "Add CSI domain alignment finetune mode"
git push origin main
```
