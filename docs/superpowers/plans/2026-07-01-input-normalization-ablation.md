# Input Normalization Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible training selection and checkpoint-driven evaluation for the three precomputed CSI normalization variants.

**Architecture:** Define canonical normalization names beside the memmap filename mapping, propagate one `normalization` value through loader factories and `TrainConfig`, and save it through the existing checkpoint serializer. Evaluation loads model configuration and input normalization together so test and visualization paths cannot silently use a different CSI file; old checkpoints fall back to `global_minmax`.

**Tech Stack:** Python 3.10+, NumPy memmap, PyTorch, argparse, pytest.

---

## File Structure

- Modify `data/memmap_dataset.py`: define canonical names and select the corresponding precomputed CSI memmap.
- Modify `dataloader.py`: propagate `normalization` through standard, split, and few-shot loader factories.
- Modify `train.py`: expose the CLI option, store it in `TrainConfig`, and pass it to every training loader.
- Modify `eval.py`: load checkpoint configuration once and use its normalization for evaluation and pose visualization datasets.
- Create `tests/test_input_normalization.py`: focused dataset, loader, CLI, checkpoint, and compatibility tests.
- Modify `tests/test_eval_diagnostics.py`: update evaluation-main fakes and verify normalization propagation.
- Modify `README.md`: document the controlled three-run source-domain ablation.
- Modify `AGENTS.md`: keep repository workflow and CLI documentation accurate.

## Ablation Control Matrix

| ID | Factor | Physical hypothesis | Control | Metrics | Mechanism evidence | Expected failure mode |
| --- | --- | --- | --- | --- | --- | --- |
| N1 | `global_minmax` | Absolute amplitude and attenuation are useful pose cues. | Same env1 random-frame membership, seed, model, loss, optimizer, scheduler, batch size, and epochs. | Test MPJPE, PCK@0.2, PCK@0.5, per-joint metrics. | Prediction/GT variance ratio, std ratio, mean-pose distance. | Outliers compress typical CSI values and weaken gradients. |
| N2 | `global_zscore` | Global standardization improves optimization without removing between-frame gain. | N1 settings; only normalization changes. | Same as N1. | Same as N1 plus per-action deltas. | Global outliers distort mean/std. |
| N3 | `per_sample_zscore` | Relative antenna/subcarrier/time structure matters more than frame-level gain. | N1 settings; only normalization changes. | Same as N1. | Distal-joint and action breakdowns plus collapse diagnostics. | Removing absolute attenuation loses position cues. |

These runs are exploratory at seed 42. A final claim requires the same matrix at three or more seeds. The existing global statistics include all configured source-subject frames, so this implementation comparison must be labelled as not yet leakage-free.

### Task 1: Canonical Dataset and Loader Selection

**Files:**
- Create: `tests/test_input_normalization.py`
- Modify: `data/memmap_dataset.py`
- Modify: `dataloader.py`

- [ ] **Step 1: Write failing dataset-selection tests**

Create `tests/test_input_normalization.py` with a tiny dataset whose three CSI files contain distinct constants:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_normalization_dataset(data_dir: Path) -> None:
    data_dir.mkdir()
    shape = (2, 64, 3, 114)
    np.save(data_dir / "csi_gminmax.npy", np.full(shape, 1.0, dtype=np.float32))
    np.save(data_dir / "csi_gzscore.npy", np.full(shape, 2.0, dtype=np.float32))
    np.save(data_dir / "csi_zscore.npy", np.full(shape, 3.0, dtype=np.float32))
    np.save(data_dir / "ground_truth.npy", np.zeros((2, 17, 2), dtype=np.float32))
    np.savez(
        data_dir / "meta.npz",
        environment=np.array(["env1", "env1"]),
        sample=np.array(["S01", "S01"]),
        action=np.array(["A01", "A01"]),
        frame_idx=np.array([1, 2]),
    )


@pytest.mark.parametrize(
    ("normalization", "expected"),
    [
        ("global_minmax", 1.0),
        ("global_zscore", 2.0),
        ("per_sample_zscore", 3.0),
    ],
)
def test_memmap_dataset_selects_normalization_file(
    tmp_path: Path,
    normalization: str,
    expected: float,
) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    dataset = MemmapDataset(data_dir, split="all", normalization=normalization)

    assert dataset.normalization == normalization
    assert np.all(dataset[0]["csi"].numpy() == expected)


def test_memmap_dataset_rejects_unknown_normalization(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    with pytest.raises(ValueError, match="Unknown normalization mode"):
        MemmapDataset(data_dir, split="all", normalization="unknown")
```

- [ ] **Step 2: Run the dataset tests and verify the red state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py -v
```

Expected: collection succeeds, then tests fail because `MemmapDataset` does not accept `normalization` and `per_sample_zscore` is not a canonical key.

- [ ] **Step 3: Implement canonical names in `data/memmap_dataset.py`**

Replace the mapping and constructor field with:

```python
CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "per_sample_zscore": "csi_zscore.npy",
}
CSI_NORMALIZATIONS = tuple(CSI_FILES)
```

```python
def __init__(
    self,
    data_dir: str | Path,
    split: str = "train",
    envs: Iterable[str] | None = None,
    random_val_ratio: float = 0.1,
    random_test_ratio: float = 0.2,
    seed: int = 42,
    normalization: str = "global_minmax",
) -> None:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"split must be train/val/test/all, got {split}")
    self.split = split
    self.normalization = normalization

    data_dir = Path(data_dir)
    if normalization not in CSI_FILES:
        raise ValueError(
            f"Unknown normalization mode: {normalization}, "
            f"expected one of {list(CSI_FILES)}"
        )
    self._csi = np.load(str(data_dir / CSI_FILES[normalization]), mmap_mode="r")
```

Leave splitting, metadata, and item conversion unchanged.

- [ ] **Step 4: Run the dataset tests and verify the green state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py -v
```

Expected: all current tests in the new file pass.

- [ ] **Step 5: Add failing loader-propagation tests**

Append to `tests/test_input_normalization.py`:

```python
@pytest.mark.parametrize(
    "normalization",
    ["global_minmax", "global_zscore", "per_sample_zscore"],
)
def test_loader_factories_propagate_normalization(
    tmp_path: Path,
    normalization: str,
) -> None:
    from dataloader import (
        create_few_shot_data_loader,
        create_memmap_data_loader,
        create_memmap_data_loaders,
    )

    data_dir = tmp_path / "memmap"
    _write_normalization_dataset(data_dir)

    loader = create_memmap_data_loader(
        data_dir=data_dir,
        split="all",
        batch_size=1,
        normalization=normalization,
    )
    split_loaders = create_memmap_data_loaders(
        data_dir=data_dir,
        batch_size=1,
        normalization=normalization,
    )
    few_shot_loader, _ = create_few_shot_data_loader(
        data_dir=data_dir,
        target_envs=("env1",),
        few_shot_subjects=1,
        few_shot_frames=1,
        batch_size=1,
        normalization=normalization,
    )

    assert loader.dataset.normalization == normalization
    assert all(item.dataset.normalization == normalization for item in split_loaders.values())
    assert few_shot_loader.dataset.dataset.normalization == normalization
```

- [ ] **Step 6: Run the loader tests and verify the red state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py::test_loader_factories_propagate_normalization -v
```

Expected: FAIL because loader factories do not accept `normalization`.

- [ ] **Step 7: Propagate normalization through `dataloader.py`**

Add `normalization: str = "global_minmax"` to `create_memmap_data_loader`, `create_memmap_data_loaders`, and `create_few_shot_data_loader`. Pass it to every nested factory and dataset construction:

```python
dataset = MemmapDataset(
    data_dir=data_dir,
    split=split,
    envs=envs,
    seed=seed,
    normalization=normalization,
)
```

```python
normalization=normalization,
```

Use the second line in the split-factory call and the few-shot `MemmapDataset` construction. Do not change shuffle, worker, pin-memory, or split behavior.

- [ ] **Step 8: Run focused dataset and loader tests**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py tests/test_dataset_protocol.py tests/test_h36m17_contract.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit the dataset and loader layer**

```bash
git add data/memmap_dataset.py dataloader.py tests/test_input_normalization.py
git commit -m "feat: select CSI input normalization"
git push origin main
```

### Task 2: Training CLI and Checkpoint Metadata

**Files:**
- Modify: `train.py`
- Modify: `tests/test_input_normalization.py`

- [ ] **Step 1: Add failing CLI and checkpoint serialization tests**

Append these imports and tests to `tests/test_input_normalization.py`:

```python
from dataclasses import asdict
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
```

```python
@pytest.mark.parametrize(
    "normalization",
    ["global_minmax", "global_zscore", "per_sample_zscore"],
)
def test_train_cli_accepts_normalization(normalization: str) -> None:
    from train import parse_args

    argv = [
        "train.py",
        "--mode", "source_only",
        "--dataset-root", "data/mmfi_pose",
        "--source-envs", "env1",
        "--normalization", normalization,
    ]
    with patch("sys.argv", argv):
        args = parse_args()

    assert args.normalization == normalization


def test_train_config_defaults_to_global_minmax() -> None:
    from train import TrainConfig

    config = TrainConfig(dataset_root="data/mmfi_pose")

    assert config.normalization == "global_minmax"
    assert asdict(config)["normalization"] == "global_minmax"


def test_checkpoint_records_normalization(tmp_path: Path) -> None:
    from train import TrainConfig, save_checkpoint

    model = nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = StepLR(optimizer, step_size=1)
    checkpoint_path = tmp_path / "checkpoint.pth"
    config = TrainConfig(
        dataset_root="data/mmfi_pose",
        normalization="global_zscore",
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        epoch=1,
        best_metric=0.5,
        config=config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["train_config"]["normalization"] == "global_zscore"
```

- [ ] **Step 2: Run the training-interface tests and verify the red state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py::test_train_cli_accepts_normalization tests/test_input_normalization.py::test_train_config_defaults_to_global_minmax tests/test_input_normalization.py::test_checkpoint_records_normalization -v
```

Expected: FAIL because the parser and `TrainConfig` do not define `normalization`.

- [ ] **Step 3: Add the training configuration and CLI option**

Import the canonical choices:

```python
from data.memmap_dataset import CSI_NORMALIZATIONS
```

Add the field after `output_dir`:

```python
normalization: str = "global_minmax"
```

Add the parser argument after `--output-dir`:

```python
parser.add_argument(
    "--normalization",
    default="global_minmax",
    choices=CSI_NORMALIZATIONS,
    help="Precomputed CSI input normalization stored in the memmap dataset.",
)
```

The existing `asdict(config)` checkpoint path requires no extra serialization code.

- [ ] **Step 4: Pass `config.normalization` to every training loader**

Add this keyword to all `create_memmap_data_loader` calls in `_run_source_only` and `_run_finetune_align`:

```python
normalization=config.normalization,
```

Add the same keyword to all `create_few_shot_data_loader` calls in `_run_finetune` and `_run_finetune_align`. Do not alter model, optimizer, scheduler, losses, or splits.

- [ ] **Step 5: Run training and loader tests**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py tests/test_domain_alignment.py tests/test_joint_weighted_loss.py -v
```

Expected: PASS.

- [ ] **Step 6: Run a parser smoke check**

Run:

```powershell
conda activate WiFiPose
python train.py --help
```

Expected: help contains `--normalization {global_minmax,global_zscore,per_sample_zscore}`.

- [ ] **Step 7: Commit the training interface**

```bash
git add train.py tests/test_input_normalization.py
git commit -m "feat: expose input normalization training API"
git push origin main
```

### Task 3: Checkpoint-Driven Evaluation

**Files:**
- Modify: `eval.py`
- Modify: `tests/test_input_normalization.py`
- Modify: `tests/test_eval_diagnostics.py`

- [ ] **Step 1: Add failing checkpoint-resolution tests**

Append to `tests/test_input_normalization.py`:

```python
def test_resolve_checkpoint_normalization_uses_saved_value() -> None:
    from eval import resolve_checkpoint_normalization

    assert resolve_checkpoint_normalization(
        {"normalization": "per_sample_zscore"}
    ) == "per_sample_zscore"


def test_resolve_checkpoint_normalization_falls_back_for_old_checkpoint() -> None:
    from eval import resolve_checkpoint_normalization

    assert resolve_checkpoint_normalization({}) == "global_minmax"
```

- [ ] **Step 2: Run resolution tests and verify the red state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py::test_resolve_checkpoint_normalization_uses_saved_value tests/test_input_normalization.py::test_resolve_checkpoint_normalization_falls_back_for_old_checkpoint -v
```

Expected: FAIL because `resolve_checkpoint_normalization` does not exist.

- [ ] **Step 3: Split checkpoint loading without changing the compatibility wrapper**

In `eval.py`, implement:

```python
def resolve_checkpoint_normalization(train_config: Mapping[str, Any]) -> str:
    return str(train_config.get("normalization", "global_minmax"))


def load_checkpoint_model_and_config(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[WiFlowModel, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")

    train_config = checkpoint.get("train_config")
    if not isinstance(train_config, Mapping):
        raise KeyError(f"Checkpoint is missing train_config: {checkpoint_path}")

    model = WiFlowModel(
        input_channels=3,
        axial_mode=str(train_config.get("axial_mode", "spatial_then_temporal")),
        decoder_type=str(train_config.get("decoder_type", "joint")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, train_config


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> WiFlowModel:
    model, _ = load_checkpoint_model_and_config(checkpoint_path, device)
    return model
```

This preserves callers of `load_checkpoint_model` while avoiding a second `torch.load` in `main`.

- [ ] **Step 4: Add a failing evaluation-main propagation test**

Update both existing `FakeDataset` constructors in `tests/test_eval_diagnostics.py` to accept and record `normalization`. Replace the `load_checkpoint_model` monkeypatches with `load_checkpoint_model_and_config` returning a model and configuration. Add this focused test:

```python
def test_eval_main_uses_checkpoint_normalization_for_eval_and_pose_viz(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from evaluation import pose_viz

    seen: list[str] = []
    args = _minimal_eval_args(tmp_path, eval_split="test")
    args.pose_viz = True

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, split, envs=None, normalization="global_minmax"):
            seen.append(normalization)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            raise AssertionError("evaluation and visualization are mocked")

    monkeypatch.setattr(eval_module, "parse_args", lambda: args)
    monkeypatch.setattr(eval_module, "select_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        eval_module,
        "load_checkpoint_model_and_config",
        lambda checkpoint, device: (object(), {"normalization": "global_zscore"}),
    )
    monkeypatch.setattr(eval_module, "MemmapDataset", FakeDataset)
    monkeypatch.setattr(eval_module, "run_evaluation", lambda model, loader, device: {
        "overall": {"mpjpe": 1.0},
        "joint_rows": [],
        "action_rows": [],
        "environment_rows": [],
        "diagnostic": {"overall": {}, "joint_rows": []},
    })
    monkeypatch.setattr(eval_module, "_write_csv", lambda path, rows: None)
    monkeypatch.setattr(pose_viz, "run_pose_visualization", lambda **kwargs: None)

    eval_module.main()

    assert seen == ["global_zscore", "global_zscore"]
```

- [ ] **Step 5: Run the evaluation test and verify the red state**

Run:

```powershell
conda activate WiFiPose
pytest tests/test_eval_diagnostics.py::test_eval_main_uses_checkpoint_normalization_for_eval_and_pose_viz -v
```

Expected: FAIL because `main` does not load or pass checkpoint normalization.

- [ ] **Step 6: Use the checkpoint normalization in `eval.py`**

Change startup to:

```python
model, train_config = load_checkpoint_model_and_config(args.checkpoint, device)
normalization = resolve_checkpoint_normalization(train_config)
```

Pass the same value to the evaluation and pose-visualization dataset constructors:

```python
test_dataset = MemmapDataset(
    data_dir=args.dataset_root,
    split=eval_split,
    envs=eval_envs,
    normalization=normalization,
)
```

```python
viz_dataset = MemmapDataset(
    data_dir=args.dataset_root,
    split=eval_split,
    envs=eval_envs,
    normalization=normalization,
)
```

Feature visualization already consumes `test_loader`, so it automatically receives the checkpoint-selected CSI representation.

- [ ] **Step 7: Update existing evaluation fakes and run focused tests**

For the two pre-existing evaluation-main tests, use:

```python
monkeypatch.setattr(
    eval_module,
    "load_checkpoint_model_and_config",
    lambda checkpoint, device: (object(), {}),
)
```

and allow `FakeDataset.__init__(..., normalization="global_minmax")`.

Run:

```powershell
conda activate WiFiPose
pytest tests/test_input_normalization.py tests/test_eval_diagnostics.py tests/test_feature_viz_layout.py -v
```

Expected: PASS, including the old-checkpoint fallback.

- [ ] **Step 8: Commit checkpoint-driven evaluation**

```bash
git add eval.py tests/test_input_normalization.py tests/test_eval_diagnostics.py
git commit -m "feat: restore normalization during evaluation"
git push origin main
```

### Task 4: Workflow Documentation and Commands

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Document the normalization contract in `README.md`**

Add a section after the source-domain split protocol:

```markdown
## CSI Input Normalization Ablation

Training exposes three precomputed CSI representations through `--normalization`:

- `global_minmax` loads `csi_gminmax.npy` and remains the default.
- `global_zscore` loads `csi_gzscore.npy`.
- `per_sample_zscore` loads `csi_zscore.npy`.

The selected value is saved in checkpoint `train_config`. Evaluation restores it automatically; do not manually substitute CSI files. Checkpoints created before this option default to `global_minmax`.
```

- [ ] **Step 2: Add the three controlled source-only commands to `README.md`**

Add these commands as separate one-line code blocks:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization global_minmax --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_global_minmax
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization global_zscore --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_global_zscore
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization per_sample_zscore --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_per_sample_zscore
```

- [ ] **Step 3: Update `AGENTS.md`**

Replace the four affected project-structure bullets with these exact descriptions:

```markdown
- `dataloader.py`: Core module for loading NPY memmap datasets, propagating canonical CSI input normalization, creating PyTorch `DataLoader` instances with `memmap_collate_fn`, and providing `create_memmap_data_loader` / `create_memmap_data_loaders` / `create_few_shot_data_loader` factory functions.
- `data/memmap_dataset.py`: NPY memmap dataset reader that selects `global_minmax`, `global_zscore`, or `per_sample_zscore` CSI amplitude, loads Human3.6M-17 keypoints and metadata, and provides zero-copy OS-cached I/O.
- `train.py`: Root-level training entrypoint for WiFlow pose regression, including input-normalization selection, losses, metrics, optimizer, scheduler, checkpointing, and CSV logging. The selected normalization is stored in checkpoint `train_config`.
- `eval.py`: Root-level evaluation entrypoint for loading checkpoints, restoring their CSI input normalization, computing test metrics, and optionally generating research-grade feature visualizations via `--feature-viz`.
```

Add this paragraph immediately after the default training configuration description:

```markdown
Supported `--normalization` values are `global_minmax`, `global_zscore`, and `per_sample_zscore`. The default is `global_minmax`. Training passes the selected representation to every loader and stores it in checkpoint `train_config`; evaluation restores it automatically, with `global_minmax` as the compatibility fallback for older checkpoints.
```

- [ ] **Step 4: Check documentation consistency**

Run:

```powershell
rg -n "normalization|global_minmax|global_zscore|per_sample_zscore" README.md AGENTS.md data/memmap_dataset.py train.py eval.py
```

Expected: all public names use `per_sample_zscore`; `csi_zscore.npy` appears only as its backing filename; docs do not tell users to pass a normalization option to `eval.py`.

- [ ] **Step 5: Commit workflow documentation**

```bash
git add README.md AGENTS.md
git commit -m "docs: document normalization ablation workflow"
git push origin main
```

### Task 5: Full Verification and Server Handoff

**Files:**
- Verify only; no new implementation files.

- [ ] **Step 1: Run the full test suite in the project environment**

Run:

```powershell
conda activate WiFiPose
pytest
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Run CLI rejection and compatibility smoke checks**

Run:

```powershell
conda activate WiFiPose
python train.py --mode source_only --dataset-root data/mmfi_pose --source-envs env1 --normalization invalid
```

Expected: argparse exits with code 2 and lists the three accepted values.

Run:

```powershell
conda activate WiFiPose
python eval.py --help
```

Expected: evaluation help contains no separate normalization override.

- [ ] **Step 3: Verify repository state and remote parity**

Run:

```bash
git diff --check
git status --short --branch
git rev-list --left-right --count origin/main...main
```

Expected: no tracked changes, only previously known local pytest directories if still present, and `0 0` remote parity.

- [ ] **Step 4: Hand off evaluation commands**

After each training run, evaluate its checkpoint with the same fixed test split. The checkpoint supplies normalization automatically:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --checkpoint runs/source_env1_global_minmax/best_val_pck_0_2.pth --eval-envs env1 --eval-split test --device cuda --batch-size 64 --num-workers 8 --output-dir outputs/source_env1_global_minmax_test
```

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --checkpoint runs/source_env1_global_zscore/best_val_pck_0_2.pth --eval-envs env1 --eval-split test --device cuda --batch-size 64 --num-workers 8 --output-dir outputs/source_env1_global_zscore_test
```

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --checkpoint runs/source_env1_per_sample_zscore/best_val_pck_0_2.pth --eval-envs env1 --eval-split test --device cuda --batch-size 64 --num-workers 8 --output-dir outputs/source_env1_per_sample_zscore_test
```

Compare the three result directories using overall MPJPE/PCK, per-action/per-joint CSVs, and collapse diagnostics. Do not interpret a one-seed difference as a final mechanism claim.
