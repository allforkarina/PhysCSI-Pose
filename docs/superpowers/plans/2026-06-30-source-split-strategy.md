# Source-Domain Split Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic `subject` and `frame_random` source-domain split strategies so the same H36M-17 model can diagnose the effect of same-subject frame leakage.

**Architecture:** `MemmapDataset` owns split membership and exposes a shared strategy constant. Loader factories propagate the strategy, while `train.py` and `eval.py` expose an explicit CLI option and store the training choice through `TrainConfig`. The existing subject-disjoint protocol remains the default.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, argparse, pytest.

---

## File Map

- Modify `data/memmap_dataset.py`: validate strategies and implement deterministic per-subject frame splitting.
- Modify `dataloader.py`: propagate the strategy through loader factories.
- Modify `train.py`: add the training configuration/CLI and select subject sets only for the subject strategy.
- Modify `eval.py`: add the evaluation CLI and apply the strategy to metric and visualization datasets.
- Modify `tests/test_dataset_protocol.py`: verify frame membership, ratios, determinism, and environment filtering.
- Create `tests/test_split_strategy_cli.py`: verify train/eval parser contracts and configuration defaults.
- Modify `tests/test_eval_diagnostics.py`: verify evaluation forwards the strategy.
- Modify `AGENTS.md`: document commands and the diagnostic-only interpretation.

### Task 1: Dataset frame-random split

**Files:**
- Modify: `tests/test_dataset_protocol.py`
- Modify: `data/memmap_dataset.py`

- [ ] **Step 1: Write failing frame-random membership tests**

Add tests using ten frames per subject so rounded counts are exact:

```python
def test_frame_random_split_is_disjoint_complete_and_70_10_20_per_subject(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    subjects = ("S01", "S02", "S03")
    _write_subject_env_memmap_dataset(
        data_dir,
        envs=("env1",),
        subjects=subjects,
        frames_per_subject=10,
    )
    datasets = {
        split: MemmapDataset(
            data_dir,
            split=split,
            envs=("env1",),
            split_strategy="frame_random",
            seed=123,
        )
        for split in ("train", "val", "test")
    }

    split_indices = {name: set(dataset.indices.tolist()) for name, dataset in datasets.items()}
    assert split_indices["train"].isdisjoint(split_indices["val"])
    assert split_indices["train"].isdisjoint(split_indices["test"])
    assert split_indices["val"].isdisjoint(split_indices["test"])
    assert set().union(*split_indices.values()) == set(range(30))

    for dataset in datasets.values():
        assert {str(dataset._samples[idx]) for idx in dataset.indices} == set(subjects)
    assert len(datasets["train"]) == 21
    assert len(datasets["val"]) == 3
    assert len(datasets["test"]) == 6


def test_frame_random_split_is_seeded_and_filters_environment_first(tmp_path: Path) -> None:
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_subject_env_memmap_dataset(
        data_dir,
        envs=("env1", "env2"),
        subjects=("S01", "S02"),
        frames_per_subject=10,
    )
    first = MemmapDataset(data_dir, split="train", envs=("env1",), split_strategy="frame_random", seed=7)
    repeated = MemmapDataset(data_dir, split="train", envs=("env1",), split_strategy="frame_random", seed=7)
    changed = MemmapDataset(data_dir, split="train", envs=("env1",), split_strategy="frame_random", seed=8)

    assert np.array_equal(first.indices, repeated.indices)
    assert not np.array_equal(first.indices, changed.indices)
    assert {str(first._envs[idx]) for idx in first.indices} == {"env1"}


def test_memmap_dataset_rejects_unknown_split_strategy(tmp_path: Path) -> None:
    import pytest
    from data.memmap_dataset import MemmapDataset

    data_dir = tmp_path / "memmap"
    _write_memmap_dataset(data_dir)
    with pytest.raises(ValueError, match="split_strategy"):
        MemmapDataset(data_dir, split_strategy="unknown")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run in the established environment:

```powershell
conda activate WiFiPose; pytest tests/test_dataset_protocol.py -k frame_random -v
```

Expected: failures reporting that `MemmapDataset.__init__()` does not accept `split_strategy`.

- [ ] **Step 3: Implement the minimal dataset strategy**

In `data/memmap_dataset.py`, import `random`, define:

```python
SPLIT_STRATEGIES = ("subject", "frame_random")
```

Add `split_strategy: str = "subject"` to `__init__`, reject values outside the constant, store it, and pass it into `_build_split`. In `_build_split`, keep the current subject branch unchanged and add this frame branch after candidate filtering:

```python
if split == "all":
    return np.asarray(sorted(candidate_indices), dtype=np.int64)

grouped: dict[str, list[int]] = {}
for idx in candidate_indices:
    grouped.setdefault(sample_list[idx], []).append(idx)

if split_strategy == "frame_random":
    rng = random.Random(seed)
    split_indices = {"train": [], "val": [], "test": []}
    for subject in sorted(grouped):
        shuffled = grouped[subject][:]
        rng.shuffle(shuffled)
        count = len(shuffled)
        test_count = min(count, int(round(count * random_test_ratio)))
        val_count = min(count - test_count, int(round(count * random_val_ratio)))
        train_count = count - val_count - test_count
        split_indices["train"].extend(shuffled[:train_count])
        split_indices["val"].extend(shuffled[train_count:train_count + val_count])
        split_indices["test"].extend(shuffled[train_count + val_count:])
    return np.asarray(sorted(split_indices[split]), dtype=np.int64)
```

Explicit subject filters apply only to `subject`; frame-random uses every environment-filtered subject.

- [ ] **Step 4: Run dataset tests and verify GREEN**

```powershell
conda activate WiFiPose; pytest tests/test_dataset_protocol.py -v
```

Expected: all dataset protocol tests pass.

- [ ] **Step 5: Commit the dataset behavior**

```powershell
git add data/memmap_dataset.py tests/test_dataset_protocol.py
git commit -m "feat: add random frame split strategy"
git push origin main
```

### Task 2: Loader and source-only training integration

**Files:**
- Create: `tests/test_split_strategy_cli.py`
- Modify: `dataloader.py`
- Modify: `train.py`

- [ ] **Step 1: Write failing loader and training contract tests**

Create `tests/test_split_strategy_cli.py` with parser tests using `monkeypatch`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from train import TrainConfig


def test_train_config_defaults_to_subject_split() -> None:
    assert TrainConfig(dataset_root="data").split_strategy == "subject"


@pytest.mark.parametrize("strategy", ("subject", "frame_random"))
def test_train_parser_accepts_split_strategy(monkeypatch, strategy: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--mode", "source_only", "--dataset-root", "data", "--split-strategy", strategy],
    )
    assert train.parse_args().split_strategy == strategy


def test_train_parser_rejects_unknown_split_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--mode", "source_only", "--dataset-root", "data", "--split-strategy", "unknown"],
    )
    with pytest.raises(SystemExit):
        train.parse_args()
```

Add a loader propagation test to `tests/test_dataset_protocol.py` that constructs `create_memmap_data_loader(..., split_strategy="frame_random")` and asserts `loader.dataset.split_strategy == "frame_random"`.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
conda activate WiFiPose; pytest tests/test_split_strategy_cli.py tests/test_dataset_protocol.py -k "split_strategy or loader" -v
```

Expected: missing `TrainConfig.split_strategy`, parser option, or loader keyword failures.

- [ ] **Step 3: Propagate the strategy through loaders**

Import `SPLIT_STRATEGIES` only where choices are needed. Add `split_strategy: str = "subject"` to `create_memmap_data_loader` and `create_memmap_data_loaders`, pass it to `MemmapDataset`, and forward it from the multi-loader factory.

- [ ] **Step 4: Integrate source-only training**

In `train.py`:

```python
from data.memmap_dataset import SPLIT_STRATEGIES
```

Add `split_strategy: str = "subject"` to `TrainConfig` and the parser option:

```python
parser.add_argument(
    "--split-strategy",
    default="subject",
    choices=SPLIT_STRATEGIES,
    help="Source split protocol: subject-disjoint or diagnostic random frames within each subject.",
)
```

In `_run_source_only`, always call `_resolve_source_subject_splits` to retain the single-environment/ten-subject validation, but pass explicit subject tuples only when `config.split_strategy == "subject"`. Always pass `split_strategy=config.split_strategy` to loaders and print either the subject assignment or frame counts.

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
conda activate WiFiPose; pytest tests/test_split_strategy_cli.py tests/test_dataset_protocol.py -v
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit and push training integration**

```powershell
git add dataloader.py train.py tests/test_dataset_protocol.py tests/test_split_strategy_cli.py
git commit -m "feat: expose source split strategy in training"
git push origin main
```

### Task 3: Evaluation integration

**Files:**
- Modify: `tests/test_split_strategy_cli.py`
- Modify: `tests/test_eval_diagnostics.py`
- Modify: `eval.py`

- [ ] **Step 1: Write failing evaluation parser and forwarding tests**

Extend `tests/test_split_strategy_cli.py` with equivalent `eval.parse_args()` acceptance and rejection tests. Add `split_strategy="frame_random"` to `_minimal_eval_args` in `tests/test_eval_diagnostics.py`, update `FakeDataset.__init__` to accept it, record it in `seen`, and assert:

```python
assert seen["split_strategy"] == "frame_random"
```

- [ ] **Step 2: Run evaluation tests and verify RED**

```powershell
conda activate WiFiPose; pytest tests/test_split_strategy_cli.py tests/test_eval_diagnostics.py -v
```

Expected: eval parser lacks the option and `MemmapDataset` is called without the expected keyword.

- [ ] **Step 3: Add evaluation strategy support**

Import `SPLIT_STRATEGIES`, add the same parser option with default `subject`, and pass `split_strategy=args.split_strategy` into both the metric dataset and the optional pose visualization dataset.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

```powershell
conda activate WiFiPose; pytest tests/test_split_strategy_cli.py tests/test_eval_diagnostics.py -v
```

Expected: all evaluation contract tests pass.

- [ ] **Step 5: Commit and push evaluation integration**

```powershell
git add eval.py tests/test_split_strategy_cli.py tests/test_eval_diagnostics.py
git commit -m "feat: evaluate explicit source split strategies"
git push origin main
```

### Task 4: Documentation and full verification

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Document both protocols and commands**

Update the source-only section to state that `subject` remains the default 7/1/2 unseen-subject protocol. Add one-line frame-random training and test commands using separate `runs/` and `outputs/` directories, and state that frame-random metrics measure same-subject interpolation and are diagnostic only.

- [ ] **Step 2: Run CLI smoke checks**

```powershell
conda activate WiFiPose; python train.py --help; python eval.py --help
```

Expected: both help outputs list `--split-strategy {subject,frame_random}`.

- [ ] **Step 3: Run the full test suite**

```powershell
conda activate WiFiPose; pytest
```

Expected: zero failures.

- [ ] **Step 4: Check the final diff**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only the planned documentation change remains unstaged after prior task commits.

- [ ] **Step 5: Commit and push documentation**

```powershell
git add AGENTS.md
git commit -m "docs: document frame split diagnostic"
git push origin main
```

- [ ] **Step 6: Report exact server commands**

Provide matched one-line Bash commands for subject and frame-random training plus train/validation/test evaluation, with training artifacts under `runs/` and evaluation artifacts under `outputs/`.
