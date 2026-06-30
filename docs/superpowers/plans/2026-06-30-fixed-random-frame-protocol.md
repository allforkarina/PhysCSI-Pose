# Fixed Random-Frame Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the split-strategy API and make deterministic per-subject 70/10/20 random frame splitting the only project train/validation/test protocol.

**Architecture:** `MemmapDataset` becomes the single owner of a fixed frame split and no longer accepts explicit subject sets or a strategy. Loader, training, and evaluation APIs become narrower by removing obsolete parameters. Runtime documentation describes only the fixed protocol.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, argparse, pytest.

---

## File Map

- Modify `data/memmap_dataset.py`: remove strategy and subject branches; retain only frame splitting.
- Modify `dataloader.py`: remove strategy and explicit subject-set parameters.
- Modify `train.py`: remove strategy configuration/CLI and replace subject resolver with one-environment validation.
- Modify `eval.py`: remove strategy CLI and constructor forwarding.
- Modify `tests/test_dataset_protocol.py`: test the fixed protocol directly.
- Modify `tests/test_domain_alignment.py`: replace 7/1/2 resolver tests with source-environment validation tests.
- Delete `tests/test_split_strategy_cli.py`: the configurable API no longer exists.
- Modify `tests/test_eval_diagnostics.py`: restore the fixed dataset constructor contract.
- Modify `AGENTS.md`, `README.md`, and `REASONIX.md`: document only random frame splitting.

### Task 1: Fix the dataset and loader protocol

**Files:**
- Modify: `tests/test_dataset_protocol.py`
- Modify: `data/memmap_dataset.py`
- Modify: `dataloader.py`

- [ ] **Step 1: Rewrite tests for the desired fixed API**

Remove subject-disjoint and strategy-specific tests. Keep one protocol test that constructs `MemmapDataset` without a strategy and asserts exact 70/10/20 counts, disjointness, completeness, all-subject presence, deterministic seeds, environment filtering, and unpartitioned `all` membership. Add signature assertions:

```python
import inspect


def test_dataset_and_loader_expose_only_fixed_frame_split_api() -> None:
    from data.memmap_dataset import MemmapDataset
    from dataloader import create_memmap_data_loader

    dataset_params = inspect.signature(MemmapDataset).parameters
    loader_params = inspect.signature(create_memmap_data_loader).parameters
    for removed in ("split_strategy", "train_subjects", "val_subjects", "test_subjects"):
        assert removed not in dataset_params
        assert removed not in loader_params
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe -m pytest tests/test_dataset_protocol.py -v --basetemp .pytest_fixed_task1_red
```

Expected: fixed-protocol membership uses the existing default subject split and signature checks find obsolete parameters.

- [ ] **Step 3: Remove dataset strategy and subject branches**

In `data/memmap_dataset.py`:

- Delete `SPLIT_STRATEGIES`.
- Delete `train_subjects`, `val_subjects`, `test_subjects`, and `split_strategy` from `__init__` and `_build_split`.
- Delete strategy validation and instance storage.
- After environment filtering, return all candidates for `split=all`; otherwise group by subject and always execute the current deterministic frame-random branch.
- Keep `random_val_ratio=0.1`, `random_test_ratio=0.2`, and `seed=42` arguments.

The retained partition is:

```python
rng = random.Random(seed)
split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
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

- [ ] **Step 4: Narrow loader factories**

Remove explicit subject-set and strategy parameters from both loader factories and stop forwarding them into `MemmapDataset`.

- [ ] **Step 5: Run dataset tests and verify GREEN**

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe -m pytest tests/test_dataset_protocol.py -v --basetemp .pytest_fixed_task1_green
```

Expected: all dataset tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add data/memmap_dataset.py dataloader.py tests/test_dataset_protocol.py
git commit -m "refactor: fix random frame dataset protocol"
git push origin main
```

### Task 2: Remove training and evaluation strategy APIs

**Files:**
- Modify: `tests/test_domain_alignment.py`
- Delete: `tests/test_split_strategy_cli.py`
- Modify: `tests/test_eval_diagnostics.py`
- Modify: `train.py`
- Modify: `eval.py`

- [ ] **Step 1: Write failing training and evaluation API tests**

Replace subject resolver tests with:

```python
def test_validate_source_envs_requires_exactly_one_environment() -> None:
    from train import _validate_source_envs

    with pytest.raises(ValueError, match="exactly one"):
        _validate_source_envs(None)
    with pytest.raises(ValueError, match="exactly one"):
        _validate_source_envs(("env1", "env2"))
    assert _validate_source_envs(("env1",)) == ("env1",)
```

Add parser rejection tests to an existing test module:

```python
def test_train_and_eval_reject_removed_split_strategy(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--mode", "source_only", "--dataset-root", "data", "--split-strategy", "frame_random"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["eval.py", "--dataset-root", "data", "--checkpoint", "model.pth", "--split-strategy", "frame_random"])
    with pytest.raises(SystemExit):
        eval_module.parse_args()
```

Update evaluation fake datasets to accept only `(data_dir, split, envs=None)` and remove strategy assertions.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe -m pytest tests/test_domain_alignment.py tests/test_eval_diagnostics.py -v --basetemp .pytest_fixed_task2_red
```

Expected: `_validate_source_envs` is missing and evaluation still forwards `split_strategy`.

- [ ] **Step 3: Simplify source-only training**

In `train.py`:

- Remove the `SPLIT_STRATEGIES` import.
- Remove `TrainConfig.split_strategy`.
- Replace `_resolve_source_subject_splits` with:

```python
def _validate_source_envs(
    source_envs: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if source_envs is None or len(source_envs) != 1:
        raise ValueError(
            "source_only mode requires exactly one source environment, e.g. --source-envs env1"
        )
    return source_envs
```

- In `_run_source_only`, validate the environment, build all three loaders without subject/strategy arguments, and print train/val/test frame counts.
- Remove the `--split-strategy` parser argument.

- [ ] **Step 4: Simplify evaluation**

In `eval.py`, import only `MemmapDataset`, remove the parser argument, and remove `split_strategy=` from metric and pose visualization dataset construction.

- [ ] **Step 5: Delete obsolete strategy tests and run focused tests**

Delete `tests/test_split_strategy_cli.py`, then run:

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe -m pytest tests/test_domain_alignment.py tests/test_eval_diagnostics.py tests/test_dataset_protocol.py -v --basetemp .pytest_fixed_task2_green
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add train.py eval.py tests/test_domain_alignment.py tests/test_eval_diagnostics.py tests/test_split_strategy_cli.py
git commit -m "refactor: remove source split strategy API"
git push origin main
```

### Task 3: Runtime documentation and final verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `REASONIX.md`

- [ ] **Step 1: Replace runtime protocol documentation**

Document that every subject in the selected environment contributes deterministic random frames to train/validation/test at 70/10/20. Remove all `--split-strategy`, 7/1/2 subject, unseen-subject, and comparison language from runtime documents. Keep commands one line each, training outputs under `runs/`, and evaluation outputs under `outputs/`.

- [ ] **Step 2: Search runtime files for removed APIs**

```powershell
rg -n "split_strategy|split-strategy|SPLIT_STRATEGIES|train_subjects|val_subjects|test_subjects|_resolve_source_subject_splits" data dataloader.py train.py eval.py tests AGENTS.md README.md REASONIX.md
```

Expected: no matches.

- [ ] **Step 3: Run CLI smoke checks**

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe train.py --help
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe eval.py --help
```

Expected: neither command lists `--split-strategy`.

- [ ] **Step 4: Run the full suite**

```powershell
D:\SoftWare\Anaconda\envs\WiFiPose\python.exe -m pytest --basetemp .pytest_fixed_verify
```

Expected: zero failures.

- [ ] **Step 5: Verify repository state**

```powershell
git diff --check
git status --short
```

Expected: only the three planned runtime documentation files remain uncommitted after earlier task commits.

- [ ] **Step 6: Commit and push documentation**

```powershell
git add AGENTS.md README.md REASONIX.md
git commit -m "docs: make random frame splitting the project protocol"
git push origin main
```

- [ ] **Step 7: Report updated Linux commands**

Provide one-line Bash commands without `--split-strategy` for source-only training and train/validation/test evaluation. State that existing runs remain usable and old checkpoint extra configuration keys are ignored during model reconstruction.
