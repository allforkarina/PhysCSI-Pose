# REASONIX.md — WiFlow Pose Estimation

## Stack
- Python 3.10+, PyTorch (torch, torch.nn, torch.optim)
- NumPy for array ops, NPY memmap dataset I/O
- Human3.6M-17 skeleton: 17 keypoints, raw xy GT coordinates
- Conda env `WiFiPose` (torch, numpy, scipy, h5py, tqdm, pytest)
- No package manifest — scripts run directly from repo root

## Layout
- `train.py` — training entrypoint (source-only + cross-domain few-shot finetune)
- `eval.py` — evaluation entrypoint (metrics, CSV, pose/feature viz)
- `dataloader.py` — NPY memmap DataLoader factories and `memmap_collate_fn`
- `data/` — `memmap_dataset.py` (Dataset), `heatmap_gt.py` (coordinate utils)
- `models/` — `WiFlowModel`, spatial encoder, axial encoder, joint/hierarchical decoders, skeleton
- `evaluation/` — forward hooks, feature viz, pose viz
- `scripts/` — `build_memmap.py`, `build_groundtruth.py`, `visualize_gt.py`
- `runs/` — training artifacts (gitignored)
- `outputs/` — evaluation artifacts (gitignored)
- `pose_targets.py` — reserved, currently empty
- `docs/` — planning docs and specs

## Commands
```powershell
# Build memmap dataset
python scripts\build_memmap.py --src D:\path\to\raw\dataset --dst data\mmfi_pose --gt-dir D:\path\to\ground_truth_npy --workers 4

# Train
python train.py --mode source_only --dataset-root data\mmfi_pose --source-envs env1 --epochs 50 --batch-size 64 --output-dir runs\source_env1
# Evaluate
python eval.py --dataset-root data\mmfi_pose --checkpoint runs\source_env1\best_val_mpjpe.pth --eval-envs env1 --eval-split test --output-dir outputs\source_env1_test
# eval.py defaults to --eval-split test; use --eval-split all only for explicit full-subset evaluation.
# Source-only split protocol: require exactly one source env, then split every subject's frames 70/10/20 into train/val/test.

# Few-shot cross-domain
python train.py --mode source_only --dataset-root data\mmfi_pose --source-envs env1 --output-dir outputs\source --epochs 50
python train.py --mode finetune --dataset-root data\mmfi_pose --target-envs env2 --output-dir outputs\finetune --finetune-from outputs\source\best_val_mpjpe.pth --few-shot-subjects 4 --few-shot-frames 5 --epochs 30
python eval.py --dataset-root data\mmfi_pose --checkpoint outputs\finetune\best_train_loss.pth --eval-envs env2 --eval-split all --output-dir outputs\ft_eval --exclude-indices outputs\finetune\few_shot_train_indices.npy

# Tests
pytest
```

## Conventions
- `from __future__ import annotations` at top of every module
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- Type hints on all signatures; `pathlib.Path` for paths
- Imports: stdlib → third-party → local; 4-space indent
- Public API via `__all__` in package `__init__.py`

## Constraints
- After every project update, commit and push changes to GitHub.
- All code modifications must follow the `karpathy-guidelines` skill.

## Watch out for
- No installable package (`pyproject.toml`/`setup.py`) — run scripts from repo root.
- `tests/` is gitignored and does not currently exist; the CLAUDE.md references test files that need creating.
- CSI shape: raw memmap `[N,64,3,114]` → collate permutes to `[B,3,114,64]` (antenna, subcarrier, time). Amplitude only; phase discarded.
- `.npy`/`.npz`/`.pt`/`.pth` artifacts are gitignored.
- `--axial-mode`: `spatial_then_temporal`, `temporal_then_spatial`, `parallel_sum`, `parallel_concat`. `--decoder-type`: `joint`, `hierarchical`.
