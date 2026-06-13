# REASONIX.md — PhysCSI-Pose

## Stack
- **Python 3** — no build system (`pyproject.toml` / `setup.py` absent); dependencies in `requirements.txt`
- **PyTorch** — tensor compute + feature extraction (amplitude-only, CPU or CUDA)
- **NumPy** — memmap-backed `.npy` arrays for cached datasets, ground-truth labels
- **SciPy** — `.mat` file loading (`scipy.io.loadmat`) for raw CSI data
- **pytest** — test runner; config in `pytest.ini`
- **PyYAML** — pipeline config files under `configs/`

## Layout
- `dataset/` — core package: features, labels, memmap_dataset, meta, splits
- `models/` — model components: `AmpFeatureMixEncoder`, `PoseAwareTokenProjection`, `TemporalLiteTransformer`
- `scripts/` — CLI tools: `build_memmap.py` (main pipeline), `inspect_memmap.py`, `scan_gt_stats.py`
- `configs/` — YAML configs consumed by `build_memmap.py`
- `tests/` — pytest suite; one `test_<module>.py` per source module
- `docs/` — plans and specs under `docs/superpowers/`

## Commands
```bash
# Run all tests
pytest

# Build memmap dataset cache (data lives outside Git)
python scripts/build_memmap.py \
  --config configs/build_memmap.yaml \
  --csi-root /path/to/csi_root \
  --gt-root /path/to/gt_root \
  --output-root /data/WiFiPose/dataset/memmap

# Inspect a built cache
python scripts/inspect_memmap.py --data-root /path/to/memmap

# Scan ground-truth stats before build
python scripts/scan_gt_stats.py --config configs/build_memmap.yaml --gt-root /path/to/gt_root
```
No lint, format, or typecheck tooling configured.

## Git Workflow
- **Remote:** `git@github.com:allforkarina/PhysCSI-Pose.git` — fixed, do not change.
- **Commit + push after every completed change.** Inspect staged files first.
- **Push gate:** push only source scripts, tests, configs, docs. Never push data (`.npy`, `.npz`, `.mat`, `.pkl`), checkpoints (`.pt`, `.pth`, `.ckpt`), experiment outputs, logs, caches, venvs, or credentials. The `.gitignore` covers all prohibited paths.
- **Do not run inference** with real CSI data unless the user explicitly provides it.

## Conventions
- `from __future__ import annotations` in every `.py` file
- Imports: stdlib → third-party → project (`from dataset.…`), blank-line separated
- Typed signatures throughout; void functions return `-> None`
- `@dataclass(frozen=True)` for data structures (e.g. `FeatureComponents`)
- Test files: `tests/test_<module>.py`; plain `def test_*` functions, no classes
- Tests use synthetic tensors / `tmp_path` fixtures — no real CSI data in tests
- Scripts append project root to `sys.path` (no package install)

## Watch out for
- **No package install.** `scripts/build_memmap.py` hacks `sys.path`; importing from `dataset/` works only from repo root or that hack.
- **Only `source_only` protocol works.** `MemmapPoseDataset(protocol="finetune", …)` raises `NotImplementedError`.
- **CSI repair at build time.** NaN / inf / negative amplitudes are median-filled; counts saved in `meta_build.json`.
- **Training loop is not implemented yet.** Encoder, token projection, and temporal transformer exist but no training/inference pipeline.
