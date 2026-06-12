# Data Layer Design

## Goal

Build the first data layer for PhysCSI-Pose: construct amplitude-only WiFi CSI pose features once, cache them as large `.npy` memmap arrays, and provide a Dataset interface that reads the cached arrays by experiment split.

This phase does not implement model networks, training loops, inference, or metric evaluation.

## Repository Structure

The implementation should create these repository paths at the project root. Do not create an extra nested `WiFiPose/` directory.

```text
scripts/
  scan_gt_stats.py
  build_memmap.py
  inspect_memmap.py
dataset/
  __init__.py
  features.py
  labels.py
  meta.py
  splits.py
  memmap_dataset.py
tests/
  test_features.py
  test_labels.py
  test_meta.py
  test_dataset.py
configs/
  build_memmap.yaml
requirements.txt
README.md
```

## Data Contracts

Raw CSI frame files are MATLAB `.mat` files. The expected raw `CSIamp` shape is:

```text
raw CSIamp frame: [3, 114, 10]
```

The build script transposes each raw frame into the standard internal format:

```text
standard CSI frame after transpose: [10, 3, 114]
CSI sequence: [297, 10, 3, 114]
Feature sequence: [297, 12, 10, 114]
```

Ground-truth pose files are `.npy` files under `--gt-root`, one file per environment-subject-action sequence:

```text
E{env_id:02d}_S{subject_id:02d}_A{action_id:02d}.npy
```

Each GT sequence has shape:

```text
GT sequence: [297, 17, 3]
[..., 0] = x
[..., 1] = y
[..., 2] = confidence
```

## Feature Definitions

All feature construction uses `torch.Tensor` and should preserve the input device during computation. The build script may choose `cpu` or `cuda`, then writes CPU NumPy arrays into `.npy` memmaps.

For a standardized CSI sequence:

```text
CSIamp_seq: [F, T, R, S] = [297, 10, 3, 114]
```

Compute:

```text
L = log(CSIamp + eps_log)
bg[r,s] = median over F,T of L[:, :, r, s]
mad[r,s] = median over F,T of abs(L[:, :, r, s] - bg[r,s])
L_norm = (L - bg) / (mad + eps_mad)
```

Use:

```text
eps_log = 1.0e-6
eps_mad = 1.0e-6
```

Derived features:

```text
D_center[f,t,r,s] = L_norm[f,t,r,s] - mean_t(L_norm[f,:,r,s])
C_ant[f,t,r,s] = L_norm[f,t,r,s] - mean_r(L_norm[f,t,:,s])
```

Subcarrier contrast:

```text
k = 15
padding = 7
stride = 1
padding mode = reflect
F_sub = L_norm - AvgPool1d_subcarrier(L_norm, kernel=15, stride=1, reflect_padding=7)
```

`F_sub` must keep the subcarrier length at 114.

Feature channel order is fixed:

```text
channel 0..2   = L_norm   rx0, rx1, rx2
channel 3..5   = D_center rx0, rx1, rx2
channel 6..8   = F_sub    rx0, rx1, rx2
channel 9..11  = C_ant    rx0, rx1, rx2
```

Output:

```text
X_seq: [297, 12, 10, 114]
X_frame: [12, 10, 114]
batch: [B, 12, 10, 114]
```

No extra global standardization or clipping is applied to `X`.

## Label Processing

Label processing happens once during data build. Training-time Dataset reads already processed labels.

Input:

```text
GT_seq: [297, 17, 3]
xy = GT_seq[..., :2]
conf = GT_seq[..., 2]
```

Processing:

- NaN or inf xy points become `(0.0, 0.0)` and `conf = 0`.
- All-zero xy points become `(0.0, 0.0)` and `conf = 0`.
- NaN or inf confidence becomes `0`.
- Confidence is clipped to `[0.0, 1.0]`.
- Coordinate format is detected once at build time from global GT statistics.
- Pixel coordinates use `x / 1920`, `y / 1080`, then map `[0,1]` to `[-0.8,0.8]`.
- Unit-normalized coordinates `[0,1]` map directly to `[-0.8,0.8]`.
- Target-normalized coordinates near `[-0.8,0.8]` are not rescaled.
- Final xy values are clipped to `[-0.8,0.8]`.

Output:

```text
Y_seq: [297, 17, 2]
Conf_seq: [297, 17]
```

The build writes GT statistics and detected coordinate format into `meta_build.json`.

## Full Dataset Cache

Build the full neutral dataset once for all four environments:

```text
N = 4 env * 10 subject/env * 27 action * 297 frame = 320760
```

Generated files live outside Git, for example:

```text
/data/WiFiPose/dataset/memmap/
  X_all.npy
  Y_all.npy
  Conf_all.npy
  meta.npz
  meta_build.json
```

Use `np.lib.format.open_memmap` for writing `.npy` files incrementally. Training reads them with:

```python
np.load(path, mmap_mode="r")
```

The build script must reject existing complete output files unless `--overwrite` is passed.

## Metadata

`meta.npz` stores neutral factual fields only:

```text
global_idx  int64  [N]
env_id      uint8  [N]   # 1..4
subject_id  uint8  [N]   # 1..40
action_id   uint8  [N]   # 1..27
frame_id    uint16 [N]   # 0..296
seq_id      uint16 [N]   # 0..1079
```

It does not store split IDs or domain-role IDs. Experiment protocols are generated at read time.

Fixed traversal and index mapping:

```text
S01 -> S40
  A01 -> A27
    frame001 -> frame297
```

```python
env_id = (subject_id - 1) // 10 + 1
seq_id = ((subject_id - 1) * 27) + (action_id - 1)
global_idx = seq_id * 297 + frame_id
```

Reference points:

```text
global_idx 0      -> S01 A01 frame001
global_idx 296    -> S01 A01 frame297
global_idx 297    -> S01 A02 frame001
global_idx 320759 -> S40 A27 frame297
```

## Source-Only Split Protocol

The first implementation supports single-environment source-only training only:

```text
protocol = source_only
env_id in {1,2,3,4}
split in {train,val,test}
```

Within the selected environment, subjects are sorted ascending:

```text
train = first 7 subjects
val   = next 2 subjects
test  = last 1 subject
```

Examples:

```text
Env01: train S01-S07, val S08-S09, test S10
Env02: train S11-S17, val S18-S19, test S20
Env03: train S21-S27, val S28-S29, test S30
Env04: train S31-S37, val S38-S39, test S40
```

`finetune` is intentionally not implemented in the first version.

## Configuration And Paths

`configs/build_memmap.yaml` should contain reusable non-private parameters only. Server data paths are command-line arguments.

CSI path template:

```text
A{action_id:02d}/S{subject_id:02d}/frame_{frame_id_1based:03d}.mat
```

GT path template:

```text
E{env_id:02d}_S{subject_id:02d}_A{action_id:02d}.npy
```

Build command shape:

```bash
python scripts/build_memmap.py \
  --config configs/build_memmap.yaml \
  --csi-root /path/to/csi_root \
  --gt-root /path/to/gt_root \
  --output-root /data/WiFiPose/dataset/memmap \
  --device auto
```

Do not commit generated data or private server paths.

## Verification

Tests use synthetic tensors and temporary `.npy` files only. They do not require real CSI/GT data.

Required invariants:

- Feature output shape is `[297,12,10,114]`.
- Channel order is fixed and tested.
- `D_center.mean(dim=T)` is approximately zero.
- `C_ant.mean(dim=R)` is approximately zero.
- `F_sub` keeps subcarrier length 114 and uses non-zero boundary-safe padding.
- Labels clean invalid points and clip xy/conf correctly.
- Metadata mapping matches the reference global indices.
- Dataset source-only filters select the correct subjects for each environment split.
