# PhysCSI-Pose

WiFi CSI based human pose recognition codebase.

The current project contains the offline data layer plus the first temporal
model path modules. Training loops, inference, evaluation, and final keypoint
regression are not implemented yet.

## Current Architecture

Implemented modules:

- `dataset/`: builds amplitude-only CSI features once and stores them as mmap-readable `.npy` arrays.
- `models.AmpFeatureMixEncoder`: encodes one frame of cached CSI features.
- `models.PoseAwareTokenProjection`: converts a window of encoder feature maps into compact frame tokens.
- `models.TemporalLiteTransformer`: models local temporal relations over short frame-token windows.

Current model path:

```text
X frame features:        [B, 12, 10, 114]
  -> AmpFeatureMixEncoder
Encoder map:             [B, 128, 10, 29]
  -> stack over L frames
Window encoder maps:     [B, L, 128, 10, 29]
  -> PoseAwareTokenProjection
Pose-aware tokens:       [B, L, 128]
  -> TemporalLiteTransformer
Temporal tokens:         [B, L, 128]
```

The remaining model path is intentionally open:

```text
Temporal tokens [B, L, 128]
  -> Pose Regression Head          # not implemented yet
  -> 17 keypoints                  # not implemented yet
```

`AmpFeatureMixEncoder` uses lightweight depthwise-separable CNN blocks with
GroupNorm and GELU. It first mixes the 12 physical input channels, then models
frequency-only, time-only, and joint time-frequency structure while downsampling
only the subcarrier axis from 114 to 29.

`PoseAwareTokenProjection` uses channel refinement, a global background token,
K residual attention maps over the 10x29 time-frequency grid, residual token
aggregation, and token fusion to produce one 128-D pose-aware token per frame.

`TemporalLiteTransformer` uses local depthwise Conv1d positional encoding, two
Pre-Norm lightweight Transformer blocks, 4-head non-causal self-attention with
learnable relative temporal bias, and a 2x FFN. It supports short windows where
`4 <= L <= 8` and preserves one output token per input frame.

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

## CSI Cleaning

During memmap construction, each raw CSI frame is converted from `[3, 114, 10]` to `[10, 3, 114]`. If a frame contains `NaN`, `inf`, or negative amplitude values, those values are repaired before feature extraction:

- Use the median of valid non-negative values from the same `(rx, subcarrier)` across the 10 packets.
- If that local packet series is fully invalid, fall back to the frame-level valid non-negative median.
- If an entire frame has no valid non-negative values, construction stops with an error.

Repair counts are recorded in `meta_build.json` under `csi_repair_stats`.

## Feature Ablations

The cached `X_all.npy` always stores all four feature groups as 12 channels. Training code can select feature groups at Dataset read time:

```python
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm", "f_sub"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train")
```

The default uses all feature groups: `["l_norm", "d_center", "f_sub", "c_ant"]`.
