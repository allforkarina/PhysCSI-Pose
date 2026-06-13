# PhysCSI-Pose

WiFi CSI based human pose recognition codebase.

The current project contains the offline data layer plus model-path modules
that cover CSI features through pose-coordinate decoding. Training loops,
inference, evaluation, and end-to-end experiment runners are not implemented yet.

## Current Architecture

Implemented modules:

- `dataset/`: offline CSI feature extraction, GT normalisation, memmap dataset
- `models.AmpFeatureMixEncoder`: lightweight depthwise-separable CNN that encodes one frame of 12-channel CSI features into a `[128, 10, 29]` time-frequency map
- `models.PoseAwareTokenProjection`: residual attention pooling that converts a window of encoder feature maps into compact 128-D pose-aware frame tokens
- `models.TemporalLiteTransformer`: 2-layer Pre-Norm Transformer over short frame-token windows (L=4–8) with learnable relative temporal bias
- `models.PoseHeatmapDecoder`: high-resolution heatmap decoder that maps temporal tokens to 17 keypoint coordinates

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
  -> PoseHeatmapDecoder
Pose coordinates:        [B, L, 17, 2]
```

`AmpFeatureMixEncoder` uses lightweight depthwise-separable CNN blocks with
GroupNorm and GELU. It first mixes the 12 physical input channels, then models
frequency-only, time-only, and joint time-frequency structure while downsampling
only the subcarrier axis from 114 to 29.

`PoseAwareTokenProjection` uses channel refinement, a global background token,
K=4 residual attention maps over the 10×29 time-frequency grid, residual token
aggregation, and token fusion to produce one 128-D pose-aware token per frame.

`TemporalLiteTransformer` uses depthwise Conv1d positional encoding, two
Pre-Norm Transformer blocks, 4-head non-causal self-attention with learnable
relative temporal bias, and 2× FFN expansion.  It operates on short windows
(4 ≤ L ≤ 8) and preserves one output token per input frame.

`PoseHeatmapDecoder` injects learned joint queries into each temporal token,
refines per-joint features, decodes a 64x64 heatmap for each joint, and uses
differentiable soft-argmax to output coordinates in the label range
`[-0.8, 0.8]`.

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
