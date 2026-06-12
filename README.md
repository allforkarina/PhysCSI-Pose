# PhysCSI-Pose

Data-layer implementation for WiFi CSI based human pose recognition.

The first version builds amplitude-only CSI features once and stores them as mmap-readable `.npy` arrays. Model networks, training loops, inference, and evaluation are intentionally not part of this phase.

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

## Feature Ablations

The cached `X_all.npy` always stores all four feature groups as 12 channels. Training code can select feature groups at Dataset read time:

```python
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train", features=["l_norm", "f_sub"])
MemmapPoseDataset(root, protocol="source_only", env_id=1, split="train")
```

The default uses all feature groups: `["l_norm", "d_center", "f_sub", "c_ant"]`.
