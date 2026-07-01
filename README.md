# PhysCSI-Pose

## Project Paths

Linux server project code:

```text
/data/WiFiPose/PhysCSI-Pose
```

Raw CSI data:

```text
/data/WiFiPose/dataset/dataset
```

Actual CSI file structure:

```text
/data/WiFiPose/dataset/dataset/A01/S01/wifi-csi/frame001.mat
```

Ground-truth labels:

```text
/data/WiFiPose/dataset/ground_truth_npy
```

Ground-truth filename example:

```text
/data/WiFiPose/dataset/ground_truth_npy/E04_S34_A09.npy
```

## Ground-Truth Skeleton

The project GT is confirmed to use the standard Human3.6M-17 human skeleton joint order. The labels are already ordered as Human3.6M and must not be converted to OpenPose.

Training targets use only raw `xy` coordinates from the GT arrays, with shape `[N, 17, 2]`. The data build step should preserve the original coordinate range and should not apply image-size normalization or clip labels into `[-0.8, 0.8]`.

The memmap CSI tensor is built from raw `CSIamp` shaped `[3, 114, 10]`, Fourier-resampled on the time axis to 64 packets, saved as `[N, 64, 3, 114]`, and loaded by the dataloader as model input `[B, 3, 114, 64]`.

## Source-Domain Split Protocol

For source-only training, split selection is fixed:

1. Select exactly one source environment, for example `--source-envs env1`.
2. Group the environment's frames by subject.
3. Deterministically shuffle each subject's frames with seed 42.
4. Assign 70% of each subject's frames to train, 10% to validation, and 20% to test.

Train, validation, and test frame indices are mutually exclusive and jointly cover the selected environment. Random frame-level splitting is the only project protocol.

## CSI Input Normalization Ablation

Training exposes three precomputed CSI representations through `--normalization`:

- `global_minmax` loads `csi_gminmax.npy` and remains the default.
- `global_zscore` loads `csi_gzscore.npy`.
- `per_sample_zscore` loads `csi_zscore.npy`.

The selected value is saved in checkpoint `train_config`. Evaluation restores it automatically; do not manually substitute CSI files. Checkpoints created before this option default to `global_minmax`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization global_minmax --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_global_minmax
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization global_zscore --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_global_zscore
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode source_only --dataset-root /data/WiFiPose/dataset/mmfi_pose_v3 --source-envs env1 --normalization per_sample_zscore --epochs 50 --batch-size 64 --num-workers 8 --output-dir runs/source_env1_per_sample_zscore
```
