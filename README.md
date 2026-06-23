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

For source-only training, split selection is strictly ordered:

1. Filter by source environment first, for example `--source-envs env1`.
2. Split subjects within that environment into train, val, and test.
3. Assign every action/trial/frame for a subject to exactly one split.

Do not randomly split frames from the same subject across train, val, and test.
