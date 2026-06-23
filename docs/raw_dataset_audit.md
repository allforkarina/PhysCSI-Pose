# Raw Dataset Audit

This project should audit the raw WiFi CSI amplitude and Human3.6M-17 GT data before data cleaning, model training, or large-scale preprocessing.

## How to Run

```bash
python scripts/scan_raw_dataset.py \
  --csi-root /data/WiFiPose/dataset/dataset \
  --gt-root /data/WiFiPose/dataset/ground_truth_npy \
  --output-root /data/WiFiPose/dataset/raw_audit
```

Optional arguments:

```bash
python scripts/scan_raw_dataset.py --help
```

Useful options:

- `--csi-key CSIamp`: MAT key used for CSI amplitude.
- `--target-time-steps 64`: target temporal length used when writing model recommendations.
- `--sample-visualizations 5`: number of GT skeleton visualizations to save.
- `--max-files N`: debug-only limit for scanned CSI files.

The script is read-only with respect to the raw data roots. It writes reports only under `--output-root`.

## Input Directory Requirements

Expected raw CSI layout:

```text
A##/S##/wifi-csi/frame###.mat
```

Example:

```text
/data/WiFiPose/dataset/dataset/A01/S01/wifi-csi/frame001.mat
```

Expected GT layout:

```text
E##_S##_A##.npy
```

Example:

```text
/data/WiFiPose/dataset/ground_truth_npy/E04_S34_A09.npy
```

The script validates these assumptions instead of treating them as guaranteed.

## Outputs

The audit writes:

```text
audit_summary.json
architecture_contract.yaml
sequence_inventory.csv
csi_file_inventory.csv
gt_file_inventory.csv
pairing_errors.csv
missing_frames.csv
abnormal_files.csv
csi_shape_histogram.json
gt_shape_histogram.json
csi_statistics.json
gt_statistics.json
environment_statistics.csv
subject_statistics.csv
action_statistics.csv
sampled_pairings.csv
visualizations/
audit.log
```

`audit_summary.json` gives the high-level pass/fail status, blocking errors, warning categories, and file counts.

`architecture_contract.yaml` is the main output for later implementation. It records observed dataset sizes, CSI layout, GT layout, alignment status, preprocessing recommendations, and model shape recommendations.

`abnormal_files.csv`, `missing_frames.csv`, and `pairing_errors.csv` should be checked before building any cleaned dataset.

## Reading `architecture_contract.yaml`

Key fields:

- `dataset.frames_per_sequence.unique_values`: observed GT sequence lengths.
- `dataset.fixed_sequence_length`: whether fixed indexing is safe.
- `dataset.indexing.use_cumulative_offsets`: use cumulative offsets when sequence lengths vary.
- `csi.layout`, `csi.antennas`, `csi.subcarriers`, `csi.packets_per_pose`: observed CSI tensor contract.
- `gt.skeleton`: fixed to `human36m17`.
- `gt.used_dimensions`: `[0, 1]`; the project uses 2D xy output.
- `alignment.all_sequences_aligned`: whether GT and CSI frame pairing passed.
- `recommended_preprocessing.temporal_resampling`: whether `[source_length] -> [target_length]` Fourier resampling is appropriate.
- `recommended_model.input_shape`: model input shape after recommended resampling.
- `recommended_model.output_shape`: always `[17, 2]`.

## Blocking Conditions

Treat these as blocking for data construction or training:

- GT filenames do not map cleanly to environment, subject, and action.
- CSI filenames do not map cleanly to action, subject, and frame.
- Same subject/action appears in multiple environments while CSI paths do not include environment.
- Missing CSI frames for a GT sequence.
- GT joint dimension is not 17.
- CSI shape is not stable enough to define a model input contract.
- Non-finite CSI values exist and no repair strategy has been decided.

Non-blocking warnings can still matter for model quality and should be reviewed.

## Adjusting Model Shapes

Use the scan results instead of hard-coding assumptions.

If `csi.antennas` changes, update model input channels and antenna mixing layers.

If `csi.subcarriers` changes, update convolution output lengths, positional encoding lengths, and token counts. The contract includes the subcarrier length after two stride-2 reductions.

If `csi.packets_per_pose.unique_values` is fixed and equals 10, Fourier resampling to 64 can be used before training. If packet counts vary, build a preprocessing stage that handles variable source length explicitly.

SWT level depends on the final temporal length. For target length 64, level 3 is supported.

GT remains fixed:

```text
17 Human3.6M joints
Human3.6M-17 adjacency
model output [B,17,2]
```

## Notes

The GT skeleton order is already confirmed as Human3.6M-17. All GT joints are considered valid; no confidence mask or valid mask should be introduced.

Do not start formal data cleaning, training, or large-scale preprocessing until the audit contract has no blocking errors.
