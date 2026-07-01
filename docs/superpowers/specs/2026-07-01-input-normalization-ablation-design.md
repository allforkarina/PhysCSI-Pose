# Input Normalization Ablation Design

## Objective

Expose the three CSI normalization variants already stored in the NPY memmap dataset as a controlled source-domain training ablation:

1. `global_minmax` -> `csi_gminmax.npy`
2. `global_zscore` -> `csi_gzscore.npy`
3. `per_sample_zscore` -> `csi_zscore.npy`

The ablation tests whether retaining absolute CSI amplitude scale, globally standardizing it, or removing each frame's absolute scale produces the best same-environment Human3.6M-17 pose regression performance.

## Experimental Hypotheses

| Variant | Physical hypothesis | Expected risk |
| --- | --- | --- |
| `global_minmax` | Absolute amplitude and relative attenuation remain available to the model. | Extreme amplitudes can compress most samples into a narrow range. |
| `global_zscore` | Global centering and scaling improve optimization while preserving between-frame amplitude differences. | Outliers still influence the global mean and standard deviation. |
| `per_sample_zscore` | Removing frame-level gain variation makes the model focus on antenna, subcarrier, and temporal structure. | Absolute attenuation cues that correlate with body position may be removed. |

All variants must use the same random frame split, seed, source environment, model, loss, optimizer, scheduler, batch size, and epoch count. Final claims should use at least three seeds; one-seed runs are exploratory screening only.

## Considered Approaches

### 1. Checkpoint-backed CLI selection (selected)

Add one training argument, pass it through every loader, store it in `train_config`, and let evaluation recover it from the checkpoint. This produces explicit, reproducible commands and prevents training/evaluation normalization mismatch.

### 2. Separate dataset directories

Create one dataset directory per normalization and retain the current implicit default. This avoids code changes but duplicates metadata, obscures the changed factor in commands, and makes accidental file substitution likely.

### 3. Replace or symlink `csi_gminmax.npy`

Swap the file behind the current default before each run. This is unsafe for concurrent experiments and makes a checkpoint insufficient to reconstruct its input pipeline.

Approach 1 is selected because it is the smallest reliable interface and records the experimental factor in every checkpoint.

## Interface

`train.py` gains:

```text
--normalization {global_minmax,global_zscore,per_sample_zscore}
```

The default remains `global_minmax`, preserving current commands and old behavior. `TrainConfig.normalization` stores the canonical public name.

`eval.py` does not gain a competing normalization override. It reads `train_config.normalization` from the checkpoint and uses that value for the evaluation and visualization datasets. Checkpoints created before this change fall back to `global_minmax`.

The existing internal dataset spelling `zscore` is replaced by the unambiguous public name `per_sample_zscore`. The backing filename remains `csi_zscore.npy`; rebuilding the dataset is not required.

## Data Flow

Training follows this path:

```text
train CLI -> TrainConfig.normalization -> loader factory -> MemmapDataset -> selected CSI .npy file
                                             |
                                             +-> checkpoint train_config
```

Evaluation follows this path:

```text
checkpoint train_config.normalization -> MemmapDataset -> selected CSI .npy file
```

The normalization value must be propagated through source-only, finetune, finetune-align, few-shot, evaluation, pose visualization, and feature visualization paths so every consumer of the checkpoint sees the same CSI representation.

## Error Handling and Compatibility

- `argparse` rejects unknown training normalization names before training starts.
- `MemmapDataset` reports the accepted canonical names when called directly with an invalid name.
- A missing selected `.npy` file fails with the existing file-loading error and identifies the required path.
- Old checkpoints without `train_config.normalization` evaluate with `global_minmax`.
- Existing commands without `--normalization` retain `global_minmax` behavior.

## Tests

Focused tests will verify:

1. Each canonical normalization name selects the expected memmap file.
2. Loader factories propagate the selected normalization.
3. The training parser accepts all three values and rejects invalid values.
4. `TrainConfig` serializes the selected value into checkpoint metadata.
5. Evaluation restores the normalization from a new checkpoint.
6. Evaluation falls back to `global_minmax` for an old checkpoint.
7. Visualization dataset construction uses the same restored normalization.
8. The existing full test suite remains green.

## Evaluation Protocol

For each variant, select `best_val_pck_0_2.pth` using the same validation split, then evaluate the fixed test split. Report test MPJPE, PCK@0.2, PCK@0.5, per-joint MPJPE/PCK, overall variance ratio, standard-deviation ratio, and mean-pose distance. Store training artifacts under `runs/` and evaluation artifacts under `outputs/` with the normalization name in each directory.

## Explicitly Out of Scope

- Adding `per_rx_subcarrier_zscore`.
- Rebuilding existing normalization files.
- Changing how global statistics were originally fitted.
- Correcting train/validation/test leakage in the precomputed global statistics.
- Changing the random frame split, model architecture, pose targets, losses, or metrics.

The statistics-fitting issue must be addressed in a separate dataset-rebuild design before using these results as a leakage-free final comparison.
