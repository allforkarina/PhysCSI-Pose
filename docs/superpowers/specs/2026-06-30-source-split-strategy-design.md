# Source-Domain Split Strategy Design

**Date:** 2026-06-30

## Objective

Add a controlled source-domain diagnostic that compares the existing subject-disjoint split with a random frame-level split while keeping the H36M-17 targets, model, loss, optimizer, scheduler, seed, and evaluation metrics unchanged.

The frame-level strategy is diagnostic only. The subject-disjoint strategy remains the default and the protocol for reporting unseen-subject generalization.

## Experimental Hypothesis

If the large PCK gap relative to the Wi-Posev2 demo is primarily caused by split leakage, the same current model and H36M-17 target pipeline should score substantially higher when every source subject contributes frames to train, validation, and test sets.

This experiment does not attempt to reproduce the demo's OpenPose-18 conversion or its historical PCK scale. It isolates only the split strategy.

## User Interface

Add a `--split-strategy` option to source-only training and evaluation:

- `subject`: current behavior and default.
- `frame_random`: deterministic random frame split within every subject.

Examples:

```bash
python train.py --mode source_only --dataset-root data/mmfi_pose --source-envs env1 --split-strategy subject --output-dir runs/source_env1_subject
```

```bash
python train.py --mode source_only --dataset-root data/mmfi_pose --source-envs env1 --split-strategy frame_random --output-dir runs/source_env1_frame_random
```

```bash
python eval.py --dataset-root data/mmfi_pose --checkpoint runs/source_env1_frame_random/best_val_pck_0_2.pth --eval-envs env1 --eval-split test --split-strategy frame_random --output-dir outputs/source_env1_frame_random_test
```

The selected training strategy is stored in checkpoint `train_config`. Evaluation still requires the strategy explicitly so the evaluated dataset protocol is visible in the command and logs.

## Dataset Behavior

`MemmapDataset` accepts `split_strategy` with a default of `subject`.

### Subject Strategy

Preserve the current source-only protocol:

- Select exactly one source environment.
- Require exactly ten subjects in that environment.
- Assign the first seven sorted subjects to train, the eighth to validation, and the final two to test.
- Keep every frame from a subject in exactly one split.

Existing callers that omit `split_strategy` retain this behavior.

### Frame-Random Strategy

Apply environment filtering first, then group candidate frame indices by subject. For each subject:

1. Copy and shuffle that subject's frame indices with deterministic randomness derived from the configured seed.
2. Allocate 20 percent to test and 10 percent to validation using the existing rounded-count convention.
3. Allocate all remaining frames to train.
4. Return sorted indices for stable memmap access.

The resulting train, validation, and test index sets must be disjoint and their union must cover every candidate frame. Every sufficiently populated subject appears in all three splits. The MM-Fi source subjects have enough frames for non-empty splits; the implementation will retain bounded count calculations for small synthetic test fixtures.

The split is performed per subject rather than globally to match the leakage mechanism under diagnosis: the model sees every person's CSI characteristics during training while validation and test contain held-out frames from those same people.

## Training Integration

Add `split_strategy: str = "subject"` to `TrainConfig` and expose the CLI option with the two supported values.

For `source_only`:

- `subject` resolves and passes the explicit 7/1/2 subject sets exactly as today.
- `frame_random` validates the single source environment and ten available subjects but does not pass explicit split-specific subject sets. `MemmapDataset` performs the per-subject frame split.
- Print the selected strategy and either the explicit subject assignment or the frame counts for all three splits.

Other training modes keep their current behavior. The new option affects the source-only comparison and source dataset construction only; few-shot target selection remains unchanged.

## Evaluation Integration

Add the same `--split-strategy` option to `eval.py` and pass it to every `MemmapDataset` created for metric evaluation or visualization.

Evaluation defaults to `subject` for backward compatibility. A frame-random checkpoint must be evaluated with `--split-strategy frame_random`. Output labels continue to use the existing `train`, `val`, `test`, and `all` names.

For `eval-split=all`, the split strategy does not alter membership because all environment-filtered frames are returned.

## Controlled Comparison

The matched comparison is:

| Factor | Subject baseline | Frame diagnostic |
| --- | --- | --- |
| Split strategy | `subject` | `frame_random` |
| Ratio | 7/1/2 subjects | 70/10/20 frames per subject |
| H36M-17 GT | Same | Same |
| PCK torso scale | Same | Same |
| Model and decoder | Same | Same |
| Loss and optimizer | Same | Same |
| Seed and epochs | Same | Same |

Primary metrics are test MPJPE, PCK@0.2, per-joint metrics, per-action metrics, and prediction variance diagnostics. The mechanism evidence is whether `frame_random` improves PCK and prediction variance when subject identity and adjacent motion frames are shared across splits.

## Tests

Add focused tests that verify:

1. Omitting `split_strategy` preserves the current subject-disjoint behavior.
2. `frame_random` produces approximately 70/10/20 frames per subject using exact expected counts in synthetic fixtures.
3. Frame-random train, validation, and test indices are pairwise disjoint.
4. Their union covers every environment-filtered candidate frame.
5. The same seed reproduces the same indices.
6. A different seed changes at least one split.
7. Environment filtering occurs before frame splitting.
8. Training and evaluation CLI parsers accept both strategies and reject unknown values.
9. Evaluation passes the selected strategy into metric and visualization datasets.
10. Saved training configuration includes `split_strategy` through the existing dataclass checkpoint serialization.

## Documentation

Update `AGENTS.md` with:

- the two supported split strategies;
- the default subject-disjoint protocol;
- one-line source-only training and evaluation commands for the frame-random diagnostic;
- an explicit warning that frame-random results measure same-subject frame interpolation and must not be reported as unseen-subject generalization.

## Non-Goals

- Reproducing the demo's OpenPose-18 conversion.
- Changing PCK normalization or H36M-17 joint semantics.
- Adding configurable split ratios.
- Persisting split indices as separate files.
- Changing model architecture, loss weights, optimizer settings, or target-domain few-shot behavior.

## Success Criteria

- Both strategies can be selected from training and evaluation commands.
- Existing commands remain subject-disjoint without modification.
- Frame-random membership is deterministic, disjoint, and complete.
- Checkpoints record the training strategy.
- Focused and full tests pass in the `WiFiPose` Conda environment.
- `AGENTS.md` accurately documents the diagnostic workflow.
