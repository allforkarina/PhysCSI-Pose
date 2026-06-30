# Fixed Random-Frame Protocol Design

**Date:** 2026-06-30

## Status

This design supersedes `2026-06-30-source-split-strategy-design.md`. The project no longer supports subject-disjoint source splits or a configurable split strategy.

## Objective

Make deterministic random frame-level splitting the only train/validation/test protocol used by the project. Remove the `--split-strategy` API and all subject-disjoint split code so commands cannot accidentally select a different task definition.

The model, H36M-17 targets, PCK definition, loss, optimizer, scheduler, and feature evaluation remain unchanged.

## Fixed Dataset Protocol

For `split=train`, `split=val`, or `split=test`, `MemmapDataset` performs these steps:

1. Filter all frames by the requested environment set.
2. Group the remaining frame indices by subject.
3. Iterate subjects in sorted order and shuffle each subject's frame indices using deterministic randomness driven by the configured seed.
4. Allocate rounded counts of 20 percent to test and 10 percent to validation.
5. Allocate all remaining frames to train, yielding an effective 70/10/20 split for normal MM-Fi sequence sizes.
6. Return sorted indices for stable memmap access.

Train, validation, and test indices are pairwise disjoint. Their union covers every environment-filtered frame. Every sufficiently populated subject contributes frames to all three sets.

For `split=all`, return every environment-filtered frame without shuffling or partitioning.

## API Removal

Remove these interfaces completely:

- `--split-strategy` from `train.py`.
- `--split-strategy` from `eval.py`.
- `TrainConfig.split_strategy`.
- `SPLIT_STRATEGIES`.
- `MemmapDataset.split_strategy` and its constructor parameter.
- `create_memmap_data_loader(..., split_strategy=...)`.
- `create_memmap_data_loaders(..., split_strategy=...)`.
- `train_subjects`, `val_subjects`, and `test_subjects` parameters from dataset and loader factories.
- The source-only 7/1/2 subject resolver and its tests.

Passing `--split-strategy` after this change is an argparse error. This is intentional because silently accepting or ignoring the obsolete option would hide the actual experiment protocol.

## Source-Only Training

`source_only` continues to require exactly one `--source-envs` value. It no longer requires exactly ten subjects and does not resolve explicit subject sets.

The source train, validation, and test loaders all use the fixed frame-level protocol. Training logs print the train, validation, and test frame counts so the active protocol is observable without a configuration flag.

The selected environment and existing training settings continue to be stored in checkpoint `train_config`. New checkpoints do not contain `split_strategy`. Older checkpoints containing that extra key remain loadable because evaluation reconstructs the model only from architecture fields.

## Other Training Modes

Few-shot target loaders use `split=all` and explicit few-shot sampling, so their behavior is unchanged.

`finetune_align` constructs its supervised source loader with `split=train`; that source loader now uses the same fixed random frame-level training partition. This keeps the project-wide source protocol consistent.

## Evaluation

`eval.py` exposes only `--eval-split {train,val,test,all}`. The selected split is automatically constructed with the fixed random frame protocol and the dataset's default seed.

Metric evaluation and pose visualization must construct datasets identically. `eval-split=all` remains available for explicitly requested full-subset or few-shot evaluation.

## Tests

Tests must verify:

1. The fixed protocol produces exact 70/10/20 counts for synthetic subjects with ten frames each.
2. Train, validation, and test indices are pairwise disjoint and jointly complete.
3. Every sufficiently populated subject occurs in every split.
4. Equal seeds reproduce equal indices and different seeds change membership.
5. Environment filtering happens before partitioning.
6. `split=all` remains unpartitioned.
7. Dataset and loader constructors no longer expose strategy or explicit subject-set parameters.
8. Training and evaluation parsers reject `--split-strategy`.
9. Source-only validation accepts one environment regardless of subject count and rejects zero or multiple environments.
10. Evaluation metric and visualization datasets use the same fixed constructor path.

## Documentation

Update `AGENTS.md`, `README.md`, and `REASONIX.md` to describe random frame-level splitting as the only project protocol. Remove subject-disjoint commands and comparisons. Training artifacts remain under `runs/`; evaluation artifacts remain under `outputs/`.

Historical specifications and plans remain in `docs/superpowers/` as records and are not runtime documentation.

## Non-Goals

- Adding a temporal-block split.
- Reproducing the demo's OpenPose-18 target conversion.
- Changing the fixed 70/10/20 ratios.
- Adding a public seed CLI.
- Changing model architecture or training hyperparameters.
- Migrating or rewriting existing checkpoint files.

## Success Criteria

- No runtime source file exposes or references `split_strategy` or subject-disjoint split parameters.
- Train, validation, and test always use the deterministic per-subject random frame protocol.
- Source-only training accepts exactly one environment without requiring ten subjects.
- `train.py --help` and `eval.py --help` contain no split-strategy option.
- Focused and full tests pass in the `WiFiPose` Conda environment.
- Runtime documentation contains only the fixed random-frame workflow.
