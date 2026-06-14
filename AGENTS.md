# PhysCSI-Pose Agent Instructions

## Project Scope

- Build a WiFi CSI based human pose recognition model from scratch in this repository.
- Use this local workspace for source-code authoring and code-correctness tests only.
- Do not run model inference that requires real WiFi CSI data unless the user explicitly provides suitable data and asks for it.
- Prefer small, testable modules with clear interfaces for CSI preprocessing, model definition, training utilities, and evaluation helpers.

## Current Project Status

- The offline data layer is implemented: CSI cleaning, amplitude-only feature construction, label cleaning, metadata construction, source-only split helpers, and mmap-backed Dataset loading.
- The current model code implements these modules:
  - `AmpFeatureMixEncoder`: `[B,12,10,114] -> [B,128,10,29]`.
  - `PoseAwareTokenProjection`: `[B,L,128,10,29] -> [B,L,128]`.
  - `TemporalLiteTransformer`: `[B,L,128] -> [B,L,128]`.
  - `PoseHeatmapDecoder`: `[B,L,128] -> [B,L,17,2]`.
  - `PhysCSIPoseNet`: `[B,L,C,10,114] -> [B,L,17,2]`, where `C` is selected from cached feature groups for ablations.
- The current implemented model architecture is:
  `amplitude feature frame -> AmpFeatureMixEncoder -> windowed encoder maps -> PoseAwareTokenProjection -> TemporalLiteTransformer -> PoseHeatmapDecoder -> pose coordinates`.
- Training and evaluation code is implemented through `train.py`, `eval.py`, and `engine/`.
- Local verification still uses synthetic tests only. Real training/evaluation should run on the Linux server with memmap data under `/data/WiFiPose/dataset/memmap`.
- Do not claim model accuracy, inference quality, or runtime performance without an explicit real-data evaluation run.

## GitHub Remote

- The repository remote must point to `git@github.com:allforkarina/PhysCSI-Pose.git`.
- After each completed modification, commit and push the full intended change set to GitHub.
- Before committing, inspect the staged files and ensure only allowed files are included.

## Allowed GitHub Content

Push only:

- Source code scripts and package files.
- Tests and lightweight test fixtures that do not contain private data.
- Configuration needed to run code and tests.
- Explanatory documentation, including this `AGENTS.md` file and project design notes.

Do not push:

- Raw or processed WiFi CSI datasets.
- Model checkpoints, weights, or exported inference artifacts.
- Experiment outputs, plots, tensorboard logs, generated reports, caches, local virtual environments, or editor metadata.
- `runs/` and `outputs/` contents, including checkpoints, metrics exports, predictions, and cached window-index files.
- Credentials, tokens, private machine paths, or user-specific runtime configuration.

## Verification Expectations

- Run code-correctness checks after changes when relevant.
- For the current data-limited phase, tests should use synthetic tensors, mocks, or small deterministic fixtures.
- Do not claim inference quality, accuracy, or runtime performance without real data and an explicit evaluation run.
