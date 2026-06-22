# WM-WiFlow Phased Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged WiFi CSI to Human3.6M-17 2D pose pipeline, from resampled data construction through baseline, wavelet ablations, and full coarse/fine WM-WiFlow.

**Architecture:** The data builder owns Fourier resampling from `[3,114,10]` to `[3,114,64]` and saves model-ready tensors as `[N,3,114,64]`. Training code owns split-specific normalization to avoid leakage. Model work proceeds from a non-wavelet H36M17 baseline to SWT feature-bank ablations, then dual-scale encoders, joint scale gates, base-plus-residual decoding, and graph refinement.

**Tech Stack:** Python, NumPy, SciPy Fourier resampling, PyTorch, PyWavelets, pytest.

---

## Stage 0: Project Contracts

**Purpose:** Make Human3.6M-17, tensor layout, and run metadata authoritative before adding model code.

**Files:**
- Create: `dataset/h36m17.py`
- Create: `dataset/layouts.py`
- Create: `tests/test_h36m17_contract.py`

- [ ] Define `H36M17_JOINT_NAMES`, `H36M17_EDGES`, and joint groups exactly once.
- [ ] Add tests asserting 17 joints, 16 edges, expected distal joints `[3, 6, 13, 16]`, and no OpenPose18 constants.
- [ ] Define layout strings:
  - raw MAT: `[antenna, subcarrier, time10]`
  - model input: `[sample, antenna, subcarrier, time64]`
  - target: `[sample, joint, xy]`
- [ ] Run: `pytest tests/test_h36m17_contract.py`
- [ ] Commit: `chore: add h36m17 project contracts`

**Exit criteria:** There is one importable source of truth for H36M17 and tensor layouts.

---

## Stage 1: Resampled Clean Dataset Builder

**Purpose:** Move Fourier resampling into data construction and write training-ready CSI tensors.

**Files:**
- Modify or replace: `scripts/build_clean_dataset.py`
- Create: `tests/test_build_clean_dataset_resampled.py`

- [ ] Add a `--resample-time-steps 64` option.
- [ ] Read each raw `CSIamp` frame as `[3,114,10]`.
- [ ] Repair non-finite values before resampling.
- [ ] Apply Fourier resampling along the last axis only: `[3,114,10] → [3,114,64]`.
- [ ] Write `X_amp_resampled.npy` with shape `[N,3,114,64]`.
- [ ] Keep `Y_2d_clean.npy` shape `[N,17,2]`.
- [ ] Update `clean_manifest.json` with:
  - `x_layout: sample,antenna,subcarrier,time`
  - `raw_csi_shape: [3,114,10]`
  - `resampled_csi_shape: [3,114,64]`
  - `resample_method: scipy.signal.resample`
- [ ] Add tests with synthetic MAT data proving the output shape and layout.
- [ ] Run: `pytest tests/test_build_clean_dataset_resampled.py`
- [ ] Commit: `feat: build resampled clean dataset`

**Exit criteria:** Training can load CSI directly as `[3,114,64]` without runtime resampling or transposition.

---

## Stage 2: Split-Aware Dataset Loading and Normalization

**Purpose:** Avoid normalization leakage while keeping dataloader fast.

**Files:**
- Create: `dataset/resampled_pose_dataset.py`
- Create: `dataset/normalization.py`
- Create: `tests/test_resampled_pose_dataset.py`

- [ ] Load `X_amp_resampled.npy` with `mmap_mode='r'`.
- [ ] Load `Y_2d_clean.npy` with `mmap_mode='r'`.
- [ ] Accept explicit frame indices from split files.
- [ ] Compute `mean/std` only from training frame indices.
- [ ] Support per-antenna/per-subcarrier stats with shape `[1,3,114,1]`.
- [ ] Save run stats as `normalization_stats.npz`.
- [ ] Ensure validation/test datasets reuse training stats.
- [ ] Add tests that fail if validation samples influence training stats.
- [ ] Run: `pytest tests/test_resampled_pose_dataset.py`
- [ ] Commit: `feat: add split-aware resampled dataset`

**Exit criteria:** Each split has leak-free normalization and returns `(x, y, meta)` where `x=[3,114,64]`, `y=[17,2]`.

---

## Stage 3: Metrics and Losses

**Purpose:** Establish evaluation before model complexity increases.

**Files:**
- Create: `losses/pose_losses.py`
- Create: `metrics/pose_metrics.py`
- Create: `tests/test_pose_losses_metrics.py`

- [ ] Implement L1 coordinate loss.
- [ ] Implement H36M17 bone length loss using `H36M17_EDGES`.
- [ ] Implement per-joint MPJPE and joint-group MPJPE.
- [ ] Implement wrist MPJPE and ankle MPJPE from H36M17 indices.
- [ ] Add tests using small deterministic poses.
- [ ] Run: `pytest tests/test_pose_losses_metrics.py`
- [ ] Commit: `feat: add h36m17 losses and metrics`

**Exit criteria:** Baseline and all later ablations report comparable H36M17 metrics.

---

## Stage 4: Non-Wavelet Baseline

**Purpose:** Create a fair baseline before adding SWT.

**Files:**
- Create: `models/baseline_csi_pose.py`
- Create: `models/axial_attention.py`
- Create: `models/joint_decoder.py`
- Create: `tests/test_baseline_shapes.py`

- [ ] Implement stem: `[B,3,114,64] → [B,32,114,64]`.
- [ ] Implement spatial encoder: `[B,32,114,64] → [B,128,29,16]`.
- [ ] Implement axial attention encoder: `[B,128,29,16] → [B,256,29,16]`.
- [ ] Implement 17-query decoder: `[B,464,256] → [B,17,256]`.
- [ ] Implement coordinate head: `[B,17,256] → [B,17,2]`.
- [ ] Add shape tests for all public model outputs.
- [ ] Run: `pytest tests/test_baseline_shapes.py`
- [ ] Train a short smoke run on a small subset and record loss decreases.
- [ ] Commit: `feat: add non-wavelet h36m17 baseline`

**Exit criteria:** A complete non-wavelet baseline trains and evaluates on the resampled dataset.

---

## Stage 5: SWT Feature Bank and Direct-Fusion Ablations

**Purpose:** Verify whether wavelet bands contain useful signal before building the full dual-branch model.

**Files:**
- Create: `models/wavelet_feature_bank.py`
- Create: `models/wavelet_concat_baseline.py`
- Create: `tests/test_wavelet_feature_bank.py`

- [ ] Add `PyWavelets`-based SWT wrapper.
- [ ] Input: `[B,3,114,64]`.
- [ ] Output: dict or structured tensor with `raw`, `A3`, `D3`, `D2`, `D1`, each `[B,3,114,64]`.
- [ ] Add tests for shape, band names, and deterministic output on fixed input.
- [ ] Implement direct-concat baseline for ablation B.
- [ ] Run: `pytest tests/test_wavelet_feature_bank.py`
- [ ] Train ablations:
  - raw only
  - raw + A3 + D3
  - raw + D2 + D1
  - all bands
  - all except D1
- [ ] Commit: `feat: add swt feature bank`

**Exit criteria:** There is evidence whether D1/D2 help distal joints before adding the full architecture.

---

## Stage 6: Dual Spatial and Axial Encoders

**Purpose:** Separate coarse and fine paths while preserving comparable interfaces.

**Files:**
- Create: `models/scale_fusion.py`
- Create: `models/scale_encoders.py`
- Create: `tests/test_dual_scale_encoders.py`

- [ ] Implement shared scale feature mapper.
- [ ] Implement scale embeddings for `raw`, `A3`, `D3`, `D2`, `D1`.
- [ ] Implement sample-conditioned softmax fusion for coarse scales `[raw,A3,D3]`.
- [ ] Implement sample-conditioned softmax fusion for fine scales `[raw,D2,D1]`.
- [ ] Implement coarse encoder output `[B,128,29,16]`.
- [ ] Implement fine encoder output `[B,128,29,32]`.
- [ ] Implement fine/coarse axial token outputs `[B,464,256]` and `[B,928,256]`.
- [ ] Run: `pytest tests/test_dual_scale_encoders.py`
- [ ] Commit: `feat: add dual scale encoders`

**Exit criteria:** Coarse and fine branches produce documented token shapes and expose fusion weights for diagnostics.

---

## Stage 7: Scale-Aware Decoder and Base-Plus-Residual Output

**Purpose:** Let each H36M17 joint choose fine-scale contribution.

**Files:**
- Create: `models/scale_aware_decoder.py`
- Create: `tests/test_scale_aware_decoder.py`

- [ ] Decode coarse tokens into `Z_coarse=[B,17,256]`.
- [ ] Predict `P_base=[B,17,2]`.
- [ ] Decode fine tokens using `LayerNorm(Q + Z_coarse)`.
- [ ] Predict `alpha=[B,17,1]` with final gate bias `-2.0`.
- [ ] Predict `Delta_P=[B,17,2]`.
- [ ] Output `P_final = P_base + alpha * Delta_P`.
- [ ] Add tests for output shapes and initial alpha near `sigmoid(-2)`.
- [ ] Run: `pytest tests/test_scale_aware_decoder.py`
- [ ] Commit: `feat: add scale aware joint decoder`

**Exit criteria:** Model can run with and without fine branch, and `P_base` remains a complete pose.

---

## Stage 8: H36M17 Graph Refinement

**Purpose:** Add skeleton-aware refinement as a separately measurable stage.

**Files:**
- Create: `models/h36m17_graph_refiner.py`
- Create: `tests/test_h36m17_graph_refiner.py`

- [ ] Build normalized adjacency from `H36M17_EDGES`.
- [ ] Add residual graph propagation over `[B,17,256]`.
- [ ] Add optional joint self-attention.
- [ ] Integrate graph refinement into baseline and WM-WiFlow behind a config flag.
- [ ] Run graph-on vs graph-off ablation.
- [ ] Run: `pytest tests/test_h36m17_graph_refiner.py`
- [ ] Commit: `feat: add h36m17 graph refinement`

**Exit criteria:** Graph refinement is independently toggleable and measurable.

---

## Stage 9: Training Orchestration and Diagnostics

**Purpose:** Make all stages trainable and comparable.

**Files:**
- Create: `train.py`
- Create: `eval.py`
- Create: `configs/baseline.yaml`
- Create: `configs/wavelet_concat.yaml`
- Create: `configs/wm_wiflow.yaml`
- Create: `tests/test_config_smoke.py`

- [ ] Add config switches for wavelet bands, fine branch, gate, graph, losses, and trainable groups.
- [ ] Log overall, per-joint, joint-group, wrist, and ankle metrics.
- [ ] Log coarse/fine fusion weights and joint alpha.
- [ ] Save model metadata: `skeleton_name=human36m17`, `num_joints=17`, `input_layout=antenna,subcarrier,time`.
- [ ] Reject checkpoint loading when metadata mismatches current config.
- [ ] Run: `pytest tests/test_config_smoke.py`
- [ ] Commit: `feat: add training orchestration`

**Exit criteria:** Baseline, wavelet concat, and full WM-WiFlow can be trained, evaluated, and compared from config files.

---

## Recommended Experimental Sequence

1. Train Stage 4 baseline.
2. Train Stage 5 direct wavelet concat ablations.
3. Train Stage 6 dual encoder without scale-aware residual decoder.
4. Train Stage 7 full base-plus-residual decoder.
5. Add Stage 8 graph refinement only after the previous stages are measured.
6. Run leave-one-environment-out and few-shot finetune after the single-source pipeline is stable.

This sequence keeps each performance gain attributable to one design change.
