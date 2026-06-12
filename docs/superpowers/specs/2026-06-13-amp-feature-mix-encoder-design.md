# AmpFeatureMixEncoder Design

**Date:** 2026-06-13
**Status:** draft
**Scope:** PhysCSI-Pose first-version CSI feature encoder

---

## 1. Module Positioning

`AmpFeatureMixEncoder` is the CSI feature encoder in PhysCSI-Pose v1. It consumes
offline-built amplitude-only physical features, not raw CSIamp.

| | Shape | Description |
|---|---|---|
| Input  | `[B, 12, 10, 114]` | 4 feature groups × 3 Rx antennas, 10 packets, 114 subcarriers |
| Output | `[B, 128, 10, 29]` | Intermediate time-frequency feature map |

The encoder does **not** output keypoints. A downstream Pose Regression Head
consumes `Z` to predict 17 2D joints.

**Design goals:**

1. Do not learn environment fingerprints from raw amplitude.
2. Exploit the 12-channel physical-feature priors.
3. Extract local subcarrier-domain frequency perturbations.
4. Preserve the short-time dynamics across 10 CSI packets.
5. Use a lightweight CNN to control parameter count and overfitting risk.
6. Produce a stable representation suitable for single-domain and cross-domain
   few-shot experiments downstream.

---

## 2. Input Channel Definitions

The 12 input channels are ordered and immutable:

| Channels | Feature Group | Rx Antennas |
|---|---|---|
| 0–2  | `l_norm`   | rx0, rx1, rx2 |
| 3–5  | `d_center` | rx0, rx1, rx2 |
| 6–8  | `f_sub`    | rx0, rx1, rx2 |
| 9–11 | `c_ant`    | rx0, rx1, rx2 |

**Physical semantics:**
- **l_norm:** sequence-level median/MAD-normalised amplitude residual —
  deviation from the per-(E,S,A) pseudo-static background.
- **d_center:** short-time centered dynamics across the 10 packets of one pose
  frame (~100 ms window) — captures limb motion, occlusion, and reflection-path
  changes.
- **f_sub:** subcarrier local contrast — `l_norm` minus a subcarrier-direction
  smooth background; highlights local frequency-domain perturbations.
- **c_ant:** inter-antenna relative amplitude — per-Rx `l_norm` minus the
  instantaneous 3-Rx mean; suppresses common receive power and link gain.

---

## 3. Why CNN (not Transformer) for v1

1. **Sequence length is only 10.** 10 packets per frame is not a long-sequence
   problem; Transformer gains are limited.
2. **Subcarrier locality.** Adjacent subcarriers have continuous frequency
   response; human-induced perturbations appear in local subcarrier
   neighbourhoods. CNN local convolutions match this structure better than
   global self-attention.
3. **Less risk of memorising environment fingerprints.** Transformers see the
   full spectrogram pattern and may latch onto domain-specific frequency-envelope
   shapes. Local convolutions bias toward local perturbation extraction.
4. **Smaller parameter count.** Lightweight CNN is easier to train, debug, and
   overfit-sanity-check.
5. **Extension path.** If CNN capacity is insufficient, attention pooling or a
   lightweight Transformer head can be added on top of the encoder output.

---

## 4. Normalisation & Activation

- **GroupNorm** throughout (preferred `num_groups=8`; fallback `4` if channels
  not divisible by 8).
  - BatchNorm running statistics risk memorising source-domain distributions
    across environments.
  - GroupNorm is batch-size-independent, stable for small batches and cross-
    subject validation.
- **GELU activation.** Smoother than ReLU; better suited to continuous
  amplitude-perturbation features (CSI is not sparse semantic image data).

---

## 5. Basic Convolution Block

Every stage uses **Depthwise Separable Convolution**:

```
Depthwise Conv2d  →  Pointwise Conv2d  →  GroupNorm  →  GELU  →  [+ Residual]
```

- **Depthwise Conv:** extracts local time-frequency patterns within each channel
  independently — preserves per-channel physical structure.
- **Pointwise Conv:** 1×1 cross-channel mixing — learns combinations of physical
  features and Rx antennas.

**Rationale:** fewer parameters, lower compute, reduced overfitting, and explicit
control over whether each stage models time, frequency, or joint time-frequency.

**Residual connection:** used when input/output channels match; otherwise a 1×1
Conv projection on the shortcut. Residuals allow the model to learn new
perturbation patterns while retaining earlier CSI representations.

---

## 6. Encoder Architecture

```
Input:        [B,  12, 10, 114]
Stage 0:      [B,  32, 10, 114]   Channel Projection (1×1 Conv)
Stage 1:      [B,  48, 10, 114]   Frequency Block
Stage 2:      [B,  64, 10, 114]   Temporal Block
Stage 3:      [B,  96, 10,  57]   Joint TF Block 1 + subcarrier ↓2
Stage 4:      [B, 128, 10,  29]   Joint TF Block 2 + subcarrier ↓2
Output:       [B, 128, 10,  29]
```

| Stage | Type | In→Out Ch | Kernel | Padding | Stride | Output Shape |
|-------|------|-----------|--------|---------|--------|--------------|
| 0 | Conv2d 1×1 | 12→32 | 1 | 0 | 1 | `[B,32,10,114]` |
| 1 | DWConv + PWConv | 32→48 | (1,7) | (0,3) | 1 | `[B,48,10,114]` |
| 2 | DWConv + PWConv | 48→64 | (3,1) | (1,0) | 1 | `[B,64,10,114]` |
| 3 | DWConv + PWConv | 64→96 | (3,5) | (1,2) | 1 | `[B,96,10,114]` |
|   | Subcarrier downsample | — | (1,3) | (0,1) | (1,2) | `[B,96,10,57]` |
| 4 | DWConv + PWConv | 96→128 | (3,5) | (1,2) | 1 | `[B,128,10,57]` |
|   | Subcarrier downsample | — | (1,3) | (0,1) | (1,2) | `[B,128,10,29]` |

**Stage 0 — Channel Projection:** 1×1 Conv only in the channel dimension.
Learns weighted combinations of the 12 physical features without mixing time or
subcarrier axes. Acts as a learnable "physical-feature fuser."

**Stage 1 — Frequency Block:** convolves only along subcarriers (kernel=(1,7)).
Extracts local frequency-domain perturbation patterns (human-induced peaks and
valleys across adjacent subcarriers). Time axis left untouched.

**Stage 2 — Temporal Block:** convolves only along packets (kernel=(3,1)).
Captures short-time (~100 ms) dynamics within one pose frame. Time dimension
kept at 10 — no early pooling that would discard packet-level motion cues.

**Stage 3 — Joint TF Block 1:** simultaneous time-frequency convolution
(kernel=(3,5)), then subcarrier downsampling 114→57. Time axis stays at 10.

**Stage 4 — Joint TF Block 2:** further abstraction (kernel=(3,5)), subcarrier
downsample 57→29. Outputs a medium-scale CSI pose feature map.

**Downsampling rule:** subcarrier dimension only (114→57→29); time dimension
always 10. Use learnable Conv2d with stride=(1,2), kernel=(1,3), padding=(0,1).

---

## 7. Why the Encoder Does NOT Global-Pool

The output stays `[B, 128, 10, 29]` rather than `[B, 128]` because:

1. The encoder is a feature extractor — it does not pre-commit to a regression
   head form.
2. Downstream heads can choose Global Average Pool, Attention Pool, or Joint
   Query aggregation.
3. Retaining the 10×29 time-frequency grid enables analysis of which temporal
   and subcarrier regions the model attends to.
4. Premature pooling risks losing local time-frequency information.

---

## 8. Implementation Interface

```python
class AmpFeatureMixEncoder(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

- **Input:** `x` — `torch.float32`, shape `[B, 12, 10, 114]`
- **Output:** `z` — `torch.float32`, shape `[B, 128, 10, 29]`
- **Guard:** `assert x.ndim == 4 and x.shape[1] == 12 and x.shape[2] == 10 and x.shape[3] == 114`

**Smoke test:**
```python
encoder = AmpFeatureMixEncoder()
x = torch.randn(2, 12, 10, 114)
z = encoder(x)
assert z.shape == (2, 128, 10, 29)
assert torch.isfinite(z).all()
```

**File location:** `models/amp_feature_mix_encoder.py`

---

## 9. Future Extension Points (NOT in v1)

If source-domain performance is insufficient after v1 training:

1. Add a Stage 5: `[B,128,10,29] → [B,192,10,15]`
2. Squeeze-and-Excitation channel attention on high-level channels.
3. Spatial attention on the 10×29 time-frequency grid.
4. Lightweight Transformer on top-level tokens (not on raw 10×114 input).
5. Compare GroupNorm vs. LayerNorm for very small batches.

None of these are included in v1 — keep it simple, debuggable, reproducible.
