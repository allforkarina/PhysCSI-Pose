# PoseHeatmapDecoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `PoseHeatmapDecoder`, the final high-resolution decoder that maps temporal frame tokens `[B, L, 128]` to 17 two-dimensional pose keypoints `[B, L, 17, 2]`.

**Architecture:** The decoder injects learned joint queries into each temporal token, refines per-joint tokens with residual MLP blocks, generates a high-resolution 64x64 heatmap for each joint, then applies differentiable soft-argmax over the heatmap to output coordinates in the existing target range `[-0.8, 0.8]`. The first version uses only temporal tokens and does not re-enter encoder feature maps, so it remains isolated and testable.

**Tech Stack:** PyTorch (`nn.Module`, `Embedding`/`Parameter`, `Linear`, `LayerNorm`, `GELU`, `Dropout`, `Conv2d`, `GroupNorm`, `Upsample`, tensor reshape), pytest, synthetic tensor tests only.

---

## Fixed Design

Public module name:

```python
PoseHeatmapDecoder
```

Default configuration:

```text
input_dim = 128
num_joints = 17
heatmap_size = 64
coord_min = -0.8
coord_max = 0.8
joint_hidden_dim = 128
decoder_channels = 64
seed_size = 8
dropout = 0.1
```

Default forward contract:

```python
coords = decoder(temporal_tokens)

# temporal_tokens: [B, L, 128]
# coords:          [B, L, 17, 2]
```

Optional heatmap diagnostics:

```python
coords, aux = decoder(temporal_tokens, return_heatmaps=True)

# aux["heatmap_logits"]: [B, L, 17, 64, 64]
# aux["heatmaps"]:       [B, L, 17, 64, 64]
```

The decoder is not responsible for:

- Building temporal windows.
- Computing losses.
- Applying confidence masks.
- Sliding-window inference or prediction averaging.
- Generating target heatmaps from GT.
- Consuming `[B,L,128,10,29]` encoder maps. That can be a later cross-attention decoder if needed.

## Layer-by-Layer Physical Meaning

### 1. Joint Query Injection

```text
temporal_tokens: [B, L, 128]
learned_joint_embedding: [17, 128]
joint_tokens: [B, L, 17, 128]
```

Physical meaning:

- The temporal token contains whole-frame CSI pose evidence after local temporal modelling.
- A learned joint embedding asks a joint-specific question: "where is the left wrist?", "where is the right elbow?", etc.
- Adding the same frame token to different joint embeddings lets every joint see the same global body state while decoding through a joint-specific channel.

Why this design:

- Directly regressing all 34 coordinates from one token forces all joints to share one representation bottleneck.
- Joint queries give small joints and limb endpoints their own decoding path without needing explicit body-part labels.
- Learned queries are lighter than a full joint Transformer and are enough for the first high-resolution decoder.

Implementation comment requirement:

- The code must explain that joint embeddings convert a global frame token into 17 joint-conditioned tokens, which is important because wrist/elbow/ankle evidence can be weaker than torso evidence.

### 2. Joint Feature Refinement

```text
joint_tokens: [B, L, 17, 128]
residual MLP block x 2
refined_joint_tokens: [B, L, 17, 128]
```

Physical meaning:

- Each joint token becomes a compact hypothesis for one body landmark.
- The residual MLP lets each joint recombine temporal pose evidence before producing spatial heatmap evidence.

Why this design:

- The temporal transformer already models cross-frame relations; the decoder should not add another large sequence model.
- A small residual MLP is enough to specialize the global temporal token into per-joint features.
- Residual paths keep the original joint-conditioned signal intact, reducing risk for few-shot fine-tuning.

Implementation comment requirement:

- The code must explain why this refinement is per-joint and residual: it sharpens joint-specific pose evidence while avoiding over-writing the temporal token.

### 3. High-Resolution Heatmap Generation

```text
refined_joint_tokens: [B, L, 17, 128]
Linear seed:           [B*L*17, 64, 8, 8]
Upsample blocks:       8x8 -> 16x16 -> 32x32 -> 64x64
heatmap_logits:        [B, L, 17, 64, 64]
```

Physical meaning:

- The heatmap is a spatial belief distribution for each joint.
- High resolution gives hands, wrists, elbows, and ankles a local peak rather than forcing them into a coarse coordinate vector.
- The progressive 8 -> 16 -> 32 -> 64 decoding path turns compact CSI pose evidence into a spatial field.

Why this design:

- A 64x64 heatmap is high enough for fine keypoint localization while still cheap.
- Starting from an 8x8 seed avoids a massive `Linear(128 -> 64*64)` projection.
- Bilinear upsampling plus convolution is stable and avoids checkerboard artifacts from transposed convolution.
- Per-joint heatmap generation keeps the first decoder simple; skeleton coupling can be added later if real-data errors show swapped or inconsistent limbs.

Implementation comment requirement:

- The code must explain that the seed feature is a coarse spatial hypothesis and each upsample block refines the spatial belief field.

### 4. Differentiable Coordinate Readout

```text
heatmap_logits: [B, L, 17, 64, 64]
softmax over 64*64
heatmaps:       [B, L, 17, 64, 64]
soft-argmax:    [B, L, 17, 2]
```

Physical meaning:

- The heatmap peak represents the most likely joint location.
- Soft-argmax reads the expected x/y position, preserving differentiability.
- Uncertain joints can express broad probability before collapsing to coordinates.

Why this design:

- The dataset currently stores coordinate labels, not dense heatmap labels.
- Soft-argmax allows coordinate loss directly on `[B,L,17,2]` while still forcing the decoder through a high-resolution spatial bottleneck.
- Coordinates are produced directly in `[-0.8, 0.8]`, matching the label preprocessing pipeline.

Implementation comment requirement:

- The code must explain why soft-argmax is used instead of hard argmax: gradients can flow through the heatmap generator.

## File Structure

- Create `models/pose_heatmap_decoder.py`
  - `JointQueryInjection`: creates `[B,L,17,128]` joint tokens.
  - `JointFeatureRefinement`: residual per-joint MLP block.
  - `HeatmapUpsampleBlock`: one 2x spatial refinement block.
  - `JointHeatmapGenerator`: maps joint tokens to `[B,L,17,64,64]` logits.
  - `HeatmapSoftArgmax2D`: maps heatmap logits to coordinates and probabilities.
  - `PoseHeatmapDecoder`: full public decoder.
- Create `tests/test_pose_heatmap_decoder.py`
  - Synthetic tensor tests for every helper and the full decoder.
- Modify `models/__init__.py`
  - Export `PoseHeatmapDecoder`.
- Modify `README.md`
  - Update current architecture status.
- Modify `AGENTS.md`
  - Update current project status and keep training/inference warnings.

---

### Task 1: Heatmap Soft-Argmax Readout

**Files:**
- Create: `tests/test_pose_heatmap_decoder.py`
- Create: `models/pose_heatmap_decoder.py`

- [ ] **Step 1: Write failing tests for coordinate readout**

Create `tests/test_pose_heatmap_decoder.py`:

```python
from __future__ import annotations

import pytest
import torch

from models.pose_heatmap_decoder import HeatmapSoftArgmax2D


def test_soft_argmax_outputs_coords_and_heatmaps():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    coords, heatmaps = readout(logits)

    assert coords.shape == (2, 4, 17, 2)
    assert heatmaps.shape == (2, 4, 17, 64, 64)
    assert torch.isfinite(coords).all()
    assert torch.isfinite(heatmaps).all()


def test_soft_argmax_heatmaps_sum_to_one():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    _, heatmaps = readout(logits)

    probs_sum = heatmaps.flatten(-2).sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_soft_argmax_coordinates_stay_in_target_range():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    coords, _ = readout(logits)

    assert coords.min() >= -0.80001
    assert coords.max() <= 0.80001


def test_soft_argmax_maps_dominant_pixel_to_expected_coordinate():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.full((1, 1, 1, 64, 64), -20.0)
    logits[..., 0, 0] = 20.0

    coords, _ = readout(logits)

    assert torch.allclose(coords[0, 0, 0], torch.tensor([-0.8, -0.8]), atol=1e-3)


def test_soft_argmax_rejects_wrong_shape():
    readout = HeatmapSoftArgmax2D()
    logits = torch.randn(2, 17, 64, 64)

    with pytest.raises(AssertionError, match="expected 5D heatmap logits"):
        readout(logits)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.pose_heatmap_decoder'`.

- [ ] **Step 3: Implement `HeatmapSoftArgmax2D`**

Create `models/pose_heatmap_decoder.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapSoftArgmax2D(nn.Module):
    """Differentiable coordinate readout from 2D joint heatmaps.

    A heatmap lets the decoder represent spatial uncertainty for fine joints.
    Soft-argmax keeps the readout differentiable, unlike hard argmax, so
    coordinate loss can train the heatmap generator end to end.
    """

    def __init__(
        self,
        heatmap_size: int = 64,
        coord_min: float = -0.8,
        coord_max: float = 0.8,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.coord_min = coord_min
        self.coord_max = coord_max

        coords = torch.linspace(coord_min, coord_max, steps=heatmap_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(1, 1, 1, heatmap_size, heatmap_size))
        self.register_buffer("grid_y", grid_y.reshape(1, 1, 1, heatmap_size, heatmap_size))

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.ndim == 5, f"expected 5D heatmap logits [B,L,J,H,W], got ndim={logits.ndim}"
        assert logits.shape[-2:] == (self.heatmap_size, self.heatmap_size), (
            f"expected heatmap size {self.heatmap_size}x{self.heatmap_size}, got {logits.shape[-2:]}"
        )

        b, l, j, h, w = logits.shape
        heatmaps = torch.softmax(logits.reshape(b, l, j, h * w), dim=-1).reshape_as(logits)
        x = (heatmaps * self.grid_x).sum(dim=(-2, -1))
        y = (heatmaps * self.grid_y).sum(dim=(-2, -1))
        coords = torch.stack([x, y], dim=-1)
        return coords, heatmaps
```

- [ ] **Step 4: Run readout tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add models/pose_heatmap_decoder.py tests/test_pose_heatmap_decoder.py
git commit -m "feat: add heatmap soft-argmax readout"
git push
```

---

### Task 2: Joint Query Injection

**Files:**
- Modify: `models/pose_heatmap_decoder.py`
- Modify: `tests/test_pose_heatmap_decoder.py`

- [ ] **Step 1: Add failing tests for joint query injection**

Append to `tests/test_pose_heatmap_decoder.py`:

```python
from models.pose_heatmap_decoder import JointQueryInjection


def test_joint_query_injection_output_shape():
    layer = JointQueryInjection(input_dim=128, num_joints=17)
    temporal_tokens = torch.randn(2, 4, 128)

    joint_tokens = layer(temporal_tokens)

    assert joint_tokens.shape == (2, 4, 17, 128)
    assert torch.isfinite(joint_tokens).all()


def test_joint_query_injection_has_one_embedding_per_joint():
    layer = JointQueryInjection(input_dim=128, num_joints=17)

    assert layer.joint_embedding.shape == (17, 128)


def test_joint_query_injection_rejects_wrong_token_dim():
    layer = JointQueryInjection(input_dim=128, num_joints=17)
    temporal_tokens = torch.randn(2, 4, 64)

    with pytest.raises(AssertionError, match="expected token dim"):
        layer(temporal_tokens)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_joint_query_injection_output_shape tests/test_pose_heatmap_decoder.py::test_joint_query_injection_has_one_embedding_per_joint tests/test_pose_heatmap_decoder.py::test_joint_query_injection_rejects_wrong_token_dim -v
```

Expected: FAIL because `JointQueryInjection` does not exist.

- [ ] **Step 3: Implement `JointQueryInjection`**

Append to `models/pose_heatmap_decoder.py`:

```python
class JointQueryInjection(nn.Module):
    """Convert each frame token into joint-conditioned tokens.

    The temporal token is a whole-body CSI summary. Adding a learned embedding
    per joint asks 17 different localisation questions from that same evidence,
    which protects small joints such as wrists from sharing one undifferentiated
    coordinate regressor.
    """

    def __init__(self, input_dim: int = 128, num_joints: int = 17) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.joint_embedding = nn.Parameter(torch.zeros(num_joints, input_dim))
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def forward(self, temporal_tokens: torch.Tensor) -> torch.Tensor:
        assert temporal_tokens.ndim == 3, (
            f"expected 3D temporal tokens [B,L,D], got ndim={temporal_tokens.ndim}"
        )
        assert temporal_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {temporal_tokens.shape[-1]}"
        )

        return temporal_tokens.unsqueeze(2) + self.joint_embedding.view(1, 1, self.num_joints, self.input_dim)
```

- [ ] **Step 4: Run joint injection tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_joint_query_injection_output_shape tests/test_pose_heatmap_decoder.py::test_joint_query_injection_has_one_embedding_per_joint tests/test_pose_heatmap_decoder.py::test_joint_query_injection_rejects_wrong_token_dim -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add models/pose_heatmap_decoder.py tests/test_pose_heatmap_decoder.py
git commit -m "feat: add joint query injection"
git push
```

---

### Task 3: Joint Feature Refinement

**Files:**
- Modify: `models/pose_heatmap_decoder.py`
- Modify: `tests/test_pose_heatmap_decoder.py`

- [ ] **Step 1: Add failing tests for joint refinement**

Append to `tests/test_pose_heatmap_decoder.py`:

```python
from models.pose_heatmap_decoder import JointFeatureRefinement


def test_joint_feature_refinement_output_shape():
    refine = JointFeatureRefinement(input_dim=128, hidden_dim=128, dropout=0.1)
    joint_tokens = torch.randn(2, 4, 17, 128)

    out = refine(joint_tokens)

    assert out.shape == (2, 4, 17, 128)
    assert torch.isfinite(out).all()


def test_joint_feature_refinement_uses_layernorm_and_residual_mlp():
    refine = JointFeatureRefinement(input_dim=128, hidden_dim=128, dropout=0.1)

    assert isinstance(refine.norm, torch.nn.LayerNorm)
    assert isinstance(refine.mlp[0], torch.nn.Linear)
    assert isinstance(refine.mlp[1], torch.nn.GELU)


def test_joint_feature_refinement_rejects_wrong_shape():
    refine = JointFeatureRefinement(input_dim=128)
    joint_tokens = torch.randn(2, 4, 128)

    with pytest.raises(AssertionError, match="expected 4D joint tokens"):
        refine(joint_tokens)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_output_shape tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_uses_layernorm_and_residual_mlp tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_rejects_wrong_shape -v
```

Expected: FAIL because `JointFeatureRefinement` does not exist.

- [ ] **Step 3: Implement `JointFeatureRefinement`**

Append to `models/pose_heatmap_decoder.py`:

```python
class JointFeatureRefinement(nn.Module):
    """Residual per-joint MLP refinement before spatial decoding.

    The temporal transformer already handled frame-to-frame context. This MLP
    only specializes each joint token, and the residual path keeps the original
    joint-conditioned evidence available for few-shot stability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        assert joint_tokens.ndim == 4, (
            f"expected 4D joint tokens [B,L,J,D], got ndim={joint_tokens.ndim}"
        )
        assert joint_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {joint_tokens.shape[-1]}"
        )
        return joint_tokens + self.mlp(self.norm(joint_tokens))
```

- [ ] **Step 4: Run joint refinement tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_output_shape tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_uses_layernorm_and_residual_mlp tests/test_pose_heatmap_decoder.py::test_joint_feature_refinement_rejects_wrong_shape -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add models/pose_heatmap_decoder.py tests/test_pose_heatmap_decoder.py
git commit -m "feat: add joint feature refinement"
git push
```

---

### Task 4: High-Resolution Joint Heatmap Generator

**Files:**
- Modify: `models/pose_heatmap_decoder.py`
- Modify: `tests/test_pose_heatmap_decoder.py`

- [ ] **Step 1: Add failing tests for heatmap generation**

Append to `tests/test_pose_heatmap_decoder.py`:

```python
from models.pose_heatmap_decoder import HeatmapUpsampleBlock, JointHeatmapGenerator


def test_heatmap_upsample_block_doubles_spatial_resolution():
    block = HeatmapUpsampleBlock(in_channels=64, out_channels=64)
    x = torch.randn(8, 64, 8, 8)

    y = block(x)

    assert y.shape == (8, 64, 16, 16)
    assert torch.isfinite(y).all()


def test_joint_heatmap_generator_output_shape():
    generator = JointHeatmapGenerator(
        input_dim=128,
        decoder_channels=64,
        seed_size=8,
        heatmap_size=64,
    )
    joint_tokens = torch.randn(2, 4, 17, 128)

    logits = generator(joint_tokens)

    assert logits.shape == (2, 4, 17, 64, 64)
    assert torch.isfinite(logits).all()


def test_joint_heatmap_generator_rejects_wrong_shape():
    generator = JointHeatmapGenerator()
    joint_tokens = torch.randn(2, 4, 128)

    with pytest.raises(AssertionError, match="expected 4D joint tokens"):
        generator(joint_tokens)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_heatmap_upsample_block_doubles_spatial_resolution tests/test_pose_heatmap_decoder.py::test_joint_heatmap_generator_output_shape tests/test_pose_heatmap_decoder.py::test_joint_heatmap_generator_rejects_wrong_shape -v
```

Expected: FAIL because `HeatmapUpsampleBlock` and `JointHeatmapGenerator` do not exist.

- [ ] **Step 3: Implement heatmap generation**

Append to `models/pose_heatmap_decoder.py`:

```python
class HeatmapUpsampleBlock(nn.Module):
    """2x spatial upsampling block for heatmap refinement.

    Bilinear upsampling avoids checkerboard artifacts from transposed
    convolution. The following convolution then learns how to sharpen the joint
    belief field at the higher resolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class JointHeatmapGenerator(nn.Module):
    """Decode joint tokens into high-resolution per-joint heatmap logits.

    Each joint token first becomes a coarse 8x8 spatial hypothesis. Progressive
    upsampling refines that hypothesis to 64x64, giving limb endpoints a fine
    spatial field instead of forcing direct coordinate regression.
    """

    def __init__(
        self,
        input_dim: int = 128,
        decoder_channels: int = 64,
        seed_size: int = 8,
        heatmap_size: int = 64,
    ) -> None:
        super().__init__()
        assert heatmap_size == seed_size * 8, (
            f"expected heatmap_size == seed_size * 8 for three upsample blocks, got {heatmap_size} and {seed_size}"
        )
        self.input_dim = input_dim
        self.decoder_channels = decoder_channels
        self.seed_size = seed_size
        self.heatmap_size = heatmap_size

        self.seed = nn.Linear(input_dim, decoder_channels * seed_size * seed_size)
        self.decoder = nn.Sequential(
            HeatmapUpsampleBlock(decoder_channels, decoder_channels),
            HeatmapUpsampleBlock(decoder_channels, decoder_channels // 2),
            HeatmapUpsampleBlock(decoder_channels // 2, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        assert joint_tokens.ndim == 4, (
            f"expected 4D joint tokens [B,L,J,D], got ndim={joint_tokens.ndim}"
        )
        assert joint_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {joint_tokens.shape[-1]}"
        )
        b, l, j, _ = joint_tokens.shape
        x = self.seed(joint_tokens)
        x = x.reshape(b * l * j, self.decoder_channels, self.seed_size, self.seed_size)
        logits = self.decoder(x)
        return logits.reshape(b, l, j, self.heatmap_size, self.heatmap_size)
```

- [ ] **Step 4: Run heatmap generation tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_heatmap_upsample_block_doubles_spatial_resolution tests/test_pose_heatmap_decoder.py::test_joint_heatmap_generator_output_shape tests/test_pose_heatmap_decoder.py::test_joint_heatmap_generator_rejects_wrong_shape -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add models/pose_heatmap_decoder.py tests/test_pose_heatmap_decoder.py
git commit -m "feat: add joint heatmap generator"
git push
```

---

### Task 5: Full PoseHeatmapDecoder

**Files:**
- Modify: `models/pose_heatmap_decoder.py`
- Modify: `tests/test_pose_heatmap_decoder.py`

- [ ] **Step 1: Add failing full decoder tests**

Append to `tests/test_pose_heatmap_decoder.py`:

```python
from models.pose_heatmap_decoder import PoseHeatmapDecoder


def test_pose_heatmap_decoder_outputs_coordinates():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)

    coords = decoder(temporal_tokens)

    assert coords.shape == (2, 4, 17, 2)
    assert torch.isfinite(coords).all()
    assert coords.min() >= -0.80001
    assert coords.max() <= 0.80001


def test_pose_heatmap_decoder_can_return_heatmaps():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)

    coords, aux = decoder(temporal_tokens, return_heatmaps=True)

    assert coords.shape == (2, 4, 17, 2)
    assert aux["heatmap_logits"].shape == (2, 4, 17, 64, 64)
    assert aux["heatmaps"].shape == (2, 4, 17, 64, 64)
    probs_sum = aux["heatmaps"].flatten(-2).sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_pose_heatmap_decoder_rejects_wrong_input_shape():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 128)

    with pytest.raises(AssertionError, match="expected 3D temporal tokens"):
        decoder(temporal_tokens)


def test_pose_heatmap_decoder_parameter_count_is_reasonable():
    decoder = PoseHeatmapDecoder()
    n_params = sum(p.numel() for p in decoder.parameters())
    assert n_params < 750_000, f"expected decoder under 750k params, got {n_params:,}"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_outputs_coordinates tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_can_return_heatmaps tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_rejects_wrong_input_shape tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_parameter_count_is_reasonable -v
```

Expected: FAIL because `PoseHeatmapDecoder` does not exist.

- [ ] **Step 3: Implement `PoseHeatmapDecoder`**

Append to `models/pose_heatmap_decoder.py`:

```python
class PoseHeatmapDecoder(nn.Module):
    """High-resolution pose decoder for temporal CSI frame tokens.

    The decoder intentionally predicts heatmaps before coordinates. This gives
    fine joints such as wrists and elbows a spatial probability field, while the
    soft-argmax readout keeps training compatible with coordinate labels.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_joints: int = 17,
        heatmap_size: int = 64,
        coord_min: float = -0.8,
        coord_max: float = 0.8,
        joint_hidden_dim: int = 128,
        decoder_channels: int = 64,
        seed_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        self.joint_query = JointQueryInjection(input_dim=input_dim, num_joints=num_joints)
        self.refine1 = JointFeatureRefinement(input_dim=input_dim, hidden_dim=joint_hidden_dim, dropout=dropout)
        self.refine2 = JointFeatureRefinement(input_dim=input_dim, hidden_dim=joint_hidden_dim, dropout=dropout)
        self.heatmap_generator = JointHeatmapGenerator(
            input_dim=input_dim,
            decoder_channels=decoder_channels,
            seed_size=seed_size,
            heatmap_size=heatmap_size,
        )
        self.readout = HeatmapSoftArgmax2D(
            heatmap_size=heatmap_size,
            coord_min=coord_min,
            coord_max=coord_max,
        )

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        return_heatmaps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert temporal_tokens.ndim == 3, (
            f"expected 3D temporal tokens [B,L,D], got ndim={temporal_tokens.ndim}"
        )
        assert temporal_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {temporal_tokens.shape[-1]}"
        )

        joint_tokens = self.joint_query(temporal_tokens)
        joint_tokens = self.refine1(joint_tokens)
        joint_tokens = self.refine2(joint_tokens)
        heatmap_logits = self.heatmap_generator(joint_tokens)
        coords, heatmaps = self.readout(heatmap_logits)

        if return_heatmaps:
            return coords, {"heatmap_logits": heatmap_logits, "heatmaps": heatmaps}
        return coords
```

- [ ] **Step 4: Run full decoder tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_outputs_coordinates tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_can_return_heatmaps tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_rejects_wrong_input_shape tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_parameter_count_is_reasonable -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run all decoder tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py -v
```

Expected: all decoder tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/pose_heatmap_decoder.py tests/test_pose_heatmap_decoder.py
git commit -m "feat: add PoseHeatmapDecoder"
git push
```

---

### Task 6: Export and Documentation

**Files:**
- Modify: `models/__init__.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `tests/test_pose_heatmap_decoder.py`

- [ ] **Step 1: Add failing export test**

Append to `tests/test_pose_heatmap_decoder.py`:

```python
def test_pose_heatmap_decoder_is_exported_from_models_package():
    from models import PoseHeatmapDecoder as ExportedPoseHeatmapDecoder

    decoder = ExportedPoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)
    coords = decoder(temporal_tokens)
    assert coords.shape == (2, 4, 17, 2)
```

- [ ] **Step 2: Run export test and verify failure**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_is_exported_from_models_package -v
```

Expected: FAIL with `ImportError` because `PoseHeatmapDecoder` is not exported.

- [ ] **Step 3: Update `models/__init__.py`**

Replace `models/__init__.py` with:

```python
from models.amp_feature_mix_encoder import AmpFeatureMixEncoder
from models.pose_aware_token_projection import PoseAwareTokenProjection
from models.pose_heatmap_decoder import PoseHeatmapDecoder
from models.temporal_lite_transformer import TemporalLiteTransformer

__all__ = [
    "AmpFeatureMixEncoder",
    "PoseAwareTokenProjection",
    "PoseHeatmapDecoder",
    "TemporalLiteTransformer",
]
```

- [ ] **Step 4: Update README architecture**

Update `README.md` implemented modules with:

```markdown
- `models.PoseHeatmapDecoder`: high-resolution heatmap decoder that maps temporal tokens to 17 keypoint coordinates
```

Update current model path:

```text
Temporal tokens:         [B, L, 128]
  -> PoseHeatmapDecoder
Pose coordinates:        [B, L, 17, 2]
```

Update the not-implemented note to keep training/inference/evaluation absent, not the pose head:

```text
Training loops, inference, and evaluation are not implemented yet.
```

- [ ] **Step 5: Update AGENTS status**

Update `AGENTS.md` current model code list with:

```markdown
  - `PoseHeatmapDecoder`: `[B,L,128] -> [B,L,17,2]`.
```

Update implemented architecture line:

```markdown
  `amplitude feature frame -> AmpFeatureMixEncoder -> windowed encoder maps -> PoseAwareTokenProjection -> TemporalLiteTransformer -> PoseHeatmapDecoder -> pose coordinates`.
```

Keep this warning:

```markdown
- Training loops, inference, and evaluation metrics are not implemented yet.
```

- [ ] **Step 6: Run export and docs-relevant tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py::test_pose_heatmap_decoder_is_exported_from_models_package -v
```

Expected: 1 PASS.

- [ ] **Step 7: Commit and push**

Run:

```bash
git add models/__init__.py README.md AGENTS.md tests/test_pose_heatmap_decoder.py
git commit -m "docs: expose PoseHeatmapDecoder architecture"
git push
```

---

### Task 7: Final Verification

**Files:**
- No source changes.

- [ ] **Step 1: Run decoder tests**

Run:

```bash
python -m pytest tests/test_pose_heatmap_decoder.py -v
```

Expected: all decoder tests PASS.

- [ ] **Step 2: Run full project tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Verify git state**

Run:

```bash
git log --oneline -1
git -c core.excludesFile= status --short --branch
```

Expected:

- Latest commit is the docs/export commit or a later explicit cleanup commit.
- Branch is synced with `origin/main`.
- No datasets, checkpoints, caches, local virtual environments, or unrelated untracked files are staged.

---

## Self-Review

- The plan implements the user's confirmed 4-step decoder structure: joint query injection, joint feature refinement, high-resolution heatmap generation, differentiable coordinate readout.
- Each planned layer includes physical meaning and design rationale.
- The public decoder only consumes `[B,L,128]`, preserving the current model boundary.
- The decoder outputs `[B,L,17,2]` coordinates and can optionally return `[B,L,17,64,64]` heatmaps for diagnostics.
- Training loop, inference, loss design, confidence masking, and sliding-window prediction averaging remain out of scope.
- The plan uses TDD task boundaries and synthetic tests only.
