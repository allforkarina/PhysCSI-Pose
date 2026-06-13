# TemporalLiteTransformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TemporalLiteTransformer`, the Step 3 Temporal Relation Module that maps pose-aware frame tokens `[B, L, 128]` to temporally enhanced tokens `[B, L, 128]` for short windows where `4 <= L <= 8`.

**Architecture:** The module uses local depthwise Conv1d positional encoding, two Pre-Norm Lite Transformer blocks, 4-head non-causal self-attention with learnable relative temporal bias, and a lightweight FFN with expansion ratio 2. It preserves one output token per input frame and does not model across windows or full 297-frame action sequences.

**Tech Stack:** PyTorch (`nn.Module`, `Conv1d`, `Linear`, `LayerNorm`, `GELU`, `Dropout`, `ModuleList`, tensor reshape/einsum-style matmul), pytest, synthetic tensor tests only.

---

## Scope

In scope:

- Add `TemporalConvPosEncoding`.
- Add `RelativeTemporalSelfAttention`.
- Add `LiteTransformerBlock`.
- Add `TemporalLiteTransformer`.
- Export `TemporalLiteTransformer` from `models`.
- Add synthetic unit tests for shape, supported window lengths, invalid inputs, relative-bias attention, aux attention output, finite outputs, and parameter count.
- Update README/AGENTS status after the module is implemented.

Out of scope for this plan:

- Window-sampling Dataset changes.
- Source-domain or target-domain training loops.
- Pose Regression Head.
- Sliding-window inference and prediction averaging.
- Causal/real-time temporal inference.
- Padding masks for mixed window lengths in the same batch.

## File Structure

- Create `models/temporal_lite_transformer.py`
  - `TemporalConvPosEncoding`: depthwise Conv1d local temporal positional encoding, shape preserving.
  - `RelativeTemporalSelfAttention`: non-causal self-attention over `L` frames with learnable per-head relative temporal bias.
  - `LiteTransformerBlock`: Pre-Norm block with CPE, MHSA residual, and FFN residual.
  - `TemporalLiteTransformer`: two-block stack with input guards and optional attention-map return.
- Create `tests/test_temporal_lite_transformer.py`
  - Unit tests using synthetic tensors only.
- Modify `models/__init__.py`
  - Export `TemporalLiteTransformer`.
- Modify `README.md`
  - Update current architecture path to include the temporal module.
- Modify `AGENTS.md`
  - Update current project status while keeping training/inference warnings.

---

### Task 1: Skeleton Module and Interface Tests

**Files:**
- Create: `tests/test_temporal_lite_transformer.py`
- Create: `models/temporal_lite_transformer.py`

- [ ] **Step 1: Write the failing interface tests**

Create `tests/test_temporal_lite_transformer.py`:

```python
from __future__ import annotations

import pytest
import torch

from models.temporal_lite_transformer import TemporalLiteTransformer


def test_temporal_transformer_output_shape_default():
    model = TemporalLiteTransformer()
    x = torch.randn(2, 4, 128)
    y = model(x)
    assert y.shape == (2, 4, 128)
    assert y.dtype == torch.float32


def test_temporal_transformer_supports_window_lengths_4_to_8():
    model = TemporalLiteTransformer()
    for window_length in (4, 5, 6, 7, 8):
        x = torch.randn(2, window_length, 128)
        y = model(x)
        assert y.shape == (2, window_length, 128)


def test_temporal_transformer_rejects_short_window():
    model = TemporalLiteTransformer()
    x = torch.randn(2, 3, 128)
    with pytest.raises(AssertionError, match="expected window length"):
        model(x)


def test_temporal_transformer_rejects_long_window():
    model = TemporalLiteTransformer()
    x = torch.randn(2, 9, 128)
    with pytest.raises(AssertionError, match="expected window length"):
        model(x)


def test_temporal_transformer_rejects_wrong_token_dim():
    model = TemporalLiteTransformer()
    x = torch.randn(2, 4, 64)
    with pytest.raises(AssertionError, match="expected token dim"):
        model(x)


def test_temporal_transformer_rejects_wrong_ndim():
    model = TemporalLiteTransformer()
    x = torch.randn(4, 128)
    with pytest.raises(AssertionError, match="expected 3D input"):
        model(x)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.temporal_lite_transformer'`.

- [ ] **Step 3: Add the minimal skeleton module**

Create `models/temporal_lite_transformer.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalLiteTransformer(nn.Module):
    """Short-window temporal relation module.

    Input:  [B, L, 128], where 4 <= L <= 8
    Output: [B, L, 128]
    """

    def __init__(
        self,
        input_dim: int = 128,
        min_window_length: int = 4,
        max_window_length: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"expected 3D input [B,L,D], got ndim={x.ndim}"
        _, window_length, token_dim = x.shape
        assert token_dim == self.input_dim, f"expected token dim {self.input_dim}, got {token_dim}"
        assert self.min_window_length <= window_length <= self.max_window_length, (
            f"expected window length in [{self.min_window_length}, {self.max_window_length}], "
            f"got {window_length}"
        )
        return x
```

- [ ] **Step 4: Run the interface tests**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit and push**

Run:

```bash
git add models/temporal_lite_transformer.py tests/test_temporal_lite_transformer.py
git commit -m "test: add TemporalLiteTransformer interface skeleton"
git push
```

---

### Task 2: Local Temporal Convolutional Positional Encoding

**Files:**
- Modify: `models/temporal_lite_transformer.py`
- Modify: `tests/test_temporal_lite_transformer.py`

- [ ] **Step 1: Add failing CPE tests**

Append to `tests/test_temporal_lite_transformer.py`:

```python
from models.temporal_lite_transformer import TemporalConvPosEncoding


def test_temporal_conv_pos_encoding_shape_and_finiteness():
    pos = TemporalConvPosEncoding(input_dim=128)
    x = torch.randn(2, 4, 128)
    y = pos(x)
    assert y.shape == (2, 4, 128)
    assert torch.isfinite(y).all()


def test_temporal_conv_pos_encoding_is_depthwise_conv1d():
    pos = TemporalConvPosEncoding(input_dim=128)
    assert pos.dwconv.in_channels == 128
    assert pos.dwconv.out_channels == 128
    assert pos.dwconv.groups == 128
    assert pos.dwconv.kernel_size == (3,)
    assert pos.dwconv.padding == (1,)
```

- [ ] **Step 2: Run CPE tests to verify they fail**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_conv_pos_encoding_shape_and_finiteness tests/test_temporal_lite_transformer.py::test_temporal_conv_pos_encoding_is_depthwise_conv1d -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `TemporalConvPosEncoding` does not exist.

- [ ] **Step 3: Add `TemporalConvPosEncoding`**

Insert this class above `TemporalLiteTransformer` in `models/temporal_lite_transformer.py`:

```python
class TemporalConvPosEncoding(nn.Module):
    """Local temporal convolutional positional encoding.

    Uses depthwise Conv1d over the window axis and residual addition.

    Input:  [B, L, D]
    Output: [B, L, D]
    """

    def __init__(self, input_dim: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"expected odd kernel_size, got {kernel_size}"
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=input_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        return residual + x
```

- [ ] **Step 4: Run the CPE tests**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_conv_pos_encoding_shape_and_finiteness tests/test_temporal_lite_transformer.py::test_temporal_conv_pos_encoding_is_depthwise_conv1d -v
```

Expected: 2 PASS.

- [ ] **Step 5: Run the full temporal module test file**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/temporal_lite_transformer.py tests/test_temporal_lite_transformer.py
git commit -m "feat: add temporal convolutional positional encoding"
git push
```

---

### Task 3: Relative Temporal Self-Attention

**Files:**
- Modify: `models/temporal_lite_transformer.py`
- Modify: `tests/test_temporal_lite_transformer.py`

- [ ] **Step 1: Add failing attention tests**

Append to `tests/test_temporal_lite_transformer.py`:

```python
from models.temporal_lite_transformer import RelativeTemporalSelfAttention


def test_relative_temporal_attention_output_shape():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(2, 4, 128)
    y = attn(x)
    assert y.shape == (2, 4, 128)
    assert torch.isfinite(y).all()


def test_relative_temporal_attention_returns_attention_maps():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(2, 6, 128)
    y, aux = attn(x, return_attention=True)
    assert y.shape == (2, 6, 128)
    assert aux["attention"].shape == (2, 4, 6, 6)
    attn_sum = aux["attention"].sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)


def test_relative_temporal_bias_table_shape():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    assert attn.relative_bias.shape == (4, 15)
    bias = attn.relative_bias_for_length(window_length=4, device=torch.device("cpu"))
    assert bias.shape == (4, 4, 4)


def test_relative_temporal_attention_rejects_bad_head_config():
    with pytest.raises(AssertionError, match="divisible by num_heads"):
        RelativeTemporalSelfAttention(input_dim=130, num_heads=4, max_window_length=8)
```

- [ ] **Step 2: Run attention tests to verify they fail**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_output_shape tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_returns_attention_maps tests/test_temporal_lite_transformer.py::test_relative_temporal_bias_table_shape tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_rejects_bad_head_config -v
```

Expected: FAIL because `RelativeTemporalSelfAttention` does not exist.

- [ ] **Step 3: Add `RelativeTemporalSelfAttention`**

Insert this class below `TemporalConvPosEncoding` in `models/temporal_lite_transformer.py`:

```python
class RelativeTemporalSelfAttention(nn.Module):
    """Non-causal multi-head self-attention over short temporal windows."""

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        max_window_length: int = 8,
        attention_dropout: float = 0.1,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.max_window_length = max_window_length
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_window_length - 1))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def relative_bias_for_length(self, window_length: int, device: torch.device) -> torch.Tensor:
        assert window_length <= self.max_window_length, (
            f"expected window_length <= {self.max_window_length}, got {window_length}"
        )
        positions = torch.arange(window_length, device=device)
        relative_distance = positions[None, :] - positions[:, None]
        bias_index = relative_distance + (self.max_window_length - 1)
        return self.relative_bias[:, bias_index]

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, window_length, token_dim = x.shape
        assert token_dim == self.input_dim, f"expected token dim {self.input_dim}, got {token_dim}"
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, window_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        relative_bias = self.relative_bias_for_length(window_length, x.device)
        score = score + relative_bias.unsqueeze(0)

        attention = torch.softmax(score, dim=-1)
        attention_for_context = self.attn_drop(attention)
        context = torch.matmul(attention_for_context, v)
        context = context.transpose(1, 2).reshape(batch_size, window_length, token_dim)
        out = self.proj(context)
        out = self.proj_drop(out)

        if return_attention:
            return out, {"attention": attention}
        return out
```

- [ ] **Step 4: Run the attention tests**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_output_shape tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_returns_attention_maps tests/test_temporal_lite_transformer.py::test_relative_temporal_bias_table_shape tests/test_temporal_lite_transformer.py::test_relative_temporal_attention_rejects_bad_head_config -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run the full temporal module test file**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/temporal_lite_transformer.py tests/test_temporal_lite_transformer.py
git commit -m "feat: add relative temporal self attention"
git push
```

---

### Task 4: Lite Transformer Block

**Files:**
- Modify: `models/temporal_lite_transformer.py`
- Modify: `tests/test_temporal_lite_transformer.py`

- [ ] **Step 1: Add failing block tests**

Append to `tests/test_temporal_lite_transformer.py`:

```python
from models.temporal_lite_transformer import LiteTransformerBlock


def test_lite_transformer_block_output_shape_and_finiteness():
    block = LiteTransformerBlock(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(2, 4, 128)
    y = block(x)
    assert y.shape == (2, 4, 128)
    assert torch.isfinite(y).all()


def test_lite_transformer_block_returns_attention_map():
    block = LiteTransformerBlock(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(2, 5, 128)
    y, aux = block(x, return_attention=True)
    assert y.shape == (2, 5, 128)
    assert aux["attention"].shape == (2, 4, 5, 5)


def test_lite_transformer_block_uses_prenorm_components():
    block = LiteTransformerBlock(input_dim=128, num_heads=4, max_window_length=8)
    assert isinstance(block.norm_attn, torch.nn.LayerNorm)
    assert isinstance(block.norm_ffn, torch.nn.LayerNorm)
    assert isinstance(block.pos_encoding, TemporalConvPosEncoding)
```

- [ ] **Step 2: Run block tests to verify they fail**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_lite_transformer_block_output_shape_and_finiteness tests/test_temporal_lite_transformer.py::test_lite_transformer_block_returns_attention_map tests/test_temporal_lite_transformer.py::test_lite_transformer_block_uses_prenorm_components -v
```

Expected: FAIL because `LiteTransformerBlock` does not exist.

- [ ] **Step 3: Add `LiteTransformerBlock`**

Insert this class below `RelativeTemporalSelfAttention` in `models/temporal_lite_transformer.py`:

```python
class LiteTransformerBlock(nn.Module):
    """Pre-Norm lightweight temporal Transformer block."""

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        max_window_length: int = 8,
        ffn_expansion: int = 2,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = input_dim * ffn_expansion
        self.pos_encoding = TemporalConvPosEncoding(input_dim=input_dim)
        self.norm_attn = nn.LayerNorm(input_dim)
        self.attn = RelativeTemporalSelfAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            max_window_length=max_window_length,
            attention_dropout=attention_dropout,
            projection_dropout=0.0,
        )
        self.attn_resid_drop = nn.Dropout(residual_dropout)
        self.norm_ffn = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(residual_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.pos_encoding(x)
        if return_attention:
            attn_out, aux = self.attn(self.norm_attn(x), return_attention=True)
        else:
            attn_out = self.attn(self.norm_attn(x), return_attention=False)
            aux = {}
        x = x + self.attn_resid_drop(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        if return_attention:
            return x, aux
        return x
```

- [ ] **Step 4: Run the block tests**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_lite_transformer_block_output_shape_and_finiteness tests/test_temporal_lite_transformer.py::test_lite_transformer_block_returns_attention_map tests/test_temporal_lite_transformer.py::test_lite_transformer_block_uses_prenorm_components -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run the full temporal module test file**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/temporal_lite_transformer.py tests/test_temporal_lite_transformer.py
git commit -m "feat: add lite temporal transformer block"
git push
```

---

### Task 5: Full `TemporalLiteTransformer` Stack

**Files:**
- Modify: `models/temporal_lite_transformer.py`
- Modify: `tests/test_temporal_lite_transformer.py`

- [ ] **Step 1: Add failing full-stack tests**

Append to `tests/test_temporal_lite_transformer.py`:

```python
def test_temporal_transformer_return_attention_shapes():
    model = TemporalLiteTransformer()
    x = torch.randn(2, 4, 128)
    y, aux = model(x, return_attention=True)
    assert y.shape == (2, 4, 128)
    assert aux["attention_maps"].shape == (2, 2, 4, 4, 4)
    attn_sum = aux["attention_maps"].sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)


def test_temporal_transformer_default_configuration():
    model = TemporalLiteTransformer()
    assert model.input_dim == 128
    assert model.min_window_length == 4
    assert model.max_window_length == 8
    assert len(model.layers) == 2
    assert model.num_heads == 4


def test_temporal_transformer_output_is_finite_for_supported_windows():
    model = TemporalLiteTransformer()
    for window_length in (4, 6, 8):
        x = torch.randn(3, window_length, 128)
        y = model(x)
        assert torch.isfinite(y).all()


def test_temporal_transformer_parameter_count_is_lightweight():
    model = TemporalLiteTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 250_000, f"expected <250k params, got {n_params:,}"
```

- [ ] **Step 2: Run full-stack tests to verify they fail against the skeleton**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_transformer_return_attention_shapes tests/test_temporal_lite_transformer.py::test_temporal_transformer_default_configuration tests/test_temporal_lite_transformer.py::test_temporal_transformer_output_is_finite_for_supported_windows tests/test_temporal_lite_transformer.py::test_temporal_transformer_parameter_count_is_lightweight -v
```

Expected: FAIL because the skeleton has no transformer layers and does not support `return_attention`.

- [ ] **Step 3: Replace `TemporalLiteTransformer` with the full stack**

Replace the existing `TemporalLiteTransformer` class in `models/temporal_lite_transformer.py` with:

```python
class TemporalLiteTransformer(nn.Module):
    """Short-window temporal relation module.

    Input:  [B, L, 128], where 4 <= L <= 8
    Output: [B, L, 128]
    """

    def __init__(
        self,
        input_dim: int = 128,
        min_window_length: int = 4,
        max_window_length: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion: int = 2,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert min_window_length >= 1, f"expected min_window_length >= 1, got {min_window_length}"
        assert max_window_length >= min_window_length, (
            f"expected max_window_length >= min_window_length, got {max_window_length} < {min_window_length}"
        )
        self.input_dim = input_dim
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                LiteTransformerBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    max_window_length=max_window_length,
                    ffn_expansion=ffn_expansion,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.ndim == 3, f"expected 3D input [B,L,D], got ndim={x.ndim}"
        _, window_length, token_dim = x.shape
        assert token_dim == self.input_dim, f"expected token dim {self.input_dim}, got {token_dim}"
        assert self.min_window_length <= window_length <= self.max_window_length, (
            f"expected window length in [{self.min_window_length}, {self.max_window_length}], "
            f"got {window_length}"
        )

        attention_maps = []
        for layer in self.layers:
            if return_attention:
                x, aux = layer(x, return_attention=True)
                attention_maps.append(aux["attention"])
            else:
                x = layer(x, return_attention=False)

        if return_attention:
            aux = {"attention_maps": torch.stack(attention_maps, dim=1)}
            return x, aux
        return x
```

- [ ] **Step 4: Run the full-stack tests**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_transformer_return_attention_shapes tests/test_temporal_lite_transformer.py::test_temporal_transformer_default_configuration tests/test_temporal_lite_transformer.py::test_temporal_transformer_output_is_finite_for_supported_windows tests/test_temporal_lite_transformer.py::test_temporal_transformer_parameter_count_is_lightweight -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run the full temporal test file**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py -v
```

Expected: all temporal tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/temporal_lite_transformer.py tests/test_temporal_lite_transformer.py
git commit -m "feat: add full TemporalLiteTransformer stack"
git push
```

---

### Task 6: Export From `models`

**Files:**
- Modify: `models/__init__.py`
- Modify: `tests/test_temporal_lite_transformer.py`

- [ ] **Step 1: Add failing package export test**

Append to `tests/test_temporal_lite_transformer.py`:

```python
def test_temporal_transformer_is_exported_from_models_package():
    from models import TemporalLiteTransformer as ExportedTemporalLiteTransformer

    model = ExportedTemporalLiteTransformer()
    x = torch.randn(2, 4, 128)
    y = model(x)
    assert y.shape == (2, 4, 128)
```

- [ ] **Step 2: Run the export test to verify it fails**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_transformer_is_exported_from_models_package -v
```

Expected: FAIL with `ImportError` because `TemporalLiteTransformer` is not exported from `models`.

- [ ] **Step 3: Update `models/__init__.py`**

Replace `models/__init__.py` with:

```python
from models.amp_feature_mix_encoder import AmpFeatureMixEncoder
from models.pose_aware_token_projection import PoseAwareTokenProjection
from models.temporal_lite_transformer import TemporalLiteTransformer

__all__ = ["AmpFeatureMixEncoder", "PoseAwareTokenProjection", "TemporalLiteTransformer"]
```

- [ ] **Step 4: Run the export test**

Run:

```bash
python -m pytest tests/test_temporal_lite_transformer.py::test_temporal_transformer_is_exported_from_models_package -v
```

Expected: 1 PASS.

- [ ] **Step 5: Run all model tests**

Run:

```bash
python -m pytest tests/test_amp_feature_mix_encoder.py tests/test_pose_aware_token_projection.py tests/test_temporal_lite_transformer.py -v
```

Expected: all model tests PASS.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add models/__init__.py tests/test_temporal_lite_transformer.py
git commit -m "feat: export TemporalLiteTransformer from models package"
git push
```

---

### Task 7: Documentation Sync

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update README architecture status**

In `README.md`, update the implemented modules list to include:

```markdown
- `models.TemporalLiteTransformer`: models local temporal relations over short frame-token windows.
```

Update the current model path to:

```text
X frame features:        [B, 12, 10, 114]
  -> AmpFeatureMixEncoder
Encoder map:             [B, 128, 10, 29]
  -> PoseAwareTokenProjection over L-frame windows
Pose-aware tokens:       [B, L, 128]
  -> TemporalLiteTransformer
Temporal tokens:         [B, L, 128]
```

Update the remaining model path to:

```text
Temporal tokens [B, L, 128]
  -> Pose Regression Head          # not implemented yet
  -> 17 keypoints                  # not implemented yet
```

Add this paragraph near the model descriptions:

```markdown
`TemporalLiteTransformer` uses local depthwise Conv1d positional encoding, two
Pre-Norm lightweight Transformer blocks, 4-head non-causal self-attention with
learnable relative temporal bias, and a 2x FFN. It supports short windows where
`4 <= L <= 8` and preserves one output token per input frame.
```

- [ ] **Step 2: Update AGENTS current status**

In `AGENTS.md`, update the current model code list to include:

```markdown
  - `TemporalLiteTransformer`: `[B,L,128] -> [B,L,128]`.
```

Update the implemented architecture line to:

```markdown
  `amplitude feature frame -> AmpFeatureMixEncoder -> windowed encoder maps -> PoseAwareTokenProjection -> TemporalLiteTransformer -> temporal frame tokens`.
```

Update the not-implemented status line to:

```markdown
- Final pose regression heads, training loops, inference, and evaluation metrics are not implemented yet.
```

- [ ] **Step 3: Inspect the docs diff**

Run:

```bash
git diff -- README.md AGENTS.md
```

Expected: docs describe the temporal module as implemented, while still making clear that pose head/training/inference/evaluation are absent.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add README.md AGENTS.md
git commit -m "docs: sync TemporalLiteTransformer architecture status"
git push
```

---

### Task 8: Final Verification

**Files:**
- No source changes.

- [ ] **Step 1: Run the full project test suite**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify latest commit and branch sync**

Run:

```bash
git log --oneline -1
git -c core.excludesFile= status --short --branch
```

Expected:

- Latest commit is the docs sync commit from Task 7 or a later explicit final cleanup commit.
- Branch is synced with `origin/main`.
- No unintended data, checkpoints, generated caches, or local-only files are staged.

- [ ] **Step 3: Record the implemented architecture in the final response**

Final response should state:

```text
[B,12,10,114]
  -> AmpFeatureMixEncoder
[B,128,10,29]
  -> PoseAwareTokenProjection over L-frame windows
[B,L,128]
  -> TemporalLiteTransformer
[B,L,128]
```

Also state that `Pose Regression Head`, training loops, inference, and evaluation are still not implemented.

---

## Self-Review

- Spec coverage: The plan implements short-window temporal modelling, `4 <= L <= 8`, `D=128`, 2 layers, 4 heads, head_dim 32, FFN expansion 2, dropout 0.1, non-causal attention, depthwise Conv1d positional encoding, relative temporal bias, residual Pre-Norm blocks, and all-frames token preservation.
- Scope boundary: Window sampling, few-shot sampling strategy, pose supervision, sliding-window inference, and prediction averaging are intentionally out of scope because they require Dataset/training/inference code, not just the temporal model architecture.
- Type consistency: The public class name is `TemporalLiteTransformer`, matching the user's requested module name. Internal helper class names are stable across tasks.
- Red-flag placeholder scan: passed.
