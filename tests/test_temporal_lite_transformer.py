from __future__ import annotations

import pytest
import torch

from models.temporal_lite_transformer import (
    LiteTransformerBlock,
    RelativeTemporalSelfAttention,
    TemporalConvPosEncoding,
    TemporalLiteTransformer,
)


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


def test_relative_temporal_attention_output_shape():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(2, 4, 128)
    y = attn(x)
    assert y.shape == (2, 4, 128)
    assert torch.isfinite(y).all()


def test_relative_temporal_attention_uses_linear_output_projection():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    assert isinstance(attn.proj, torch.nn.Linear)
    assert attn.proj.in_features == 128
    assert attn.proj.out_features == 128


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


def test_relative_temporal_bias_indexes_follow_temporal_offset_mapping():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=2, max_window_length=8)
    with torch.no_grad():
        index_values = torch.arange(attn.relative_bias.shape[1], dtype=attn.relative_bias.dtype)
        attn.relative_bias.copy_(index_values.repeat(attn.num_heads, 1))

    bias = attn.relative_bias_for_length(window_length=4, device=torch.device("cpu"))

    positions = torch.arange(4)
    expected = positions[None, :] - positions[:, None] + attn.max_window_length - 1
    assert torch.equal(bias, expected.expand(attn.num_heads, -1, -1))


def test_relative_temporal_attention_rejects_wrong_ndim():
    attn = RelativeTemporalSelfAttention(input_dim=128, num_heads=4, max_window_length=8)
    x = torch.randn(4, 128)
    with pytest.raises(AssertionError, match="expected 3D input"):
        attn(x)


def test_relative_temporal_attention_rejects_bad_head_config():
    with pytest.raises(AssertionError, match="divisible by num_heads"):
        RelativeTemporalSelfAttention(input_dim=130, num_heads=4, max_window_length=8)


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


def test_temporal_transformer_parameter_count_is_still_lightweight():
    model = TemporalLiteTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 280_000, f"expected lightweight model with <280k params, got {n_params:,}"


def test_temporal_transformer_is_exported_from_models_package():
    from models import TemporalLiteTransformer as ExportedTemporalLiteTransformer

    model = ExportedTemporalLiteTransformer()
    x = torch.randn(2, 4, 128)
    y = model(x)
    assert y.shape == (2, 4, 128)
