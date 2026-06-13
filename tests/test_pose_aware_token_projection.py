from __future__ import annotations

import pytest
import torch

from models.pose_aware_token_projection import PoseAwareTokenProjection


def test_output_shape_default():
    proj = PoseAwareTokenProjection()
    z = torch.randn(2, 32, 128, 10, 29)
    h = proj(z)
    assert h.shape == (2, 32, 128)
    assert h.dtype == torch.float32


def test_output_shape_variable_window():
    proj = PoseAwareTokenProjection()
    for L in (1, 16, 64):
        z = torch.randn(2, L, 128, 10, 29)
        h = proj(z)
        assert h.shape == (2, L, 128)


def test_return_attention_shapes():
    proj = PoseAwareTokenProjection()
    z = torch.randn(2, 32, 128, 10, 29)
    h, aux = proj(z, return_attention=True)
    assert h.shape == (2, 32, 128)
    assert aux["attention_maps"].shape == (2, 32, 4, 10, 29)
    assert aux["h_avg"].shape == (2, 32, 128)
    assert aux["h_res_multi"].shape == (2, 32, 4, 128)


def test_attention_softmax_sums_to_one():
    proj = PoseAwareTokenProjection()
    z = torch.randn(2, 32, 128, 10, 29)
    _, aux = proj(z, return_attention=True)
    alpha = aux["attention_maps"]  # [B, L, K, T, S]
    alpha_sum = alpha.flatten(-2).sum(-1)  # [B, L, K]
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)


def test_output_finite():
    proj = PoseAwareTokenProjection()
    z = torch.randn(4, 16, 128, 10, 29)
    h = proj(z)
    assert torch.isfinite(h).all()


def test_attention_maps_finite():
    proj = PoseAwareTokenProjection()
    z = torch.randn(2, 8, 128, 10, 29)
    _, aux = proj(z, return_attention=True)
    assert torch.isfinite(aux["attention_maps"]).all()
