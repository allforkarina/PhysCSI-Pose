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
