from __future__ import annotations

import pytest
import torch

from models import AmpFeatureMixEncoder


def test_encoder_output_shape():
    """Stage 0→4 produce the expected [B,128,10,29] output."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(2, 12, 10, 114)
    z = encoder(x)
    assert z.shape == (2, 128, 10, 29)


def test_encoder_output_finite():
    """Output must contain no NaN or Inf after a forward pass."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(4, 12, 10, 114)
    z = encoder(x)
    assert torch.isfinite(z).all()


def test_different_batch_sizes():
    """Encoder handles batch sizes 1, 2, 8 without shape errors."""
    encoder = AmpFeatureMixEncoder()
    for b in (1, 2, 8):
        x = torch.randn(b, 12, 10, 114)
        z = encoder(x)
        assert z.shape == (b, 128, 10, 29)


def test_input_guard_wrong_channels():
    """Forward raises on wrong channel count."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(2, 8, 10, 114)
    with pytest.raises(AssertionError, match="expected 12 input channels"):
        encoder(x)


def test_encoder_accepts_configurable_input_channels():
    encoder = AmpFeatureMixEncoder(input_channels=6)
    x = torch.randn(2, 6, 10, 114)

    z = encoder(x)

    assert z.shape == (2, 128, 10, 29)


def test_configurable_encoder_guard_reports_expected_channels():
    encoder = AmpFeatureMixEncoder(input_channels=6)
    x = torch.randn(2, 12, 10, 114)

    with pytest.raises(AssertionError, match="expected 6 input channels"):
        encoder(x)


def test_input_guard_wrong_time():
    """Forward raises on wrong time dimension."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(2, 12, 5, 114)
    with pytest.raises(AssertionError, match="expected 10 time steps"):
        encoder(x)


def test_input_guard_wrong_subcarriers():
    """Forward raises on wrong subcarrier dimension."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(2, 12, 10, 64)
    with pytest.raises(AssertionError, match="expected 114 subcarriers"):
        encoder(x)


def test_input_guard_wrong_ndim():
    """Forward raises on 3D input."""
    encoder = AmpFeatureMixEncoder()
    x = torch.randn(12, 10, 114)
    with pytest.raises(AssertionError, match="expected 4D input"):
        encoder(x)


def test_parameter_count_reasonable():
    """Encoder is lightweight — under 200k parameters."""
    encoder = AmpFeatureMixEncoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    assert n_params < 200_000, f"expected <200k params, got {n_params:,}"
