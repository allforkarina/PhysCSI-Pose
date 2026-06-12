import pytest
import torch

from dataset.features import FEATURE_CHANNELS, build_amplitude_features, selected_feature_channels


def make_csi_sequence():
    base = torch.arange(297 * 10 * 3 * 114, dtype=torch.float32).reshape(297, 10, 3, 114)
    return base.remainder(37).add(1.0)


def test_feature_output_shape_and_device():
    csi = make_csi_sequence()
    x = build_amplitude_features(csi)
    assert x.shape == (297, 12, 10, 114)
    assert x.dtype == torch.float32
    assert x.device == csi.device


def test_d_center_has_zero_short_time_mean():
    x = build_amplitude_features(make_csi_sequence())
    d_center = x[:, 3:6].permute(0, 2, 1, 3)
    assert torch.allclose(d_center.mean(dim=1), torch.zeros_like(d_center.mean(dim=1)), atol=1e-5)


def test_c_ant_has_zero_rx_mean():
    x = build_amplitude_features(make_csi_sequence())
    c_ant = x[:, 9:12].permute(0, 2, 1, 3)
    assert torch.allclose(c_ant.mean(dim=2), torch.zeros_like(c_ant.mean(dim=2)), atol=1e-5)


def test_channel_order_uses_feature_blocks_then_rx():
    csi = make_csi_sequence()
    outputs = build_amplitude_features(csi, return_components=True)
    x = outputs.x
    assert torch.allclose(x[:, 0], outputs.l_norm[:, :, 0, :])
    assert torch.allclose(x[:, 1], outputs.l_norm[:, :, 1, :])
    assert torch.allclose(x[:, 2], outputs.l_norm[:, :, 2, :])
    assert torch.allclose(x[:, 3], outputs.d_center[:, :, 0, :])
    assert torch.allclose(x[:, 6], outputs.f_sub[:, :, 0, :])
    assert torch.allclose(x[:, 9], outputs.c_ant[:, :, 0, :])


def test_feature_channel_mapping_for_ablation():
    assert FEATURE_CHANNELS == {
        "l_norm": (0, 1, 2),
        "d_center": (3, 4, 5),
        "f_sub": (6, 7, 8),
        "c_ant": (9, 10, 11),
    }
    assert selected_feature_channels(["l_norm"]) == [0, 1, 2]
    assert selected_feature_channels(["f_sub", "c_ant"]) == [6, 7, 8, 9, 10, 11]


def test_invalid_feature_selection_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        selected_feature_channels(["raw_amp"])


def test_invalid_shape_raises():
    bad = torch.ones(297, 3, 10, 114)
    with pytest.raises(ValueError, match=r"\[297, 10, 3, 114\]"):
        build_amplitude_features(bad)
