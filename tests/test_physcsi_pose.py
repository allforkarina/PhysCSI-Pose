from __future__ import annotations

import torch

from models import PhysCSIPoseNet


def test_physcsi_pose_net_outputs_window_coordinates():
    model = PhysCSIPoseNet(input_channels=12, token_dim=128, num_joints=17)
    x = torch.randn(2, 4, 12, 10, 114)

    pred = model(x)

    assert pred.shape == (2, 4, 17, 2)
    assert torch.isfinite(pred).all()
    assert pred.min() >= -0.80001
    assert pred.max() <= 0.80001


def test_physcsi_pose_net_accepts_feature_ablation_channels():
    model = PhysCSIPoseNet(input_channels=6, token_dim=128, num_joints=17)
    x = torch.randn(2, 4, 6, 10, 114)

    pred = model(x)

    assert pred.shape == (2, 4, 17, 2)


def test_physcsi_pose_net_returns_auxiliary_outputs():
    model = PhysCSIPoseNet(input_channels=12, token_dim=128, num_joints=17)
    x = torch.randn(1, 4, 12, 10, 114)

    pred, aux = model(x, return_aux=True)

    assert pred.shape == (1, 4, 17, 2)
    assert aux["encoder_maps"].shape == (1, 4, 128, 10, 29)
    assert aux["tokens"].shape == (1, 4, 128)
    assert aux["temporal_tokens"].shape == (1, 4, 128)
