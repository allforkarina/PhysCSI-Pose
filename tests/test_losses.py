from __future__ import annotations

import torch

from engine.losses import masked_smooth_l1_loss


def test_masked_smooth_l1_uses_only_positive_confidence_as_valid():
    pred = torch.tensor([[[[1.0, 0.0], [10.0, 10.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    conf = torch.tensor([[[1.0, 0.0]]])

    loss = masked_smooth_l1_loss(pred, target, conf, beta=1.0)

    assert torch.isclose(loss, torch.tensor(0.25))


def test_masked_smooth_l1_treats_zero_zero_target_as_valid_when_conf_positive():
    pred = torch.tensor([[[[0.5, -0.5]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    conf = torch.tensor([[[1.0]]])

    loss = masked_smooth_l1_loss(pred, target, conf, beta=1.0)

    assert torch.isclose(loss, torch.tensor(0.125))


def test_masked_smooth_l1_returns_zero_when_no_labels_are_valid():
    pred = torch.tensor([[[[5.0, 5.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    conf = torch.tensor([[[0.0]]])

    loss = masked_smooth_l1_loss(pred, target, conf)

    assert torch.isclose(loss, torch.tensor(0.0))
