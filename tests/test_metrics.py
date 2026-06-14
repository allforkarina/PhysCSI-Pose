from __future__ import annotations

import torch

from engine.metrics import aggregate_window_predictions_mean, compute_pose_metrics


def test_compute_pose_metrics_uses_only_positive_confidence_as_valid():
    pred = torch.tensor([[[[0.0, 0.0], [10.0, 10.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    conf = torch.tensor([[[1.0, 0.0]]])

    metrics = compute_pose_metrics(pred, target, conf, pck_thresholds=(0.05, 0.10, 0.20, 0.50))

    assert metrics["mpjpe_norm"] == 0.0
    assert metrics["pck_0.05"] == 1.0
    assert metrics["pck_0.10"] == 1.0
    assert metrics["pck_0.20"] == 1.0
    assert metrics["pck_0.50"] == 1.0
    assert metrics["per_joint_pck_0.05"] == [1.0, 0.0]


def test_compute_pose_metrics_counts_zero_zero_coordinate_as_valid():
    pred = torch.tensor([[[[0.03, 0.04], [0.30, 0.40]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    conf = torch.tensor([[[1.0, 1.0]]])

    metrics = compute_pose_metrics(pred, target, conf, pck_thresholds=(0.05, 0.10, 0.20, 0.50))

    assert torch.isclose(torch.tensor(metrics["mpjpe_norm"]), torch.tensor(0.275))
    assert metrics["pck_0.05"] == 0.5
    assert metrics["pck_0.10"] == 0.5
    assert metrics["pck_0.20"] == 0.5
    assert metrics["pck_0.50"] == 1.0
    assert metrics["per_joint_pck_0.50"] == [1.0, 1.0]


def test_compute_pose_metrics_reports_prediction_and_ground_truth_joint_std():
    pred = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[2.0, 0.0], [1.0, 3.0]]],
        ]
    )
    target = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 4.0], [5.0, 1.0]]],
        ]
    )
    conf = torch.ones(2, 1, 2)

    metrics = compute_pose_metrics(pred, target, conf, pck_thresholds=(0.50,))

    assert torch.allclose(torch.tensor(metrics["per_joint_std"]), torch.tensor([1.0, 1.0]))
    assert metrics["mean_joint_std"] == 1.0
    assert metrics["min_joint_std"] == 1.0
    assert torch.allclose(torch.tensor(metrics["gt_per_joint_std"]), torch.tensor([2.0, 2.0]))
    assert metrics["gt_mean_joint_std"] == 2.0
    assert metrics["gt_min_joint_std"] == 2.0


def test_aggregate_window_predictions_mean_averages_overlapping_global_indices():
    pred = torch.tensor(
        [
            [[[1.0, 1.0]], [[3.0, 3.0]]],
            [[[5.0, 5.0]], [[7.0, 7.0]]],
        ]
    )
    global_idx = torch.tensor([[10, 11], [11, 12]])

    aggregated, unique_idx = aggregate_window_predictions_mean(pred, global_idx)

    assert unique_idx.tolist() == [10, 11, 12]
    assert torch.allclose(
        aggregated,
        torch.tensor([[[1.0, 1.0]], [[4.0, 4.0]], [[7.0, 7.0]]]),
    )
