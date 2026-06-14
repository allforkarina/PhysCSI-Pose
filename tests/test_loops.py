from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from engine.loops import evaluate_one_epoch, train_one_epoch


class TinyWindowDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.ones(4, 3, 10, 114),
            "y": torch.zeros(4, 17, 2),
            "conf": torch.ones(4, 17),
            "global_idx": torch.arange(idx * 4, idx * 4 + 4),
        }


class TinyPoseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(x.shape[0], x.shape[1], 17, 2)


def test_train_one_epoch_updates_model():
    model = TinyPoseModel()
    loader = DataLoader(TinyWindowDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = model.bias.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
        grad_clip_norm=1.0,
    )

    assert metrics["train_loss"] > 0
    assert not torch.allclose(model.bias.detach(), before)


def test_evaluate_one_epoch_returns_metrics():
    model = TinyPoseModel()
    loader = DataLoader(TinyWindowDataset(), batch_size=2)

    metrics = evaluate_one_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        amp_enabled=False,
        pck_thresholds=(0.05, 0.10),
    )

    assert "val_loss" in metrics
    assert "mpjpe_norm" in metrics
    assert "pck_0.05" in metrics
    assert "mean_joint_std" in metrics


class OverlappingPredictionDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.zeros(2, 1, 10, 114)
        if idx == 1:
            x[0] = 2.0
        return {
            "x": x,
            "y": torch.zeros(2, 17, 2),
            "conf": torch.ones(2, 17),
            "global_idx": torch.tensor([0, 1]) if idx == 0 else torch.tensor([1, 2]),
        }


class InputEncodedPoseModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = torch.zeros(x.shape[0], x.shape[1], 17, 2)
        pred[..., 0] = x[:, :, 0, 0, 0].unsqueeze(-1)
        return pred


def test_evaluate_one_epoch_averages_overlapping_predictions_before_metrics():
    model = InputEncodedPoseModel()
    loader = DataLoader(OverlappingPredictionDataset(), batch_size=2)

    metrics = evaluate_one_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        amp_enabled=False,
        pck_thresholds=(0.50,),
    )

    assert torch.isclose(torch.tensor(metrics["mpjpe_norm"]), torch.tensor(1.0 / 3.0))
    assert torch.isclose(torch.tensor(metrics["pck_0.50"]), torch.tensor(2.0 / 3.0))
