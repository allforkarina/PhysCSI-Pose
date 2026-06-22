from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
from torch import nn


class SharedScaleFeatureMapper(nn.Module):
    def __init__(self, *, scale_names: Sequence[str], out_channels: int = 32) -> None:
        super().__init__()
        self.scale_names = tuple(scale_names)
        self.mapper = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=(5, 3), padding=(2, 1)),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.scale_embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, out_channels, 1, 1) * 0.02) for name in self.scale_names}
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        mapped = OrderedDict()
        for name in self.scale_names:
            if name not in features:
                raise KeyError(f"missing scale feature: {name}")
            mapped[name] = self.mapper(features[name]) + self.scale_embeddings[name]
        return mapped


class SampleConditionedScaleFusion(nn.Module):
    def __init__(self, *, scale_names: Sequence[str], hidden_dim: int = 16) -> None:
        super().__init__()
        self.scale_names = tuple(scale_names)
        self.weight_mlp = nn.Sequential(
            nn.Linear(len(self.scale_names), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.scale_names)),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([features[name] for name in self.scale_names], dim=1)
        pooled = stacked.mean(dim=(2, 3, 4))
        weights = torch.softmax(self.weight_mlp(pooled), dim=1)
        fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
        return fused, weights
