from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pywt
import torch
from torch import nn


class TemporalSWTFeatureBank(nn.Module):
    def __init__(self, *, wavelet: str = "db2", levels: int = 3) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.levels = int(levels)
        if self.levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}")

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        if x.ndim != 4:
            raise ValueError(f"expected input [batch,antenna,subcarrier,time], got {tuple(x.shape)}")
        if x.shape[-1] % (2**self.levels) != 0:
            raise ValueError(f"time length {x.shape[-1]} must be divisible by {2**self.levels} for SWT")

        source = x.detach().cpu().numpy()
        coeffs = pywt.swt(source, self.wavelet, level=self.levels, axis=-1, trim_approx=False)
        approx = coeffs[-1][0]
        details = {f"D{level}": coeffs[level - 1][1] for level in range(1, self.levels + 1)}
        tensors = OrderedDict(
            [
                ("raw", x),
                (f"A{self.levels}", self._to_tensor(approx, x)),
            ]
        )
        for level in range(self.levels, 0, -1):
            tensors[f"D{level}"] = self._to_tensor(details[f"D{level}"], x)
        return tensors

    @staticmethod
    def _to_tensor(values: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
