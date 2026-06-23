from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import pywt
import torch
from torch import nn


class TemporalSWTFeatureBank(nn.Module):
    VALID_BANDS = ("raw", "A3", "D3", "D2", "D1")

    def __init__(self, *, wavelet: str = "db2", levels: int = 3, bands: Sequence[str] | None = None) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.levels = int(levels)
        if self.levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}")
        if self.levels != 3:
            raise ValueError("TemporalSWTFeatureBank currently names bands for levels=3")
        self.bands = tuple(bands or self.VALID_BANDS)
        unknown = set(self.bands) - set(self.VALID_BANDS)
        if unknown:
            raise ValueError(f"unknown wavelet bands: {sorted(unknown)}")

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        if x.ndim != 4:
            raise ValueError(f"expected input [batch,antenna,subcarrier,time], got {tuple(x.shape)}")
        if x.shape[-1] % (2**self.levels) != 0:
            raise ValueError(f"time length {x.shape[-1]} must be divisible by {2**self.levels} for SWT")

        source = x.detach().cpu().numpy()
        coeffs = pywt.swt(source, self.wavelet, level=self.levels, axis=-1, trim_approx=False)
        by_band = {
            "raw": x,
            "A3": self._to_tensor(coeffs[0][0], x),
            "D3": self._to_tensor(coeffs[0][1], x),
            "D2": self._to_tensor(coeffs[1][1], x),
            "D1": self._to_tensor(coeffs[2][1], x),
        }
        tensors = OrderedDict((band, by_band[band]) for band in self.bands)
        return tensors

    @staticmethod
    def _to_tensor(values: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
