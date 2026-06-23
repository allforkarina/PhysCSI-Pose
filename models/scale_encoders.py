from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from models.axial_attention import AxialAttentionEncoder
from models.baseline_csi_pose import ResidualBlock
from models.scale_fusion import SampleConditionedScaleFusion, SharedScaleFeatureMapper
from models.wavelet_feature_bank import TemporalSWTFeatureBank


class DualScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        wavelet: str = "db2",
        d_model: int = 256,
        wavelet_bands: Sequence[str] | None = None,
        use_fine_branch: bool = True,
    ) -> None:
        super().__init__()
        selected_bands = tuple(wavelet_bands or ("raw", "A3", "D3", "D2", "D1"))
        self.coarse_scale_names = tuple(name for name in ("raw", "A3", "D3") if name in selected_bands)
        self.fine_scale_names = tuple(name for name in ("raw", "D2", "D1") if name in selected_bands and use_fine_branch)
        if not self.coarse_scale_names:
            raise ValueError("at least one coarse scale must be enabled")
        self.use_fine_branch = bool(use_fine_branch and self.fine_scale_names)
        required_bands = tuple(dict.fromkeys((*self.coarse_scale_names, *self.fine_scale_names)))
        self.feature_bank = TemporalSWTFeatureBank(wavelet=wavelet, levels=3, bands=required_bands)
        self.feature_mapper = SharedScaleFeatureMapper(scale_names=required_bands)
        self.coarse_fusion = SampleConditionedScaleFusion(scale_names=self.coarse_scale_names)
        self.fine_fusion = (
            SampleConditionedScaleFusion(scale_names=self.fine_scale_names) if self.use_fine_branch else None
        )
        self.coarse_spatial_encoder = nn.Sequential(
            ResidualBlock(32, 64, stride=(2, 2)),
            ResidualBlock(64, 128, stride=(2, 2)),
            ResidualBlock(128, 128, stride=(1, 1)),
        )
        self.fine_spatial_encoder = nn.Sequential(
            ResidualBlock(32, 64, stride=(2, 1)),
            ResidualBlock(64, 128, stride=(2, 2)),
            ResidualBlock(128, 128, stride=(1, 1)),
        )
        self.coarse_axial_encoder = AxialAttentionEncoder(
            in_channels=128,
            d_model=d_model,
            freq_tokens=29,
            time_tokens=16,
        )
        self.fine_axial_encoder = AxialAttentionEncoder(
            in_channels=128,
            d_model=d_model,
            freq_tokens=29,
            time_tokens=32,
        )

    def forward(self, x: torch.Tensor | Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        if isinstance(x, Mapping):
            wavelet_features = x
            raw = wavelet_features["raw"]
        else:
            raw = x
            wavelet_features = self.feature_bank(x)
        if raw.ndim != 4 or raw.shape[1:] != (3, 114, 64):
            raise ValueError(f"expected input [batch,3,114,64], got {tuple(raw.shape)}")
        mapped_features = self.feature_mapper(wavelet_features)
        coarse_input, coarse_weights = self.coarse_fusion(mapped_features)

        coarse_features = self.coarse_spatial_encoder(coarse_input)
        coarse_encoded = self.coarse_axial_encoder(coarse_features)
        outputs: dict[str, torch.Tensor | None] = {
            "coarse_features": coarse_features,
            "coarse_encoded": coarse_encoded,
            "coarse_tokens": _tokenize(coarse_encoded),
            "coarse_fusion_weights": coarse_weights,
            "fine_features": None,
            "fine_encoded": None,
            "fine_tokens": None,
            "fine_fusion_weights": None,
        }
        if self.use_fine_branch:
            if self.fine_fusion is None:
                raise RuntimeError("fine_fusion is not initialized")
            fine_input, fine_weights = self.fine_fusion(mapped_features)
            fine_features = self.fine_spatial_encoder(fine_input)
            fine_encoded = self.fine_axial_encoder(fine_features)
            outputs.update(
                {
                    "fine_features": fine_features,
                    "fine_encoded": fine_encoded,
                    "fine_tokens": _tokenize(fine_encoded),
                    "fine_fusion_weights": fine_weights,
                }
            )
        return outputs


def _tokenize(x: torch.Tensor) -> torch.Tensor:
    batch, channels, freq, time = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch, freq * time, channels)
