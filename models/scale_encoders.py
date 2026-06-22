from __future__ import annotations

import torch
from torch import nn

from models.axial_attention import AxialAttentionEncoder
from models.baseline_csi_pose import ResidualBlock
from models.scale_fusion import SampleConditionedScaleFusion, SharedScaleFeatureMapper
from models.wavelet_feature_bank import TemporalSWTFeatureBank


class DualScaleEncoder(nn.Module):
    def __init__(self, *, wavelet: str = "db2", d_model: int = 256) -> None:
        super().__init__()
        self.feature_bank = TemporalSWTFeatureBank(wavelet=wavelet, levels=3)
        self.feature_mapper = SharedScaleFeatureMapper(scale_names=("raw", "A3", "D3", "D2", "D1"))
        self.coarse_fusion = SampleConditionedScaleFusion(scale_names=("raw", "A3", "D3"))
        self.fine_fusion = SampleConditionedScaleFusion(scale_names=("raw", "D2", "D1"))
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (3, 114, 64):
            raise ValueError(f"expected input [batch,3,114,64], got {tuple(x.shape)}")
        wavelet_features = self.feature_bank(x)
        mapped_features = self.feature_mapper(wavelet_features)
        coarse_input, coarse_weights = self.coarse_fusion(mapped_features)
        fine_input, fine_weights = self.fine_fusion(mapped_features)

        coarse_features = self.coarse_spatial_encoder(coarse_input)
        fine_features = self.fine_spatial_encoder(fine_input)
        coarse_encoded = self.coarse_axial_encoder(coarse_features)
        fine_encoded = self.fine_axial_encoder(fine_features)
        return {
            "coarse_features": coarse_features,
            "fine_features": fine_features,
            "coarse_encoded": coarse_encoded,
            "fine_encoded": fine_encoded,
            "coarse_tokens": _tokenize(coarse_encoded),
            "fine_tokens": _tokenize(fine_encoded),
            "coarse_fusion_weights": coarse_weights,
            "fine_fusion_weights": fine_weights,
        }


def _tokenize(x: torch.Tensor) -> torch.Tensor:
    batch, channels, freq, time = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch, freq * time, channels)
