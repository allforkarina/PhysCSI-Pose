from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from models.axial_attention import AxialAttentionEncoder
from models.baseline_csi_pose import ResidualBlock
from models.h36m17_graph_refiner import H36M17GraphRefiner
from models.joint_decoder import JointQueryDecoder
from models.wavelet_feature_bank import TemporalSWTFeatureBank


class WaveletConcatBaseline(nn.Module):
    def __init__(
        self,
        *,
        num_joints: int = 17,
        d_model: int = 256,
        wavelet: str = "db2",
        wavelet_bands: Sequence[str] | None = None,
        use_graph_refiner: bool = False,
    ) -> None:
        super().__init__()
        self.feature_bank = TemporalSWTFeatureBank(wavelet=wavelet, levels=3, bands=wavelet_bands)
        self.stem = nn.Sequential(
            nn.Conv2d(3 * len(self.feature_bank.bands), 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.spatial_encoder = nn.Sequential(
            ResidualBlock(32, 64, stride=(2, 2)),
            ResidualBlock(64, 128, stride=(2, 2)),
            ResidualBlock(128, 128, stride=(1, 1)),
        )
        self.axial_encoder = AxialAttentionEncoder(in_channels=128, d_model=d_model)
        self.joint_decoder = JointQueryDecoder(num_joints=num_joints, d_model=d_model)
        self.graph_refiner = H36M17GraphRefiner(d_model=d_model) if use_graph_refiner else None
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, Mapping):
            features = x
            raw = features["raw"]
        else:
            raw = x
            features = self.feature_bank(x)
        if raw.ndim != 4 or raw.shape[1:] != (3, 114, 64):
            raise ValueError(f"expected input [batch,3,114,64], got {tuple(raw.shape)}")
        concat = torch.cat([features[band] for band in self.feature_bank.bands], dim=1)
        stem_features = self.stem(concat)
        spatial_features = self.spatial_encoder(stem_features)
        encoded_features = self.axial_encoder(spatial_features)
        tokens = encoded_features.permute(0, 2, 3, 1).reshape(raw.shape[0], 29 * 16, -1)
        joint_features = self.joint_decoder(tokens)
        if self.graph_refiner is not None:
            joint_features = self.graph_refiner(joint_features)
        return self.pose_head(joint_features)
