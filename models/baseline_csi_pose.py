from __future__ import annotations

import torch
from torch import nn

from models.axial_attention import AxialAttentionEncoder
from models.joint_decoder import JointQueryDecoder


class BaselineCSIPoseModel(nn.Module):
    def __init__(self, *, num_joints: int = 17, d_model: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
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
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, *, return_intermediates: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (3, 114, 64):
            raise ValueError(f"expected input [batch,3,114,64], got {tuple(x.shape)}")
        stem_features = self.stem(x)
        spatial_features = self.spatial_encoder(stem_features)
        encoded_features = self.axial_encoder(spatial_features)
        tokens = encoded_features.permute(0, 2, 3, 1).reshape(x.shape[0], 29 * 16, -1)
        joint_features = self.joint_decoder(tokens)
        pose = self.pose_head(joint_features)
        if not return_intermediates:
            return pose
        return {
            "stem_features": stem_features,
            "spatial_features": spatial_features,
            "encoded_features": encoded_features,
            "tokens": tokens,
            "joint_features": joint_features,
            "pose": pose,
        }


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


def _group_count(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1
