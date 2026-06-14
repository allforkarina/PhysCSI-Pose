from __future__ import annotations

import torch
import torch.nn as nn

from models.amp_feature_mix_encoder import AmpFeatureMixEncoder
from models.pose_aware_token_projection import PoseAwareTokenProjection
from models.pose_heatmap_decoder import PoseHeatmapDecoder
from models.temporal_lite_transformer import TemporalLiteTransformer


class PhysCSIPoseNet(nn.Module):
    """PhysCSI-Pose model wrapper for fixed-length temporal windows.

    Input:  [B, L, C, 10, 114]
    Output: [B, L, 17, 2]
    """

    def __init__(
        self,
        input_channels: int = 12,
        token_dim: int = 128,
        num_joints: int = 17,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_max_window_length: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.token_dim = token_dim
        self.num_joints = num_joints

        self.encoder = AmpFeatureMixEncoder(input_channels=input_channels)
        self.token_projection = PoseAwareTokenProjection(
            in_channels=128,
            token_dim=token_dim,
            num_attention_maps=4,
            dropout=dropout,
        )
        self.temporal = TemporalLiteTransformer(
            input_dim=token_dim,
            min_window_length=4,
            max_window_length=temporal_max_window_length,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            ffn_expansion=2,
            attention_dropout=dropout,
            ffn_dropout=dropout,
            residual_dropout=dropout,
        )
        self.decoder = PoseHeatmapDecoder(
            input_dim=token_dim,
            num_joints=num_joints,
            heatmap_size=64,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.ndim == 5, f"expected 5D input [B,L,C,10,114], got ndim={x.ndim}"
        b, window_length, channels, time_steps, subcarriers = x.shape
        assert channels == self.input_channels, (
            f"expected {self.input_channels} channels, got {channels}"
        )
        assert time_steps == 10, f"expected 10 packets, got {time_steps}"
        assert subcarriers == 114, f"expected 114 subcarriers, got {subcarriers}"

        x_flat = x.reshape(b * window_length, channels, time_steps, subcarriers)
        encoder_maps = self.encoder(x_flat).reshape(b, window_length, 128, 10, 29)
        tokens = self.token_projection(encoder_maps)
        temporal_tokens = self.temporal(tokens)
        pred = self.decoder(temporal_tokens)

        if return_aux:
            return pred, {
                "encoder_maps": encoder_maps,
                "tokens": tokens,
                "temporal_tokens": temporal_tokens,
            }
        return pred
