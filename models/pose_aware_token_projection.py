from __future__ import annotations

import torch
import torch.nn as nn


class PoseAwareTokenProjection(nn.Module):
    """Frame-level token projection via residual attention pooling.

    Converts the encoder's per-frame CSI time-frequency feature map into a
    compact pose-aware frame token.  Uses global average pooling as a stable
    background baseline and residual attention pooling to extract foreground
    pose-perturbation patterns.

    Input:  [B, L, 128, 10, 29]
    Output: [B, L, 128]
    """

    def __init__(
        self,
        in_channels: int = 128,
        token_dim: int = 128,
        num_attention_maps: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_attention_maps = num_attention_maps

        # ── Stage 1: Channel Refinement ──────────────────────────────
        # 1×1 Conv recombines the encoder's 128 high-level CSI channels
        # without altering the 10×29 time-frequency layout.  This learns
        # which combinations of encoder channels are most useful for
        # pose-aware attention, rather than using the raw encoder output.
        # GroupNorm (not BatchNorm): avoids memorising source-domain
        # batch statistics — important for cross-domain few-shot fine-tuning.
        # GELU: smoother than ReLU for continuous CSI amplitude features.
        # ------------------------------------------------------------------
        self.channel_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.GELU(),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C, T, S = z.shape
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"
        N = B * L
        z_flat = z.reshape(N, C, T, S)

        # Stage 1: Channel Refinement  [N, C, T, S] -> [N, C, T, S]
        z_ref = self.channel_refine(z_flat)

        # Placeholder fusion: global average pool (replaced in Task 3-4)
        h = z_ref.mean(dim=(2, 3))
        h = h.reshape(B, L, self.token_dim)

        if return_attention:
            return h, {}
        return h
