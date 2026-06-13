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

        # ── Stage 2: Global Background Token ──────────────────────────
        # Mean over time (10 packets) and subcarrier (29 positions) gives
        # the frame's overall CSI time-frequency response.  This serves as
        # a stable baseline representing:
        #   • residual environment and link-gain patterns
        #   • coarse body occlusion / reflection state
        #   • overall frame energy distribution
        # It acts as a fallback when residual attention is uncertain and
        # helps stabilise few-shot fine-tuning.
        # ------------------------------------------------------------------
        h_avg = z_ref.mean(dim=(2, 3))  # [N, C]

        # ── Stage 3: Residual Feature Map ─────────────────────────────
        # Subtract the frame-level per-channel mean from every time-frequency
        # position.  Positions that are close to the frame background cancel
        # to near-zero; positions with local pose-related perturbations are
        # emphasised.  This continues the de-environmenting philosophy from
        # CSI preprocessing:
        #   l_norm:  current amplitude - sequence-level background
        #   f_sub:   current subcarrier - local smooth background
        #   c_ant:   current antenna - antenna mean background
        #   R here:  current TF position - frame-level TF background
        # Using R (not Z_ref) for attention reduces the risk of the model
        # latching onto strong-but-not-pose-related environment responses.
        # ------------------------------------------------------------------
        z_bg = z_ref.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        r = z_ref - z_bg  # [N, C, T, S]

        # Placeholder fusion: just use h_avg (attention + fusion in Task 4)
        h = h_avg.reshape(B, L, self.token_dim)

        if return_attention:
            return h, {}
        return h
