from __future__ import annotations

import torch
import torch.nn as nn


class PoseAwareTokenProjection(nn.Module):
    """Frame-level token projection via residual attention pooling.

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

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C, T, S = z.shape
        z = z.reshape(B * L, C, T, S)
        h = z.mean(dim=(2, 3))  # placeholder: global avg pool
        h = h.reshape(B, L, -1)
        if return_attention:
            return h, {}
        return h
