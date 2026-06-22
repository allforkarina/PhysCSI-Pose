from __future__ import annotations

import torch
from torch import nn


class AxialAttentionEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        d_model: int = 256,
        freq_tokens: int = 29,
        time_tokens: int = 16,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.freq_pos = nn.Parameter(torch.zeros(1, in_channels, freq_tokens, 1))
        self.time_pos = nn.Parameter(torch.zeros(1, in_channels, 1, time_tokens))
        self.freq_attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.time_attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.freq_norm = nn.LayerNorm(in_channels)
        self.time_norm = nn.LayerNorm(in_channels)
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [batch,channels,freq,time], got {tuple(x.shape)}")
        x = x + self.freq_pos + self.time_pos
        x = self._attend_frequency(x)
        x = self._attend_time(x)
        return self.projection(x)

    def _attend_frequency(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, time = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch * time, freq, channels)
        attended, _ = self.freq_attention(seq, seq, seq, need_weights=False)
        seq = self.freq_norm(seq + attended)
        return seq.reshape(batch, time, freq, channels).permute(0, 3, 2, 1)

    def _attend_time(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, freq, time = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(batch * freq, time, channels)
        attended, _ = self.time_attention(seq, seq, seq, need_weights=False)
        seq = self.time_norm(seq + attended)
        return seq.reshape(batch, freq, time, channels).permute(0, 3, 1, 2)
