from __future__ import annotations

import torch
from torch import nn


class JointQueryDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_joints: int = 17,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.joint_queries = nn.Parameter(torch.randn(num_joints, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_JointCrossAttentionBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected tokens [batch,tokens,channels], got {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        queries = self.joint_queries.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            queries = layer(queries, tokens)
        return queries


class _JointCrossAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(queries, tokens, tokens, need_weights=False)
        queries = self.cross_norm(queries + attended)
        return self.ffn_norm(queries + self.ffn(queries))
