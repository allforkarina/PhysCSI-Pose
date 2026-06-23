from __future__ import annotations

import torch
from torch import nn


class ScaleAwareJointDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_joints: int = 17,
        d_model: int = 256,
        num_heads: int = 8,
        coarse_layers: int = 2,
        fine_layers: int = 2,
        use_gate: bool = True,
        gate_initial_bias: float = -2.0,
        joint_refiner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.joint_refiner = joint_refiner
        self.joint_queries = nn.Parameter(torch.randn(num_joints, d_model) * 0.02)
        self.coarse_layers = nn.ModuleList(
            [_CrossAttentionBlock(d_model=d_model, num_heads=num_heads) for _ in range(coarse_layers)]
        )
        self.fine_layers = nn.ModuleList(
            [_CrossAttentionBlock(d_model=d_model, num_heads=num_heads) for _ in range(fine_layers)]
        )
        self.fine_query_norm = nn.LayerNorm(d_model)
        self.base_head = nn.Linear(d_model, 2)
        self.residual_head = nn.Linear(d_model, 2)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        gate_logits = self.gate[-2]
        if not isinstance(gate_logits, nn.Linear):
            raise TypeError("gate logits layer must be linear")
        nn.init.zeros_(gate_logits.weight)
        nn.init.constant_(gate_logits.bias, gate_initial_bias)

    def forward(
        self,
        *,
        coarse_tokens: torch.Tensor,
        fine_tokens: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if coarse_tokens.ndim != 3:
            raise ValueError(f"expected coarse_tokens [batch,tokens,channels], got {tuple(coarse_tokens.shape)}")
        batch = coarse_tokens.shape[0]
        queries = self.joint_queries.unsqueeze(0).expand(batch, -1, -1)

        z_coarse = queries
        for layer in self.coarse_layers:
            z_coarse = layer(z_coarse, coarse_tokens)

        if fine_tokens is None:
            if self.joint_refiner is not None:
                z_coarse = self.joint_refiner(z_coarse)
            p_base = self.base_head(z_coarse)
            z_fine = torch.zeros_like(z_coarse)
            delta_p = torch.zeros_like(p_base)
            alpha = p_base.new_zeros(batch, self.joint_queries.shape[0], 1)
            return {
                "Z_coarse": z_coarse,
                "Z_fine": z_fine,
                "P_base": p_base,
                "Delta_P": delta_p,
                "alpha": alpha,
                "P_final": p_base,
            }

        if fine_tokens.ndim != 3:
            raise ValueError(f"expected fine_tokens [batch,tokens,channels], got {tuple(fine_tokens.shape)}")
        if fine_tokens.shape[0] != batch:
            raise ValueError("coarse_tokens and fine_tokens must have the same batch size")

        z_fine = self.fine_query_norm(queries + z_coarse)
        for layer in self.fine_layers:
            z_fine = layer(z_fine, fine_tokens)
        if self.joint_refiner is not None:
            z_coarse = self.joint_refiner(z_coarse)
            z_fine = self.joint_refiner(z_fine)
        p_base = self.base_head(z_coarse)
        delta_p = self.residual_head(z_fine)
        if self.use_gate:
            alpha = self.gate(torch.cat([z_coarse, z_fine, queries], dim=-1))
        else:
            alpha = p_base.new_ones(batch, self.joint_queries.shape[0], 1)
        return {
            "Z_coarse": z_coarse,
            "Z_fine": z_fine,
            "P_base": p_base,
            "Delta_P": delta_p,
            "alpha": alpha,
            "P_final": p_base + alpha * delta_p,
        }


class _CrossAttentionBlock(nn.Module):
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
