from __future__ import annotations

import torch
from torch import nn

from dataset.h36m17 import H36M17_EDGES


def build_h36m17_normalized_adjacency(
    *,
    num_joints: int = 17,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=dtype)
    for left, right in H36M17_EDGES:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


class H36M17GraphRefiner(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("adjacency", build_h36m17_normalized_adjacency())
        self.layers = nn.ModuleList([_GraphRefinementLayer(d_model=d_model) for _ in range(num_layers)])

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        if joint_features.ndim != 3 or joint_features.shape[1] != 17:
            raise ValueError(f"expected joint_features [batch,17,channels], got {tuple(joint_features.shape)}")
        if joint_features.shape[2] != self.d_model:
            raise ValueError(f"expected joint feature channels {self.d_model}, got {joint_features.shape[2]}")

        refined = joint_features
        adjacency = self.adjacency.to(device=joint_features.device, dtype=joint_features.dtype)
        for layer in self.layers:
            refined = layer(refined, adjacency)
        return refined


class _GraphRefinementLayer(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.message = nn.Linear(d_model, d_model)
        self.message_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, joint_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjd->bid", adjacency, joint_features)
        joint_features = self.message_norm(joint_features + self.message(aggregated))
        return self.ffn_norm(joint_features + self.ffn(joint_features))
