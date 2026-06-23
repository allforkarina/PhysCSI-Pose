from __future__ import annotations

import torch


NUM_OPENPOSE_KEYPOINTS = 18
OPENPOSE_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 8),
    (5, 11),
)


def build_normalized_adjacency(
    num_nodes: int = NUM_OPENPOSE_KEYPOINTS,
    edges: tuple[tuple[int, int], ...] = OPENPOSE_BONE_EDGES,
) -> torch.Tensor:
    """Build symmetric normalized adjacency with self-loops for OpenPose keypoints."""

    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for start, end in edges:
        adjacency[start, end] = 1.0
        adjacency[end, start] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]