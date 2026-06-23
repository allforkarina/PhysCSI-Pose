from __future__ import annotations

import torch


NUM_H36M_KEYPOINTS = 17
H36M17_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)
H36M17_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)
NUM_POSE_KEYPOINTS = NUM_H36M_KEYPOINTS
POSE_BONE_EDGES = H36M17_BONE_EDGES


def build_normalized_adjacency(
    num_nodes: int = NUM_H36M_KEYPOINTS,
    edges: tuple[tuple[int, int], ...] = H36M17_BONE_EDGES,
) -> torch.Tensor:
    """Build symmetric normalized adjacency with self-loops for H36M-17 keypoints."""

    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    for start, end in edges:
        adjacency[start, end] = 1.0
        adjacency[end, start] = 1.0

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
