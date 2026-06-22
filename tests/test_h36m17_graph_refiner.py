from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_h36m17_normalized_adjacency_matches_skeleton_edges() -> None:
    from dataset.h36m17 import H36M17_EDGES
    from models.h36m17_graph_refiner import build_h36m17_normalized_adjacency

    adjacency = build_h36m17_normalized_adjacency()

    assert adjacency.shape == (17, 17)
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(adjacency.diagonal() > 0)
    for left, right in H36M17_EDGES:
        assert adjacency[left, right] > 0
        assert adjacency[right, left] > 0
    assert adjacency[3, 10] == 0


def test_h36m17_graph_refiner_preserves_joint_feature_shape_and_gradients() -> None:
    from models.h36m17_graph_refiner import H36M17GraphRefiner

    refiner = H36M17GraphRefiner(d_model=256, num_layers=2)
    joint_features = torch.randn(2, 17, 256, requires_grad=True)

    refined = refiner(joint_features)
    refined.sum().backward()

    assert refined.shape == (2, 17, 256)
    assert torch.isfinite(refined).all()
    assert joint_features.grad is not None


def test_h36m17_graph_refiner_rejects_wrong_joint_count() -> None:
    from models.h36m17_graph_refiner import H36M17GraphRefiner

    refiner = H36M17GraphRefiner(d_model=256)
    bad_features = torch.randn(2, 18, 256)

    try:
        refiner(bad_features)
    except ValueError as error:
        assert "expected joint_features [batch,17,channels]" in str(error)
    else:
        raise AssertionError("expected wrong joint count to raise ValueError")
