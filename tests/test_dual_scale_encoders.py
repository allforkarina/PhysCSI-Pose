from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_dual_scale_encoder_outputs_documented_shapes() -> None:
    from models.scale_encoders import DualScaleEncoder

    encoder = DualScaleEncoder(wavelet="haar", d_model=256)
    x = torch.randn(1, 3, 114, 64)

    outputs = encoder(x)

    assert outputs["coarse_features"].shape == (1, 128, 29, 16)
    assert outputs["fine_features"].shape == (1, 128, 29, 32)
    assert outputs["coarse_tokens"].shape == (1, 464, 256)
    assert outputs["fine_tokens"].shape == (1, 928, 256)
    assert outputs["coarse_fusion_weights"].shape == (1, 3)
    assert outputs["fine_fusion_weights"].shape == (1, 3)
    assert torch.allclose(outputs["coarse_fusion_weights"].sum(dim=1), torch.ones(1))
    assert torch.allclose(outputs["fine_fusion_weights"].sum(dim=1), torch.ones(1))


def test_shared_scale_feature_mapper_preserves_scale_identity() -> None:
    from models.scale_fusion import SharedScaleFeatureMapper

    mapper = SharedScaleFeatureMapper(scale_names=("raw", "A3", "D3"))
    features = {
        "raw": torch.ones(1, 3, 4, 8),
        "A3": torch.ones(1, 3, 4, 8),
        "D3": torch.ones(1, 3, 4, 8),
    }

    mapped = mapper(features)

    assert tuple(mapped) == ("raw", "A3", "D3")
    assert all(value.shape == (1, 32, 4, 8) for value in mapped.values())
    assert not torch.equal(mapped["raw"], mapped["A3"])
