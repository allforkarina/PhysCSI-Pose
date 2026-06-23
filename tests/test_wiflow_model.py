from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import WiFlowModel


def test_encode_features_returns_axial_feature_map() -> None:
    model = WiFlowModel()
    model.eval()
    x = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        features = model.encode_features(x)

    assert features.shape == (2, 256, 29, 16)


def test_forward_decodes_encoded_features() -> None:
    model = WiFlowModel()
    model.eval()
    x = torch.randn(2, 3, 114, 64)

    with torch.no_grad():
        prediction = model(x)
        decoded = model.decode_features(model.encode_features(x))

    assert prediction.shape == (2, 17, 2)
    assert torch.allclose(prediction, decoded)
