from __future__ import annotations

import sys
from pathlib import Path

import pywt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_temporal_swt_feature_bank_shapes_and_keys() -> None:
    from models.wavelet_feature_bank import TemporalSWTFeatureBank

    x = torch.arange(1 * 3 * 4 * 8, dtype=torch.float32).reshape(1, 3, 4, 8)
    bank = TemporalSWTFeatureBank(wavelet="db2", levels=3)

    features = bank(x)

    assert tuple(features) == ("raw", "A3", "D3", "D2", "D1")
    for value in features.values():
        assert value.shape == x.shape
        assert value.dtype == x.dtype
    assert torch.equal(features["raw"], x)


def test_temporal_swt_feature_bank_matches_pywavelets_level_order() -> None:
    from models.wavelet_feature_bank import TemporalSWTFeatureBank

    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    bank = TemporalSWTFeatureBank(wavelet="haar", levels=3)

    features = bank(x)
    coeffs = pywt.swt(x.numpy(), "haar", level=3, axis=-1, trim_approx=False)

    assert torch.allclose(features["A3"], torch.as_tensor(coeffs[0][0]))
    assert torch.allclose(features["D3"], torch.as_tensor(coeffs[0][1]))
    assert torch.allclose(features["D2"], torch.as_tensor(coeffs[1][1]))
    assert torch.allclose(features["D1"], torch.as_tensor(coeffs[2][1]))


def test_temporal_swt_feature_bank_can_select_bands_without_computing_unused_details() -> None:
    from models.wavelet_feature_bank import TemporalSWTFeatureBank

    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    bank = TemporalSWTFeatureBank(wavelet="haar", levels=3, bands=("raw", "A3", "D3"))

    features = bank(x)

    assert tuple(features) == ("raw", "A3", "D3")


def test_temporal_swt_feature_bank_is_deterministic() -> None:
    from models.wavelet_feature_bank import TemporalSWTFeatureBank

    x = torch.randn(2, 3, 4, 8)
    bank = TemporalSWTFeatureBank(wavelet="haar", levels=3)

    first = bank(x)
    second = bank(x)

    for key in first:
        assert torch.allclose(first[key], second[key])


def test_wavelet_concat_baseline_output_shape() -> None:
    from models.wavelet_concat_baseline import WaveletConcatBaseline

    model = WaveletConcatBaseline(num_joints=17, d_model=256, wavelet="haar")
    x = torch.randn(1, 3, 114, 64)

    pose = model(x)

    assert pose.shape == (1, 17, 2)
    assert torch.isfinite(pose).all()
