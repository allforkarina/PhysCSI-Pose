from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_scale_aware_decoder_outputs_base_residual_and_gate() -> None:
    from models.scale_aware_decoder import ScaleAwareJointDecoder

    decoder = ScaleAwareJointDecoder(num_joints=17, d_model=256)
    coarse_tokens = torch.randn(2, 464, 256)
    fine_tokens = torch.randn(2, 928, 256)

    outputs = decoder(coarse_tokens=coarse_tokens, fine_tokens=fine_tokens)

    assert outputs["Z_coarse"].shape == (2, 17, 256)
    assert outputs["Z_fine"].shape == (2, 17, 256)
    assert outputs["P_base"].shape == (2, 17, 2)
    assert outputs["Delta_P"].shape == (2, 17, 2)
    assert outputs["alpha"].shape == (2, 17, 1)
    assert outputs["P_final"].shape == (2, 17, 2)
    assert torch.allclose(outputs["P_final"], outputs["P_base"] + outputs["alpha"] * outputs["Delta_P"])


def test_scale_gate_initial_alpha_is_near_sigmoid_minus_two() -> None:
    from models.scale_aware_decoder import ScaleAwareJointDecoder

    decoder = ScaleAwareJointDecoder(num_joints=17, d_model=256, gate_initial_bias=-2.0)
    coarse_tokens = torch.randn(1, 464, 256)
    fine_tokens = torch.randn(1, 928, 256)

    alpha = decoder(coarse_tokens=coarse_tokens, fine_tokens=fine_tokens)["alpha"]

    expected = torch.full_like(alpha, 1.0 / (1.0 + math.exp(2.0)))
    assert torch.allclose(alpha, expected, atol=1.0e-6)


def test_scale_aware_decoder_can_disable_fine_branch() -> None:
    from models.scale_aware_decoder import ScaleAwareJointDecoder

    decoder = ScaleAwareJointDecoder(num_joints=17, d_model=256)
    coarse_tokens = torch.randn(1, 464, 256)

    outputs = decoder(coarse_tokens=coarse_tokens, fine_tokens=None)

    assert outputs["P_final"].shape == (1, 17, 2)
    assert torch.equal(outputs["P_final"], outputs["P_base"])
    assert torch.equal(outputs["alpha"], torch.zeros(1, 17, 1))
    assert torch.equal(outputs["Delta_P"], torch.zeros(1, 17, 2))
