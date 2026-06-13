from __future__ import annotations

import pytest
import torch

from models.pose_heatmap_decoder import (
    HeatmapSoftArgmax2D,
    HeatmapUpsampleBlock,
    JointFeatureRefinement,
    JointHeatmapGenerator,
    JointQueryInjection,
    PoseHeatmapDecoder,
)


def test_soft_argmax_outputs_coords_and_heatmaps():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    coords, heatmaps = readout(logits)

    assert coords.shape == (2, 4, 17, 2)
    assert heatmaps.shape == (2, 4, 17, 64, 64)
    assert torch.isfinite(coords).all()
    assert torch.isfinite(heatmaps).all()


def test_soft_argmax_heatmaps_sum_to_one():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    _, heatmaps = readout(logits)

    probs_sum = heatmaps.flatten(-2).sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_soft_argmax_coordinates_stay_in_target_range():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.randn(2, 4, 17, 64, 64)

    coords, _ = readout(logits)

    assert coords.min() >= -0.80001
    assert coords.max() <= 0.80001


def test_soft_argmax_maps_dominant_pixel_to_expected_coordinate():
    readout = HeatmapSoftArgmax2D(heatmap_size=64, coord_min=-0.8, coord_max=0.8)
    logits = torch.full((1, 1, 1, 64, 64), -20.0)
    logits[..., 0, 0] = 20.0

    coords, _ = readout(logits)

    assert torch.allclose(coords[0, 0, 0], torch.tensor([-0.8, -0.8]), atol=1e-3)


def test_soft_argmax_rejects_wrong_shape():
    readout = HeatmapSoftArgmax2D()
    logits = torch.randn(2, 17, 64, 64)

    with pytest.raises(AssertionError, match="expected 5D heatmap logits"):
        readout(logits)


def test_joint_query_injection_output_shape():
    layer = JointQueryInjection(input_dim=128, num_joints=17)
    temporal_tokens = torch.randn(2, 4, 128)

    joint_tokens = layer(temporal_tokens)

    assert joint_tokens.shape == (2, 4, 17, 128)
    assert torch.isfinite(joint_tokens).all()


def test_joint_query_injection_has_one_embedding_per_joint():
    layer = JointQueryInjection(input_dim=128, num_joints=17)

    assert layer.joint_embedding.shape == (17, 128)


def test_joint_query_injection_rejects_wrong_token_dim():
    layer = JointQueryInjection(input_dim=128, num_joints=17)
    temporal_tokens = torch.randn(2, 4, 64)

    with pytest.raises(AssertionError, match="expected token dim"):
        layer(temporal_tokens)


def test_joint_feature_refinement_output_shape():
    refine = JointFeatureRefinement(input_dim=128, hidden_dim=128, dropout=0.1)
    joint_tokens = torch.randn(2, 4, 17, 128)

    out = refine(joint_tokens)

    assert out.shape == (2, 4, 17, 128)
    assert torch.isfinite(out).all()


def test_joint_feature_refinement_uses_layernorm_and_residual_mlp():
    refine = JointFeatureRefinement(input_dim=128, hidden_dim=128, dropout=0.1)

    assert isinstance(refine.norm, torch.nn.LayerNorm)
    assert isinstance(refine.mlp[0], torch.nn.Linear)
    assert isinstance(refine.mlp[1], torch.nn.GELU)


def test_joint_feature_refinement_rejects_wrong_shape():
    refine = JointFeatureRefinement(input_dim=128)
    joint_tokens = torch.randn(2, 4, 128)

    with pytest.raises(AssertionError, match="expected 4D joint tokens"):
        refine(joint_tokens)


def test_heatmap_upsample_block_doubles_spatial_resolution():
    block = HeatmapUpsampleBlock(in_channels=64, out_channels=64)
    x = torch.randn(8, 64, 8, 8)

    y = block(x)

    assert y.shape == (8, 64, 16, 16)
    assert torch.isfinite(y).all()


def test_joint_heatmap_generator_output_shape():
    generator = JointHeatmapGenerator(
        input_dim=128,
        decoder_channels=64,
        seed_size=8,
        heatmap_size=64,
    )
    joint_tokens = torch.randn(2, 4, 17, 128)

    logits = generator(joint_tokens)

    assert logits.shape == (2, 4, 17, 64, 64)
    assert torch.isfinite(logits).all()


def test_joint_heatmap_generator_rejects_wrong_shape():
    generator = JointHeatmapGenerator()
    joint_tokens = torch.randn(2, 4, 128)

    with pytest.raises(AssertionError, match="expected 4D joint tokens"):
        generator(joint_tokens)


def test_pose_heatmap_decoder_outputs_coordinates():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)

    coords = decoder(temporal_tokens)

    assert coords.shape == (2, 4, 17, 2)
    assert torch.isfinite(coords).all()
    assert coords.min() >= -0.80001
    assert coords.max() <= 0.80001


def test_pose_heatmap_decoder_can_return_heatmaps():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)

    coords, aux = decoder(temporal_tokens, return_heatmaps=True)

    assert coords.shape == (2, 4, 17, 2)
    assert aux["heatmap_logits"].shape == (2, 4, 17, 64, 64)
    assert aux["heatmaps"].shape == (2, 4, 17, 64, 64)
    probs_sum = aux["heatmaps"].flatten(-2).sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_pose_heatmap_decoder_rejects_wrong_input_shape():
    decoder = PoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 128)

    with pytest.raises(AssertionError, match="expected 3D temporal tokens"):
        decoder(temporal_tokens)


def test_pose_heatmap_decoder_parameter_count_is_reasonable():
    decoder = PoseHeatmapDecoder()
    n_params = sum(p.numel() for p in decoder.parameters())
    assert n_params < 750_000, f"expected decoder under 750k params, got {n_params:,}"


def test_pose_heatmap_decoder_is_exported_from_models_package():
    from models import PoseHeatmapDecoder as ExportedPoseHeatmapDecoder

    decoder = ExportedPoseHeatmapDecoder()
    temporal_tokens = torch.randn(2, 4, 128)
    coords = decoder(temporal_tokens)
    assert coords.shape == (2, 4, 17, 2)
