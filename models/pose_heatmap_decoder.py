from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapSoftArgmax2D(nn.Module):
    """Differentiable coordinate readout from 2D joint heatmaps.

    A heatmap lets the decoder represent spatial uncertainty for fine joints.
    Soft-argmax keeps the readout differentiable, unlike hard argmax, so a
    coordinate loss can train the high-resolution heatmap generator end to end.
    """

    def __init__(
        self,
        heatmap_size: int = 64,
        coord_min: float = -0.8,
        coord_max: float = 0.8,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.coord_min = coord_min
        self.coord_max = coord_max

        coords = torch.linspace(coord_min, coord_max, steps=heatmap_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(1, 1, 1, heatmap_size, heatmap_size))
        self.register_buffer("grid_y", grid_y.reshape(1, 1, 1, heatmap_size, heatmap_size))

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.ndim == 5, f"expected 5D heatmap logits [B,L,J,H,W], got ndim={logits.ndim}"
        assert logits.shape[-2:] == (self.heatmap_size, self.heatmap_size), (
            f"expected heatmap size {self.heatmap_size}x{self.heatmap_size}, got {logits.shape[-2:]}"
        )

        b, l, j, h, w = logits.shape

        # Softmax over the spatial field turns raw logits into one probability
        # distribution per joint. This lets weak wrist/elbow evidence stay broad
        # during learning instead of being forced into an early hard location.
        heatmaps = torch.softmax(logits.reshape(b, l, j, h * w), dim=-1).reshape_as(logits)

        # Expected x/y over the fixed target-coordinate grid gives coordinates
        # directly in the same [-0.8, 0.8] space as the cleaned labels.
        x = (heatmaps * self.grid_x).sum(dim=(-2, -1))
        y = (heatmaps * self.grid_y).sum(dim=(-2, -1))
        coords = torch.stack([x, y], dim=-1)
        return coords, heatmaps


class JointQueryInjection(nn.Module):
    """Convert each frame token into joint-conditioned tokens.

    The temporal token is a whole-body CSI summary. Adding a learned embedding
    per joint asks 17 different localization questions from that same evidence,
    protecting small joints such as wrists from sharing one undifferentiated
    coordinate regressor with the torso.
    """

    def __init__(self, input_dim: int = 128, num_joints: int = 17) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.joint_embedding = nn.Parameter(torch.zeros(num_joints, input_dim))
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def forward(self, temporal_tokens: torch.Tensor) -> torch.Tensor:
        assert temporal_tokens.ndim == 3, (
            f"expected 3D temporal tokens [B,L,D], got ndim={temporal_tokens.ndim}"
        )
        assert temporal_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {temporal_tokens.shape[-1]}"
        )

        # Broadcast the same frame-level CSI pose evidence to every joint, then
        # add a learned query so each joint can decode a different body landmark.
        return temporal_tokens.unsqueeze(2) + self.joint_embedding.view(1, 1, self.num_joints, self.input_dim)


class JointFeatureRefinement(nn.Module):
    """Residual per-joint MLP refinement before spatial decoding.

    The temporal transformer has already handled frame-to-frame context. This
    block specializes each joint token for its own landmark, and the residual
    path keeps the original joint-conditioned evidence available for few-shot
    stability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # LayerNorm is batch-size independent, which matches the rest of the
        # model design and avoids source-domain batch statistics in fine-tuning.
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        assert joint_tokens.ndim == 4, (
            f"expected 4D joint tokens [B,L,J,D], got ndim={joint_tokens.ndim}"
        )
        assert joint_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {joint_tokens.shape[-1]}"
        )

        # Residual refinement sharpens joint-specific features without forcing
        # the decoder to discard the temporal token evidence it started from.
        return joint_tokens + self.mlp(self.norm(joint_tokens))


class HeatmapUpsampleBlock(nn.Module):
    """2x spatial upsampling block for heatmap refinement.

    Bilinear upsampling avoids checkerboard artifacts from transposed
    convolution. The following convolution learns how to sharpen the joint
    belief field at the higher resolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        assert out_channels % 8 == 0, f"expected out_channels divisible by 8, got {out_channels}"
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class JointHeatmapGenerator(nn.Module):
    """Decode joint tokens into high-resolution per-joint heatmap logits.

    Each joint token first becomes a coarse 8x8 spatial hypothesis. Progressive
    upsampling refines that hypothesis to 64x64, giving limb endpoints a fine
    spatial field instead of forcing direct coordinate regression.
    """

    def __init__(
        self,
        input_dim: int = 128,
        decoder_channels: int = 64,
        seed_size: int = 8,
        heatmap_size: int = 64,
    ) -> None:
        super().__init__()
        assert heatmap_size == seed_size * 8, (
            f"expected heatmap_size == seed_size * 8 for three upsample blocks, got {heatmap_size} and {seed_size}"
        )
        assert decoder_channels % 16 == 0, f"expected decoder_channels divisible by 16, got {decoder_channels}"
        self.input_dim = input_dim
        self.decoder_channels = decoder_channels
        self.seed_size = seed_size
        self.heatmap_size = heatmap_size

        # The seed projection creates a coarse per-joint spatial field. Starting
        # from 8x8 is much cheaper than directly projecting to 64x64.
        self.seed = nn.Linear(input_dim, decoder_channels * seed_size * seed_size)

        # Three refinement steps give a high-resolution probability field:
        # 8x8 captures coarse body-region evidence, then 16/32/64 refine it for
        # small structures such as wrists and elbows.
        self.decoder = nn.Sequential(
            HeatmapUpsampleBlock(decoder_channels, decoder_channels),
            HeatmapUpsampleBlock(decoder_channels, decoder_channels // 2),
            HeatmapUpsampleBlock(decoder_channels // 2, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        assert joint_tokens.ndim == 4, (
            f"expected 4D joint tokens [B,L,J,D], got ndim={joint_tokens.ndim}"
        )
        assert joint_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {joint_tokens.shape[-1]}"
        )

        b, l, j, _ = joint_tokens.shape
        x = self.seed(joint_tokens)
        x = x.reshape(b * l * j, self.decoder_channels, self.seed_size, self.seed_size)
        logits = self.decoder(x)
        return logits.reshape(b, l, j, self.heatmap_size, self.heatmap_size)


class PoseHeatmapDecoder(nn.Module):
    """High-resolution pose decoder for temporal CSI frame tokens.

    The decoder intentionally predicts heatmaps before coordinates. This gives
    fine joints such as wrists and elbows a spatial probability field, while the
    soft-argmax readout keeps training compatible with coordinate labels.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_joints: int = 17,
        heatmap_size: int = 64,
        coord_min: float = -0.8,
        coord_max: float = 0.8,
        joint_hidden_dim: int = 128,
        decoder_channels: int = 64,
        seed_size: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # Step 1: turn each whole-frame temporal token into 17 joint-specific
        # queries so each landmark gets its own decoding path.
        self.joint_query = JointQueryInjection(input_dim=input_dim, num_joints=num_joints)

        # Step 2: two small residual MLP blocks specialize each joint token
        # without adding another temporal model on top of TemporalLiteTransformer.
        self.refine1 = JointFeatureRefinement(input_dim=input_dim, hidden_dim=joint_hidden_dim, dropout=dropout)
        self.refine2 = JointFeatureRefinement(input_dim=input_dim, hidden_dim=joint_hidden_dim, dropout=dropout)

        # Step 3: decode every joint token into a high-resolution belief map.
        self.heatmap_generator = JointHeatmapGenerator(
            input_dim=input_dim,
            decoder_channels=decoder_channels,
            seed_size=seed_size,
            heatmap_size=heatmap_size,
        )

        # Step 4: read differentiable coordinates in the same normalized range
        # as the labels, while optionally keeping heatmaps for diagnostics.
        self.readout = HeatmapSoftArgmax2D(
            heatmap_size=heatmap_size,
            coord_min=coord_min,
            coord_max=coord_max,
        )

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        return_heatmaps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert temporal_tokens.ndim == 3, (
            f"expected 3D temporal tokens [B,L,D], got ndim={temporal_tokens.ndim}"
        )
        assert temporal_tokens.shape[-1] == self.input_dim, (
            f"expected token dim {self.input_dim}, got {temporal_tokens.shape[-1]}"
        )

        joint_tokens = self.joint_query(temporal_tokens)
        joint_tokens = self.refine1(joint_tokens)
        joint_tokens = self.refine2(joint_tokens)
        heatmap_logits = self.heatmap_generator(joint_tokens)
        coords, heatmaps = self.readout(heatmap_logits)

        if return_heatmaps:
            return coords, {"heatmap_logits": heatmap_logits, "heatmaps": heatmaps}
        return coords
