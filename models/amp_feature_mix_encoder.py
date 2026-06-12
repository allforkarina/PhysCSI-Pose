from __future__ import annotations

import torch
import torch.nn as nn


class AmpFeatureMixEncoder(nn.Module):
    """CSI feature encoder for PhysCSI-Pose v1.

    Consumes offline-built amplitude-only physical features and produces an
    intermediate time-frequency feature map.  A downstream Pose Regression Head
    consumes this output to predict 17 2D keypoints.

    Input:  [B, 12, 10, 114]  — 4 feature groups × 3 Rx antennas, 10 packets, 114 subcarriers
    Output: [B, 128, 10, 29]  — time-frequency feature map (no global pooling)
    """

    # ------------------------------------------------------------------
    # Input channel layout (immutable — defined by the data build pipeline)
    # ------------------------------------------------------------------
    # Channel  0– 2 : l_norm    rx0, rx1, rx2   — median/MAD-normalised amplitude residual
    # Channel  3– 5 : d_center  rx0, rx1, rx2   — short-time centred dynamics (~100 ms)
    # Channel  6– 8 : f_sub     rx0, rx1, rx2   — subcarrier local contrast
    # Channel  9–11 : c_ant     rx0, rx1, rx2   — inter-antenna relative amplitude

    def __init__(self) -> None:
        super().__init__()

        # ── Stage 0: Channel Projection ─────────────────────────────
        # 1×1 Conv only in the channel dimension — does NOT mix time or subcarrier axes.
        # Acts as a learnable "physical-feature fuser":
        #   • learns weighted combinations of l_norm, d_center, f_sub, c_ant
        #   • learns how to combine the three Rx antennas
        #   • preserves the raw time (10) and subcarrier (114) structure
        # Using a 1×1 Conv avoids prematurely destroying the input's physical
        # channel semantics with a large spatial kernel.
        # ------------------------------------------------------------------
        self.stage0 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
        )

        # ── Stage 1: Frequency Block ─────────────────────────────────
        # Convolves ONLY along the subcarrier axis (kernel=(1,7)), NOT along time.
        # Extracts local frequency-domain perturbation patterns:
        #   • human reflection / occlusion / multipath changes manifest as local
        #     peaks and valleys across adjacent subcarriers
        #   • kernel=7 covers a modest local frequency neighbourhood without
        #     relearning the environment's global frequency-envelope shape
        # Keeping the time-axis kernel=1 defers temporal modelling to Stage 2,
        # so this stage focuses purely on spectral structure.
        # Depthwise Separable Conv: fewer parameters, explicit control over
        # which axis is modelled, lower overfitting risk.
        # Residual with 1×1 projection (32 → 48 channels mismatch).
        # ------------------------------------------------------------------
        self.stage1_dw = nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3), groups=32)
        self.stage1_pw = nn.Conv2d(32, 48, kernel_size=1)
        self.stage1_norm = nn.GroupNorm(num_groups=8, num_channels=48)
        self.stage1_act = nn.GELU()
        self.stage1_skip = nn.Conv2d(32, 48, kernel_size=1)  # channel projection

        # ── Stage 2: Temporal Block ──────────────────────────────────
        # Convolves ONLY along the packet (time) axis (kernel=(3,1)), NOT along
        # subcarriers.
        # Captures short-time (~100 ms) dynamics within one pose frame:
        #   • kernel=3 sees adjacent packets to detect limb-motion-induced
        #     amplitude changes
        #   • leverages the d_center features that already encode packet-level
        #     deviations
        # Time dimension kept at 10 — no early averaging/pooling that would
        # discard packet-level motion cues.
        # Residual with 1×1 projection (48 → 64 channels mismatch).
        # ------------------------------------------------------------------
        self.stage2_dw = nn.Conv2d(48, 48, kernel_size=(3, 1), padding=(1, 0), groups=48)
        self.stage2_pw = nn.Conv2d(48, 64, kernel_size=1)
        self.stage2_norm = nn.GroupNorm(num_groups=8, num_channels=64)
        self.stage2_act = nn.GELU()
        self.stage2_skip = nn.Conv2d(48, 64, kernel_size=1)

        # ── Stage 3: Joint Time-Frequency Block 1 + Subcarrier ↓2 ────
        # First stage that models time and subcarrier JOINTLY (kernel=(3,5)).
        # By now the frequency structure (Stage 1) and temporal dynamics (Stage 2)
        # have been extracted separately; this stage learns their coupling:
        #   • how a short-time motion pattern relates to a local spectral change
        #   • e.g. a limb movement causing a specific subcarrier-group perturbation
        # Subcarrier downsampling 114→57 via learnable strided Conv2d:
        #   • subcarrier axis has higher redundancy than time (114 vs 10)
        #   • stride only on frequency dimension — time stays at 10
        #   • learnable downsampling is more flexible than fixed average pooling
        #     for CSI perturbation patterns
        # Residual with 1×1 projection (64 → 96 channels mismatch).
        # ------------------------------------------------------------------
        self.stage3_dw = nn.Conv2d(64, 64, kernel_size=(3, 5), padding=(1, 2), groups=64)
        self.stage3_pw = nn.Conv2d(64, 96, kernel_size=1)
        self.stage3_norm = nn.GroupNorm(num_groups=8, num_channels=96)
        self.stage3_act = nn.GELU()
        self.stage3_skip = nn.Conv2d(64, 96, kernel_size=1)
        self.stage3_down = nn.Conv2d(96, 96, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # ── Stage 4: Joint Time-Frequency Block 2 + Subcarrier ↓2 ────
        # Further abstracts the joint time-frequency patterns into higher-level
        # CSI pose features.
        # Subcarrier downsampling 57→29: outputs a medium-scale 10×29 time-
        # frequency grid — enough resolution for a downstream head to pool or
        # attend over, without carrying the full 114-subcarrier redundancy.
        # After two 2× downsamples the subcarrier axis is ~¼ of the original,
        # while the time axis remains intact at 10 packets.
        # Residual with 1×1 projection (96 → 128 channels mismatch).
        # ------------------------------------------------------------------
        self.stage4_dw = nn.Conv2d(96, 96, kernel_size=(3, 5), padding=(1, 2), groups=96)
        self.stage4_pw = nn.Conv2d(96, 128, kernel_size=1)
        self.stage4_norm = nn.GroupNorm(num_groups=8, num_channels=128)
        self.stage4_act = nn.GELU()
        self.stage4_skip = nn.Conv2d(96, 128, kernel_size=1)
        self.stage4_down = nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input guard — catch shape mismatches early
        assert x.ndim == 4, f"expected 4D input [B,12,10,114], got ndim={x.ndim}"
        assert x.shape[1] == 12, f"expected 12 input channels, got {x.shape[1]}"
        assert x.shape[2] == 10, f"expected 10 time steps, got {x.shape[2]}"
        assert x.shape[3] == 114, f"expected 114 subcarriers, got {x.shape[3]}"

        # Stage 0: Channel Projection  [B,12,10,114] → [B,32,10,114]
        x = self.stage0(x)

        # Stage 1: Frequency Block  [B,32,10,114] → [B,48,10,114]
        identity = self.stage1_skip(x)
        x = self.stage1_dw(x)
        x = self.stage1_pw(x)
        x = self.stage1_norm(x)
        x = self.stage1_act(x)
        x = x + identity

        # Stage 2: Temporal Block  [B,48,10,114] → [B,64,10,114]
        identity = self.stage2_skip(x)
        x = self.stage2_dw(x)
        x = self.stage2_pw(x)
        x = self.stage2_norm(x)
        x = self.stage2_act(x)
        x = x + identity

        # Stage 3: Joint TF Block 1 + Downsample  [B,64,10,114] → [B,96,10,57]
        identity = self.stage3_skip(x)
        x = self.stage3_dw(x)
        x = self.stage3_pw(x)
        x = self.stage3_norm(x)
        x = self.stage3_act(x)
        x = x + identity
        x = self.stage3_down(x)

        # Stage 4: Joint TF Block 2 + Downsample  [B,96,10,57] → [B,128,10,29]
        identity = self.stage4_skip(x)
        x = self.stage4_dw(x)
        x = self.stage4_pw(x)
        x = self.stage4_norm(x)
        x = self.stage4_act(x)
        x = x + identity
        x = self.stage4_down(x)

        return x
