from __future__ import annotations

import torch
import torch.nn as nn


class PoseAwareTokenProjection(nn.Module):
    """Frame-level token projection via residual attention pooling.

    Converts the encoder's per-frame CSI time-frequency feature map into a
    compact pose-aware frame token.  Uses global average pooling as a stable
    background baseline and residual attention pooling to extract foreground
    pose-perturbation patterns.

    Input:  [B, L, 128, 10, 29]
    Output: [B, L, 128]
    """

    def __init__(
        self,
        in_channels: int = 128,
        token_dim: int = 128,
        num_attention_maps: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.num_attention_maps = num_attention_maps

        # ── Stage 1: Channel Refinement ──────────────────────────────
        # 1×1 Conv recombines the encoder's 128 high-level CSI channels
        # without altering the 10×29 time-frequency layout.  This learns
        # which combinations of encoder channels are most useful for
        # pose-aware attention, rather than using the raw encoder output.
        # GroupNorm (not BatchNorm): avoids memorising source-domain
        # batch statistics — important for cross-domain few-shot fine-tuning.
        # GELU: smoother than ReLU for continuous CSI amplitude features.
        # ------------------------------------------------------------------
        self.channel_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.GELU(),
        )

        # ── Stage 4: Multi-map Residual Attention ─────────────────────
        # A 1×1 Conv on the residual map R produces K unnormalised score
        # maps.  Softmax is applied over the 290 time-frequency positions
        # (T×S), NOT over channels — the goal is to select WHERE pose-
        # relevant perturbations occur in the time-frequency grid, not
        # WHICH feature channels matter.
        # K=4 allows the model to learn complementary attention patterns
        # (e.g. upper-body vs. lower-body perturbations) without explicit
        # body-part supervision.  Larger K increases parameters and
        # overfitting risk for few-shot fine-tuning.
        # ------------------------------------------------------------------
        self.attention_score = nn.Conv2d(in_channels, num_attention_maps, kernel_size=1)

        # ── Stage 5: Token Fusion ────────────────────────────────────
        # Concatenate the global background token with the K residual
        # attention tokens, then fuse via a learned linear projection.
        #   • h_avg provides the stable frame-level baseline (coarse pose,
        #     environment residual, energy distribution)
        #   • h_res_multi provides K pose-perturbation foreground summaries
        #     (local deviations from the frame background)
        # The fusion layer learns to balance them adaptively — relying on
        # the background when attention is uncertain, and on residual
        # tokens when fine-grained pose cues are available.
        # LayerNorm (not BatchNorm): token-scale stability for downstream
        # Temporal Relation Module, independent of batch size.
        # Dropout: reduces source-domain overfitting.
        # ------------------------------------------------------------------
        fusion_in = in_channels * (1 + num_attention_maps)  # 128 * (1 + 4) = 640
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C, T, S = z.shape
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"
        N = B * L
        z_flat = z.reshape(N, C, T, S)

        # Stage 1: Channel Refinement  [N, C, T, S] -> [N, C, T, S]
        z_ref = self.channel_refine(z_flat)

        # ── Stage 2: Global Background Token ──────────────────────────
        # Mean over time (10 packets) and subcarrier (29 positions) gives
        # the frame's overall CSI time-frequency response.  This serves as
        # a stable baseline representing:
        #   • residual environment and link-gain patterns
        #   • coarse body occlusion / reflection state
        #   • overall frame energy distribution
        # It acts as a fallback when residual attention is uncertain and
        # helps stabilise few-shot fine-tuning.
        # ------------------------------------------------------------------
        h_avg = z_ref.mean(dim=(2, 3))  # [N, C]

        # ── Stage 3: Residual Feature Map ─────────────────────────────
        # Subtract the frame-level per-channel mean from every time-frequency
        # position.  Positions that are close to the frame background cancel
        # to near-zero; positions with local pose-related perturbations are
        # emphasised.  This continues the de-environmenting philosophy from
        # CSI preprocessing:
        #   l_norm:  current amplitude - sequence-level background
        #   f_sub:   current subcarrier - local smooth background
        #   c_ant:   current antenna - antenna mean background
        #   R here:  current TF position - frame-level TF background
        # Using R (not Z_ref) for attention reduces the risk of the model
        # latching onto strong-but-not-pose-related environment responses.
        # ------------------------------------------------------------------
        z_bg = z_ref.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        r = z_ref - z_bg  # [N, C, T, S]

        # ── Stage 4: Attention scores on residual map ────────────────
        # Generate K score maps from the residual R (NOT from Z_ref).
        # Using R means attention focuses on "what's unusual relative to
        # this frame's background" rather than "what's strong in absolute
        # terms" — this suppresses environment-biased attention.
        # ------------------------------------------------------------------
        score = self.attention_score(r)  # [N, K, T, S]

        # Softmax over time-frequency positions (T*S = 290), not over channels.
        # This forces each attention map to distribute weight across the
        # time-frequency grid, selecting WHERE pose perturbations occur.
        score_flat = score.flatten(2)  # [N, K, T*S]
        alpha_flat = torch.softmax(score_flat, dim=-1)  # [N, K, T*S]
        alpha = alpha_flat.view_as(score)  # [N, K, T, S]

        # ── Stage 5: Residual attention tokens ────────────────────────
        # Weighted sum of the residual map R by each attention map alpha.
        # Because aggregation is over R (not Z_ref), the tokens capture
        # perturbation patterns relative to the frame background — the
        # core "residual attention" mechanism that suppresses environmental
        # bias while preserving pose-relevant local deviations.
        # ------------------------------------------------------------------
        r_flat = r.flatten(2)  # [N, C, T*S]
        h_res = torch.einsum("nkp,ncp->nkc", alpha_flat, r_flat)  # [N, K, C]

        # ── Stage 6: Token Fusion ────────────────────────────────────
        # Concat global background + K residual tokens, then fuse.
        # h_avg:  stable whole-frame baseline
        # h_res:  K foreground perturbation summaries
        # The Linear layer learns when to trust each source.
        # ------------------------------------------------------------------
        h_res_flat = h_res.flatten(1)  # [N, K*C]
        h_cat = torch.cat([h_avg, h_res_flat], dim=-1)  # [N, C*(1+K)]
        h = self.fusion(h_cat)  # [N, D]

        # ── Stage 7: Reshape to window structure ─────────────────────
        # Restore the [B, L, D] layout for the downstream Temporal
        # Relation Module which models inter-frame motion across the
        # L-frame window.
        # ------------------------------------------------------------------
        h = h.reshape(B, L, self.token_dim)

        if return_attention:
            aux = {
                "attention_maps": alpha.reshape(B, L, self.num_attention_maps, T, S),
                "h_avg": h_avg.reshape(B, L, self.in_channels),
                "h_res_multi": h_res.reshape(B, L, self.num_attention_maps, self.in_channels),
            }
            return h, aux
        return h
