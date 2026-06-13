from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvPosEncoding(nn.Module):
    """Local temporal convolutional positional encoding for short frame windows.

    Depthwise Conv1d gives each token channel a small learnable local-time cue
    without mixing semantic channels; this is enough for 4-8 frame windows and
    cheaper than absolute position tables tied to one fixed window length.
    """

    def __init__(self, input_dim: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"expected odd kernel_size, got {kernel_size}"
        padding = kernel_size // 2

        # One filter per channel preserves the pose-token feature basis while
        # injecting nearest-neighbor temporal order, which matters for motion.
        self.dwconv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=input_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)

        # Residual positional encoding keeps the original frame token intact so
        # later attention can use position without overwriting pose evidence.
        return residual + x


class RelativeTemporalSelfAttention(nn.Module):
    """Non-causal multi-head self-attention with learnable temporal offsets."""

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        max_window_length: int = 8,
        attention_dropout: float = 0.1,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.max_window_length = max_window_length
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Four 32-D heads are a small enough split for this data-limited phase
        # while still letting different heads prefer different temporal offsets.
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        # Multi-head context is concatenated back to D=128, then projected so
        # the attention sublayer can learn a channel regrouping across heads
        # instead of leaving them mechanically stitched together. The FFN also
        # performs block-level nonlinear refinement, but this output projection
        # is the attention sublayer's own head-fusion step.
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

        # Relative bias is used instead of absolute time IDs so the same module
        # can handle L=4..8 while learning whether nearby or farther frames help.
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_window_length - 1))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def relative_bias_for_length(self, window_length: int, device: torch.device) -> torch.Tensor:
        assert window_length <= self.max_window_length, (
            f"expected window_length <= {self.max_window_length}, got {window_length}"
        )
        positions = torch.arange(window_length, device=device)
        relative_distance = positions[None, :] - positions[:, None]
        bias_index = relative_distance + (self.max_window_length - 1)
        return self.relative_bias[:, bias_index]

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.ndim == 3, f"expected 3D input [B,L,D], got ndim={x.ndim}"
        batch_size, window_length, token_dim = x.shape
        assert token_dim == self.input_dim, f"expected token dim {self.input_dim}, got {token_dim}"

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, window_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        score = score + self.relative_bias_for_length(window_length, x.device).unsqueeze(0)

        # Non-causal attention is intentional for offline pose windows: each
        # frame token may use both earlier and later context in the same window.
        attention = torch.softmax(score, dim=-1)
        attention_for_context = self.attn_drop(attention)
        context = torch.matmul(attention_for_context, v)
        context = context.transpose(1, 2).reshape(batch_size, window_length, token_dim)
        out = self.proj(context)
        out = self.proj_drop(out)

        if return_attention:
            return out, {"attention": attention}
        return out


class LiteTransformerBlock(nn.Module):
    """Pre-Norm temporal Transformer block for short pose-aware token windows."""

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        max_window_length: int = 8,
        ffn_expansion: int = 2,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = input_dim * ffn_expansion

        # CPE before attention exposes local motion order before the global
        # window-level token mixing step decides which frames should interact.
        self.pos_encoding = TemporalConvPosEncoding(input_dim=input_dim)

        # PreNorm keeps the residual path close to identity, which is useful for
        # shallow stacks and avoids relying on downstream training code for warmup.
        self.norm_attn = nn.LayerNorm(input_dim)
        self.attn = RelativeTemporalSelfAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            max_window_length=max_window_length,
            attention_dropout=attention_dropout,
            projection_dropout=0.0,
        )
        self.attn_resid_drop = nn.Dropout(residual_dropout)

        # A 2x FFN adds per-frame nonlinear refinement without dominating the
        # parameter budget; dropout is included for later training or few-shot
        # fine-tuning to reduce overfitting risk. Current tests only verify
        # code correctness with synthetic tensors.
        self.norm_ffn = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(residual_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.pos_encoding(x)

        if return_attention:
            attn_out, aux = self.attn(self.norm_attn(x), return_attention=True)
        else:
            attn_out = self.attn(self.norm_attn(x), return_attention=False)
            aux = {}

        # Residual attention and FFN updates preserve one output token per input
        # frame while allowing each block to add only the temporal evidence it needs.
        x = x + self.attn_resid_drop(attn_out)
        x = x + self.ffn(self.norm_ffn(x))

        if return_attention:
            return x, aux
        return x


class TemporalLiteTransformer(nn.Module):
    """Short-window temporal relation module.

    Input:  [B, L, 128], where 4 <= L <= 8
    Output: [B, L, 128]
    """

    def __init__(
        self,
        input_dim: int = 128,
        min_window_length: int = 4,
        max_window_length: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion: int = 2,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert min_window_length >= 1, f"expected min_window_length >= 1, got {min_window_length}"
        assert max_window_length >= min_window_length, (
            f"expected max_window_length >= min_window_length, got {max_window_length} < {min_window_length}"
        )
        assert num_layers >= 1, f"expected num_layers >= 1, got {num_layers}"
        self.input_dim = input_dim
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Two blocks are the default because L is only 4-8: one block can align
        # local frame evidence, and a second can refine relations without making
        # the temporal module the dominant model component.
        self.layers = nn.ModuleList(
            [
                LiteTransformerBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    max_window_length=max_window_length,
                    ffn_expansion=ffn_expansion,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert x.ndim == 3, f"expected 3D input [B,L,D], got ndim={x.ndim}"
        _, window_length, token_dim = x.shape
        assert token_dim == self.input_dim, f"expected token dim {self.input_dim}, got {token_dim}"
        assert self.min_window_length <= window_length <= self.max_window_length, (
            f"expected window length in [{self.min_window_length}, {self.max_window_length}], "
            f"got {window_length}"
        )

        attention_maps = []
        for layer in self.layers:
            if return_attention:
                x, aux = layer(x, return_attention=True)
                attention_maps.append(aux["attention"])
            else:
                x = layer(x, return_attention=False)

        if return_attention:
            return x, {"attention_maps": torch.stack(attention_maps, dim=1)}
        return x
