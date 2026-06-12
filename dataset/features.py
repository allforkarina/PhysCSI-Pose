from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


FEATURE_CHANNELS: dict[str, tuple[int, int, int]] = {
    "l_norm": (0, 1, 2),
    "d_center": (3, 4, 5),
    "f_sub": (6, 7, 8),
    "c_ant": (9, 10, 11),
}
DEFAULT_FEATURES: tuple[str, ...] = ("l_norm", "d_center", "f_sub", "c_ant")


@dataclass(frozen=True)
class FeatureComponents:
    x: torch.Tensor
    l_norm: torch.Tensor
    d_center: torch.Tensor
    f_sub: torch.Tensor
    c_ant: torch.Tensor


def selected_feature_channels(features: list[str] | tuple[str, ...] | None = None) -> list[int]:
    names = DEFAULT_FEATURES if features is None else tuple(features)
    if not names:
        raise ValueError("features must contain at least one feature name")

    channels: list[int] = []
    for name in names:
        if name not in FEATURE_CHANNELS:
            raise ValueError(f"unknown feature {name!r}; expected one of {list(FEATURE_CHANNELS)}")
        channels.extend(FEATURE_CHANNELS[name])
    return channels


def _validate_csi_sequence(csiamp: torch.Tensor) -> None:
    if tuple(csiamp.shape) != (297, 10, 3, 114):
        raise ValueError(f"CSIamp sequence must have shape [297, 10, 3, 114], got {tuple(csiamp.shape)}")
    if not torch.isfinite(csiamp).all():
        raise ValueError("CSIamp sequence must contain only finite values")
    if torch.any(csiamp < 0):
        raise ValueError("CSIamp sequence must be non-negative")


def _subcarrier_smooth(l_norm: torch.Tensor, kernel_size: int, padding_mode: str) -> torch.Tensor:
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if padding_mode != "reflect":
        raise ValueError(f"only reflect padding is supported, got {padding_mode!r}")

    f_count, t_count, rx_count, sub_count = l_norm.shape
    flat = l_norm.reshape(f_count * t_count * rx_count, 1, sub_count)
    padded = F.pad(flat, (kernel_size // 2, kernel_size // 2), mode=padding_mode)
    smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return smoothed.reshape(f_count, t_count, rx_count, sub_count)


def build_amplitude_features(
    csiamp: torch.Tensor,
    *,
    eps_log: float = 1.0e-6,
    eps_mad: float = 1.0e-6,
    subcarrier_smooth_kernel: int = 15,
    subcarrier_padding_mode: str = "reflect",
    return_components: bool = False,
) -> torch.Tensor | FeatureComponents:
    csiamp = csiamp.to(dtype=torch.float32)
    _validate_csi_sequence(csiamp)

    l_value = torch.log(csiamp + eps_log)
    bg = torch.median(l_value.reshape(-1, 3, 114), dim=0).values
    centered = l_value - bg.view(1, 1, 3, 114)
    mad = torch.median(torch.abs(centered).reshape(-1, 3, 114), dim=0).values
    l_norm = centered / (mad.view(1, 1, 3, 114) + eps_mad)

    d_center = l_norm - l_norm.mean(dim=1, keepdim=True)
    smooth_sub = _subcarrier_smooth(
        l_norm,
        kernel_size=subcarrier_smooth_kernel,
        padding_mode=subcarrier_padding_mode,
    )
    f_sub = l_norm - smooth_sub
    c_ant = l_norm - l_norm.mean(dim=2, keepdim=True)

    blocks = [l_norm, d_center, f_sub, c_ant]
    x = torch.cat([block.permute(0, 2, 1, 3) for block in blocks], dim=1).contiguous()

    if return_components:
        return FeatureComponents(x=x, l_norm=l_norm, d_center=d_center, f_sub=f_sub, c_ant=c_ant)
    return x
