from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from models.h36m17_graph_refiner import H36M17GraphRefiner
from models.scale_aware_decoder import ScaleAwareJointDecoder
from models.scale_encoders import DualScaleEncoder


class WMWiFlowPoseModel(nn.Module):
    def __init__(
        self,
        *,
        num_joints: int = 17,
        d_model: int = 256,
        wavelet: str = "db2",
        wavelet_bands: Sequence[str] | None = None,
        use_fine_branch: bool = True,
        use_gate: bool = True,
        use_graph_refiner: bool = False,
    ) -> None:
        super().__init__()
        self.use_fine_branch = use_fine_branch
        graph_refiner = H36M17GraphRefiner(d_model=d_model) if use_graph_refiner else None
        self.encoder = DualScaleEncoder(
            wavelet=wavelet,
            d_model=d_model,
            wavelet_bands=wavelet_bands,
            use_fine_branch=use_fine_branch,
        )
        self.decoder = ScaleAwareJointDecoder(
            num_joints=num_joints,
            d_model=d_model,
            use_gate=use_gate,
            joint_refiner=graph_refiner,
        )

    def forward(
        self,
        x: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        encoded = self.encoder(x)
        decoded = self.decoder(
            coarse_tokens=encoded["coarse_tokens"],
            fine_tokens=encoded["fine_tokens"],
        )
        pose = decoded["P_final"]
        outputs = {**encoded, **decoded, "pose": pose}

        if not return_intermediates:
            return outputs["pose"]
        return outputs
