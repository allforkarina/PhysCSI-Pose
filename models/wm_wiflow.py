from __future__ import annotations

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
        use_fine_branch: bool = True,
        use_graph_refiner: bool = False,
    ) -> None:
        super().__init__()
        self.use_fine_branch = use_fine_branch
        self.encoder = DualScaleEncoder(wavelet=wavelet, d_model=d_model)
        self.decoder = ScaleAwareJointDecoder(num_joints=num_joints, d_model=d_model)
        self.graph_refiner = H36M17GraphRefiner(d_model=d_model) if use_graph_refiner else None
        self.graph_pose_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, *, return_intermediates: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(
            coarse_tokens=encoded["coarse_tokens"],
            fine_tokens=encoded["fine_tokens"] if self.use_fine_branch else None,
        )
        pose = decoded["P_final"]
        outputs = {**encoded, **decoded, "pose": pose}

        if self.graph_refiner is not None:
            joint_features = decoded["Z_coarse"] + decoded["alpha"] * decoded["Z_fine"]
            refined_features = self.graph_refiner(joint_features)
            outputs["pre_graph_pose"] = pose
            outputs["graph_joint_features"] = refined_features
            outputs["pose"] = self.graph_pose_head(refined_features)

        if not return_intermediates:
            return outputs["pose"]
        return outputs
