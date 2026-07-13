"""
losses.refinement_loss

Loss for Adaptive Anchor-based Refinement (AAR).

This loss supervises the refined trajectories produced by the
Bezier refinement module.

The objective consists of

    • Trajectory regression
    • Endpoint accuracy
    • Bezier smoothness regularization
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import RefinedPrediction

from models.refinement.bezier import (
    bezier_regularization,
)

class RefinementLoss(nn.Module):
    """
    Loss for refined trajectories.
    """

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        endpoint_weight: float = 0.5,
        smoothness_weight: float = 0.05,
    ) -> None:

        super().__init__()

        self.trajectory_weight = trajectory_weight

        self.endpoint_weight = endpoint_weight

        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        prediction: RefinedPrediction,
        ground_truth: torch.Tensor,
    ):

        refined = prediction.trajectories

        if refined.ndim != 5:
            raise ValueError(
                "Expected (B,N,K,T,2)."
            )

        if ground_truth.ndim != 4:
            raise ValueError(
                "Expected GT (B,N,T,2)."
            )

        gt = ground_truth.unsqueeze(2)

        gt = gt.expand_as(
            refined,
        )

        ###############################################################
        # Best refined trajectory
        ###############################################################

        displacement = torch.norm(
            refined - gt,
            dim=-1,
        )

        ade = displacement.mean(
            dim=-1,
        )

        best_mode = ade.argmin(
            dim=-1,
        )

        B, N, K, T, _ = refined.shape

        gather_index = (
            best_mode
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B, N, 1, T, 2)
        )

        best_prediction = torch.gather(
            refined,
            dim=2,
            index=gather_index,
        ).squeeze(2)

        ###############################################################
        # Regression
        ###############################################################

        regression_loss = F.smooth_l1_loss(
            best_prediction,
            ground_truth,
        )

        ###############################################################
        # Final point accuracy
        ###############################################################

        endpoint_loss = F.smooth_l1_loss(
            best_prediction[..., -1, :],
            ground_truth[..., -1, :],
        )

        ###############################################################
        # Bezier regularization
        ###############################################################

        smoothness_loss = bezier_regularization(
            ground_truth,
            best_prediction,
        )

        total = (
            self.trajectory_weight
            * regression_loss
            +
            self.endpoint_weight
            * endpoint_loss
            +
            self.smoothness_weight
            * smoothness_loss
        )

        return {
            "loss": total,
            "regression": regression_loss,
            "endpoint": endpoint_loss,
            "smoothness": smoothness_loss,
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "RefinementLoss("
            f"trajectory={self.trajectory_weight}, "
            f"endpoint={self.endpoint_weight}, "
            f"smoothness={self.smoothness_weight})"
        )


