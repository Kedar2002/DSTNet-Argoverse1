"""
losses.proposal_loss

Loss for supervising coarse trajectory proposals.

This loss is applied before the Adaptive Anchor-based
Refinement (AAR) module.

Each predicted mode is compared against the ground-truth
trajectory, and only the best-matching proposal contributes
to the regression loss.

This follows the common best-of-K supervision strategy used
by multimodal trajectory prediction methods.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import Prediction

class ProposalLoss(nn.Module):
    """
    Proposal regression loss.

    Inputs
    ------

    prediction

        trajectories
            (B,N,K,T,2)

    ground_truth

        (B,N,T,2)

    Returns
    -------

    scalar loss
    """

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:

        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        prediction: Prediction,
        ground_truth: torch.Tensor,
        *,
        return_best_mode: bool = False,
    ):

        trajectories = prediction.trajectories

        if trajectories.ndim != 5:

            raise ValueError(
                "Prediction must have shape (B,N,K,T,2)."
            )

        if ground_truth.ndim != 4:

            raise ValueError(
                "Ground truth must have shape (B,N,T,2)."
            )

        ###############################################################
        # Expand GT over modes
        ###############################################################

        gt = ground_truth.unsqueeze(2)

        gt = gt.expand_as(
            trajectories,
        )

        ###############################################################
        # L2 error for every mode
        ###############################################################

        displacement = torch.norm(
            trajectories - gt,
            dim=-1,
        )

        trajectory_error = displacement.mean(
            dim=-1,
        )

        ###############################################################
        # Best proposal
        ###############################################################

        best_error, best_mode = trajectory_error.min(
            dim=-1,
        )

        if self.reduction == "mean":
            loss = best_error.mean()

        elif self.reduction == "sum":
            loss = best_error.sum()

        else:
            loss = best_error

        if return_best_mode:
            return loss, best_mode

        return loss

    def __repr__(
        self,
    ) -> str:

        return (
            "ProposalLoss("
            f"reduction={self.reduction})"
        )

