"""
losses.score_loss

Confidence calibration loss.

Unlike ClassificationLoss, which predicts the best proposal,
ScoreLoss encourages trajectory confidence scores to correlate
with trajectory quality.

Better trajectories should receive higher confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import Prediction

class ScoreLoss(nn.Module):
    """
    Confidence calibration loss.

    Inputs
    ------

    prediction

    ground_truth

    Returns
    -------
    scalar loss
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):

        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        prediction: Prediction,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:

        trajectories = prediction.trajectories

        scores = prediction.scores

        if trajectories.ndim != 5:

            raise ValueError(
                "Expected trajectories "
                "(B,N,K,T,2)."
            )

        if ground_truth.ndim != 4:

            raise ValueError(
                "Expected GT "
                "(B,N,T,2)."
            )

        ###############################################################
        # Expand GT
        ###############################################################

        gt = ground_truth.unsqueeze(
            2,
        )

        gt = gt.expand_as(
            trajectories,
        )

        ###############################################################
        # ADE
        ###############################################################

        error = torch.norm(
            trajectories - gt,
            dim=-1,
        )

        error = error.mean(
            dim=-1,
        )

        ###############################################################
        # Convert error into confidence target
        #
        # Smaller error
        # ->
        # Higher confidence
        ###############################################################

        target = torch.softmax(
            -error,
            dim=-1,
        )

        prediction_prob = torch.softmax(
            scores,
            dim=-1,
        )

        loss = F.kl_div(
            prediction_prob.log(),
            target,
            reduction="none",
        )

        loss = loss.sum(
            dim=-1,
        )

        if self.reduction == "mean":

            return loss.mean()

        if self.reduction == "sum":

            return loss.sum()

        return loss

    def __repr__(
        self,
    ) -> str:

        return (
            "ScoreLoss("
            f"reduction={self.reduction})"
        )

