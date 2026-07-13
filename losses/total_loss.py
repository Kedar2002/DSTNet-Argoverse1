"""
losses.total_loss

Complete training objective for DSTNet.

Combines

    • Proposal Loss
    • Classification Loss
    • Score Loss
    • Refinement Loss

into a single optimization objective.
"""

from __future__ import annotations

import torch
from torch import nn

from models.model_types import (
    Prediction,
    RefinedPrediction,
)

from losses.proposal_loss import ProposalLoss
from losses.classification_loss import ClassificationLoss
from losses.score_loss import ScoreLoss
from losses.refinement_loss import RefinementLoss

class TotalLoss(nn.Module):
    """
    Overall DSTNet loss.
    """

    def __init__(
        self,
        proposal_weight: float = 1.0,
        classification_weight: float = 1.0,
        score_weight: float = 0.5,
        refinement_weight: float = 1.0,
    ):

        super().__init__()

        self.proposal_weight = proposal_weight

        self.classification_weight = classification_weight

        self.score_weight = score_weight

        self.refinement_weight = refinement_weight

        ###############################################################

        self.proposal_loss = ProposalLoss()

        self.classification_loss = ClassificationLoss()

        self.score_loss = ScoreLoss()

        self.refinement_loss = RefinementLoss()

    def forward(
        self,
        prediction: Prediction,
        refined_prediction: RefinedPrediction,
        ground_truth: torch.Tensor,
    ):
        ###############################################################
        # Proposal
        ###############################################################

        proposal_loss, best_mode = self.proposal_loss(
            prediction,
            ground_truth,
            return_best_mode=True,
        )

        ###############################################################
        # Classification
        ###############################################################

        classification_loss = self.classification_loss(
            prediction,
            best_mode,
        )

        ###############################################################
        # Score
        ###############################################################

        score_loss = self.score_loss(
            prediction,
            ground_truth,
        )

        ###############################################################
        # Refinement
        ###############################################################

        refinement_output = self.refinement_loss(
            refined_prediction,
            ground_truth,
        )

        if isinstance(
            refinement_output,
            dict,
        ):

            refinement_loss = refinement_output["loss"]

            refinement_metrics = refinement_output

        else:

            refinement_loss = refinement_output

            refinement_metrics = {}

        total_loss = (

            self.proposal_weight
            * proposal_loss

            +

            self.classification_weight
            * classification_loss

            +

            self.score_weight
            * score_loss

            +

            self.refinement_weight
            * refinement_loss
        )

        return {

            "loss": total_loss,

            "proposal_loss": proposal_loss,

            "classification_loss": classification_loss,

            "score_loss": score_loss,

            "refinement_loss": refinement_loss,

            **refinement_metrics,
        }

    def __repr__(
        self,
    ) -> str:

        return (
            "TotalLoss("
            f"proposal={self.proposal_weight}, "
            f"classification={self.classification_weight}, "
            f"score={self.score_weight}, "
            f"refinement={self.refinement_weight})"
        )


