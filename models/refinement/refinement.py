"""
models.refinement.refinement

Adaptive Anchor-based Refinement (AAR)

This module performs iterative trajectory refinement using

    • Anchor Selection
    • Context Encoding
    • Bezier Refinement

The implementation reconstructs the AAR module described in
the DSTNet paper while remaining fully differentiable.
"""

from __future__ import annotations

import torch
from torch import nn

from models.model_types import Prediction
from models.model_types import RefinedPrediction

from models.refinement.anchor_selector import AnchorSelector
from models.refinement.context_encoder import ContextEncoder
from models.refinement.bezier import ResidualBezierRefinement

class Refinement(nn.Module):
    """
    Adaptive Anchor-based Refinement.

    Inputs
    ------

    encoder_features
        (B,N,C)

    prediction

        trajectories
            (B,N,K,T,2)

        scores
            (B,N,K)

    Output
    ------

    RefinedPrediction
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        prediction_steps: int = 30,
        refinement_iterations: int = 2,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        self.prediction_steps = prediction_steps

        self.refinement_iterations = refinement_iterations

        ###############################################################
        # Modules
        ###############################################################

        self.anchor_selector = AnchorSelector()

        self.context_encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            prediction_steps=prediction_steps,
        )

        self.refinement_layers = nn.ModuleList(
            [
                ResidualBezierRefinement(
                    hidden_dim=hidden_dim,
                    prediction_steps=prediction_steps,
                )
                for _ in range(refinement_iterations)
            ]
        )

    def forward(
        self,
        encoder_features: torch.Tensor,
        prediction: Prediction,
    ) -> RefinedPrediction:
        trajectories = prediction.trajectories

        scores = prediction.scores

        ###############################################################
        # Initial Anchor Selection
        ###############################################################

        anchors = self.anchor_selector(
            trajectories,
            scores,
        )

        refined = anchors

        ###############################################################
        # Iterative Refinement
        ###############################################################

        for refinement_layer in self.refinement_layers:

            context = self.context_encoder(
                scene_features=encoder_features,
                anchors=refined,
            )

            refined = refinement_layer(
                trajectory=refined,
                feature=context,
            )

        ###############################################################
        # Trajectory Offsets
        ###############################################################

        offsets = refined - trajectories

        ###############################################################
        # Return refined prediction
        ###############################################################

        return RefinedPrediction(
            trajectories=refined,
            scores=scores,
            offsets=offsets,
        )

    ###################################################################
    # Representation
    ###################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Refinement("
            f"hidden_dim={self.hidden_dim}, "
            f"prediction_steps={self.prediction_steps}, "
            f"iterations={self.refinement_iterations})"
        )


