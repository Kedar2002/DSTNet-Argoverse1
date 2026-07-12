"""
models.decoder.decoder

DSTNet trajectory decoder.

Paper
-----
Section III-E

The decoder predicts

    • K coarse trajectories
    • K confidence scores

from the encoded motion features.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.mlp import MLP
from models.model_types import Prediction


class Decoder(nn.Module):
    """
    DSTNet Decoder.

    Input
    -----
        Encoded Features

            (B,N,K,C)

    Output
    ------
        Prediction
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        prediction_steps: int = 30,
        num_modes: int = 6,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps
        self.num_modes = num_modes

        #######################################################################
        # Shared Decoder
        #######################################################################

        self.shared = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Trajectory Regression
        #######################################################################

        self.trajectory_head = nn.Linear(
            hidden_dim,
            prediction_steps * 2,
        )

        #######################################################################
        # Confidence Prediction
        #######################################################################

        self.score_head = nn.Linear(
            hidden_dim,
            1,
        )

    ###########################################################################

    def forward(
        self,
        features: torch.Tensor,
    ) -> Prediction:
        """
        Parameters
        ----------
        features

            (B,N,K,C)

        Returns
        -------
        Prediction
        """

        B, N, K, _ = features.shape

        x = self.shared(
            features,
        )

        ###############################################################
        # Trajectories
        ###############################################################

        trajectories = self.trajectory_head(
            x,
        )

        trajectories = trajectories.view(
            B,
            N,
            K,
            self.prediction_steps,
            2,
        )

        ###############################################################
        # Scores
        ###############################################################

        scores = self.score_head(
            x,
        ).squeeze(-1)

        return Prediction(
            trajectories=trajectories,
            scores=scores,
        )

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Decoder("
            f"prediction_steps={self.prediction_steps}, "
            f"num_modes={self.num_modes})"
        )
