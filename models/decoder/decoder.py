"""
models.decoder.decoder

DSTNet trajectory decoder.

Paper
-----
Section III-E

The decoder predicts

    • K coarse trajectories
    • K confidence scores

from the mode-aware encoder features.

The paper specifies a two-layer MLP decoder
(Eq. 25) followed by confidence prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.mlp import MLP
from models.model_types import ModeFeatures
from models.model_types import Prediction


class Decoder(nn.Module):
    """
    DSTNet Decoder.

    Input
    -----
        ModeFeatures

            features
                (B,N,K,C)

    Output
    ------
        Prediction
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        prediction_steps: int = 30,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps

        #######################################################################
        # Shared Decoder
        #######################################################################

        self.decoder = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Trajectory Head
        #######################################################################

        self.trajectory_head = nn.Linear(
            hidden_dim,
            prediction_steps * 2,
        )

        #######################################################################
        # Score Head
        #######################################################################

        self.score_head = nn.Linear(
            hidden_dim,
            1,
        )

    ###########################################################################

    def forward(
        self,
        mode_features: ModeFeatures,
    ) -> Prediction:
        """
        Parameters
        ----------
        mode_features

            Mode-aware encoder features.

        Returns
        -------
        Prediction
        """

        x = mode_features.features

        if x.ndim != 4:
            raise ValueError(
                "Expected shape (B,N,K,C)."
            )

        B, N, K, _ = x.shape

        ###################################################################
        # Shared decoder
        ###################################################################

        x = self.decoder(
            x,
        )

        ###################################################################
        # Coarse trajectories
        ###################################################################

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

        ###################################################################
        # Confidence
        ###################################################################

        scores = self.score_head(
            x,
        )

        scores = scores.squeeze(
            -1,
        )

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
            f"prediction_steps={self.prediction_steps})"
        )
