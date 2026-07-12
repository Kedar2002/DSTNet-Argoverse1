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

Implementation Notes
--------------------
This implementation follows the paper while adding
a residual connection and LayerNorm around the shared
MLP for improved optimization stability.
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
        # Decoder Normalization
        #######################################################################

        self.norm = nn.LayerNorm(
            hidden_dim,
        )

        #######################################################################
        # Trajectory Prediction
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

        #######################################################################
        # Initialization
        #######################################################################

        nn.init.zeros_(
            self.trajectory_head.bias,
        )

        nn.init.zeros_(
            self.score_head.bias,
        )

    ###########################################################################
    # Forward
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

        B, N, K, C = x.shape

        if C != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension "
                f"{self.hidden_dim}, got {C}."
            )

        #######################################################################
        # Shared Decoder
        #######################################################################

        residual = x

        x = self.decoder(
            x,
        )

        x = residual + x

        x = self.norm(
            x,
        )

        #######################################################################
        # Trajectory Prediction
        #######################################################################

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

        #######################################################################
        # Confidence Prediction
        #######################################################################

        scores = self.score_head(
            x,
        )

        scores = scores.squeeze(
            dim=-1,
        )

        #######################################################################
        # Output
        #######################################################################

        return Prediction(
            trajectories=trajectories,
            scores=scores,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Decoder("
            f"hidden_dim={self.hidden_dim}, "
            f"prediction_steps={self.prediction_steps})"
        )
