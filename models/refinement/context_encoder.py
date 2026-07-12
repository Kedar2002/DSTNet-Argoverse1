"""
models.refinement.context_encoder

Context encoder for Adaptive Anchor-based Refinement (AAR).

This module fuses

    • Scene context
    • Anchor trajectory geometry

to produce context-aware refinement features.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.mlp import MLP


class ContextEncoder(nn.Module):
    """
    Context Encoder.

    Inputs
    ------
    scene_features
        (B,N,C)

    anchors
        (B,N,K,T,2)

    Outputs
    -------
    context_features
        (B,N,K,C)
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
        # Anchor Geometry Encoder
        #######################################################################

        self.anchor_encoder = MLP(
            input_dim=prediction_steps * 2,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Score Encoder
        #######################################################################

        self.score_encoder = nn.Sequential(
            nn.Linear(
                1,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        #######################################################################
        # Scene + Anchor Fusion
        #######################################################################

        self.fusion = MLP(
            input_dim=hidden_dim * 3,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(
            hidden_dim,
        )

    ###########################################################################

    def forward(
        self,
        scene_features: torch.Tensor,
        anchors: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        scene_features

            (B,N,C)

        anchors

            (B,N,K,T,2)

        Returns
        -------
        (B,N,K,C)
        """

        if scene_features.ndim != 3:
            raise ValueError(
                "scene_features must have shape (B,N,C)."
            )

        if anchors.ndim != 5:
            raise ValueError(
                "anchors must have shape (B,N,K,T,2)."
            )

        B, N, K, T, D = anchors.shape

        if D != 2:
            raise ValueError(
                "Last anchor dimension must equal 2."
            )

        ###################################################################
        # Encode anchor geometry
        ###################################################################

        anchor_features = anchors.reshape(
            B,
            N,
            K,
            T * 2,
        )

        anchor_features = self.anchor_encoder(
            anchor_features,
        )

        ###################################################################
        # Encode confidence score
        ###################################################################

        score_features = self.score_encoder(
            scores.unsqueeze(-1),
        )

        ###################################################################
        # Expand scene features
        ###################################################################

        scene_features = scene_features.unsqueeze(
            2,
        )

        scene_features = scene_features.expand(
            B,
            N,
            K,
            self.hidden_dim,
        )

        ###################################################################
        # Fuse
        ###################################################################

        fused = torch.cat(
            (
                scene_features,
                anchor_features,
                score_features,
            ),
            dim=-1,
        )

        context = self.fusion(
            fused,
        )

        ###################################################################
        # Residual
        ###################################################################

        context = context + scene_features

        context = self.norm(
            context,
        )

        return context

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "ContextEncoder("
            f"hidden_dim={self.hidden_dim})"
        )
