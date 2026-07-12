"""
models.encoders.relative_embedding

Relative geometric embedding for DSTNet.

Computes pairwise geometric features between agents and projects
them into the latent feature space.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.mlp import MLP
from models.model_types import RelativeFeatures


class RelativeEmbedding(nn.Module):
    """
    Relative geometry embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = MLP(
            input_dim=4,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
    ) -> RelativeFeatures:
        """
        Parameters
        ----------
        positions
            Shape (B, N, 2)

        headings
            Shape (B, N)

        Returns
        -------
        RelativeFeatures
        """

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        if headings.ndim != 2:
            raise ValueError(
                "headings must have shape (B,N)."
            )

        ###############################################################
        # Pairwise position differences
        ###############################################################

        delta = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        dx = delta[..., 0]

        dy = delta[..., 1]

        ###############################################################
        # Distance
        ###############################################################

        distance = torch.linalg.norm(
            delta,
            dim=-1,
        )

        ###############################################################
        # Heading difference
        ###############################################################

        heading_delta = (
            headings[:, :, None]
            - headings[:, None, :]
        )

        ###############################################################
        # Relative vector
        ###############################################################

        features = torch.stack(
            (
                dx,
                dy,
                distance,
                heading_delta,
            ),
            dim=-1,
        )

        embedding = self.embedding(
            features
        )

        return RelativeFeatures(
            dx=dx,
            dy=dy,
            distance=distance,
            heading_delta=heading_delta,
            embedding=embedding,
        )

    def __repr__(self) -> str:

        return (
            "RelativeEmbedding("
            f"hidden_dim={self.hidden_dim})"
        )

