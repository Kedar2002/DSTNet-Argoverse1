"""
models.encoders.relative_embedding

Relative spatial-temporal embedding for DSTNet.

This module computes pairwise geometric embeddings that are
consumed by the graph-aware attention blocks.
"""

from __future__ import annotations

import torch
from torch import nn


class RelativeEmbedding(nn.Module):
    """
    Relative spatial embedding.

    Input
    -----
    positions : (B, N, 2)

    headings : (B, N)

    Output
    ------
    (B, N, N, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
    ) -> None:

        super().__init__()

        self._hidden_dim = hidden_dim

        #
        # Input features
        #
        # dx
        # dy
        # distance
        # sin(d_heading)
        # cos(d_heading)
        #

        self.embedding = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
    ) -> torch.Tensor:

        #
        # positions
        #
        # (B,N,2)
        #

        delta = (
            positions[:, :, None, :]
            -
            positions[:, None, :, :]
        )

        dx = delta[..., 0]

        dy = delta[..., 1]

        distance = torch.sqrt(
            dx * dx + dy * dy + 1e-6
        )

        heading_delta = (
            headings[:, :, None]
            -
            headings[:, None, :]
        )

        features = torch.stack(
            (
                dx,
                dy,
                distance,
                torch.sin(heading_delta),
                torch.cos(heading_delta),
            ),
            dim=-1,
        )

        return self.embedding(
            features,
        )

    def __repr__(
        self,
    ) -> str:

        return (
            "RelativeEmbedding("
            f"hidden_dim={self._hidden_dim})"
        )

    
