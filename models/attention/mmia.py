"""
models.attention.mmia

Multi-Modal Interaction Attention (MMIA)

Fuses the outputs of the spatial attention branch
and the cross-attention branch.

Pipeline
--------
Spatial Features
        │
Cross Features
        │
Concatenate
        │
Linear Projection
        │
LayerNorm
        │
Feed Forward
        │
LayerNorm
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm


class MMIA(nn.Module):
    """
    Multi-Modal Interaction Attention.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        ###################################################################
        # Feature Fusion
        ###################################################################

        self.fusion = nn.Linear(
            hidden_dim * 2,
            hidden_dim,
        )

        ###################################################################
        # Feed Forward
        ###################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=4,
            dropout=dropout,
        )

        ###################################################################
        # Normalization
        ###################################################################

        self.norm1 = LayerNorm(
            hidden_dim,
        )

        self.norm2 = LayerNorm(
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        spatial_features: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_features
            (B,N,C)

        cross_features
            (B,N,C)

        Returns
        -------
        Tensor
            (B,N,C)
        """

        ###################################################################
        # Feature Fusion
        ###################################################################

        fused = torch.cat(
            (
                spatial_features,
                cross_features,
            ),
            dim=-1,
        )

        fused = self.fusion(
            fused,
        )

        ###################################################################
        # Residual
        ###################################################################

        features = self.norm1(
            spatial_features +
            self.dropout(fused)
        )

        ###################################################################
        # Feed Forward
        ###################################################################

        ff = self.feed_forward(
            features,
        )

        ###################################################################
        # Residual
        ###################################################################

        output = self.norm2(
            features +
            self.dropout(ff)
        )

        return output

    ###########################################################################

    def __repr__(self) -> str:

        return (
            "MMIA("
            f"hidden_dim={self.hidden_dim})"
        ) 
