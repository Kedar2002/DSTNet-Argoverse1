"""
models.encoders.gsta

Global Spatio-Temporal Attention (GSTA)

One GSTA layer consists of

    MSPA
        ↓
    MHCA
        ↓
    MMIA
        ↓
    FeedForward
        ↓
    Residual + LayerNorm

This follows the DSTNet encoder design.
"""

from __future__ import annotations

import torch
from torch import nn

from models.attention.mspa import MSPA
from models.attention.mhca import MHCA
from models.attention.mmia import MMIA
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Attention layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        ###############################################################
        # Attention Branches
        ###############################################################

        self.spatial_attention = MSPA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.cross_attention = MHCA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.interaction_attention = MMIA(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        ###############################################################
        # Feed Forward
        ###############################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=4,
            dropout=dropout,
        )

        ###############################################################
        # Normalization
        ###############################################################

        self.norm = LayerNorm(
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    ###################################################################
    # Forward
    ###################################################################

    def forward(
        self,
        *,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        ###############################################################
        # Spatial Attention
        ###############################################################

        spatial = self.spatial_attention(
            agent_features,
            mask=agent_mask,
            attention_bias=attention_bias,
        )

        ###############################################################
        # Cross Attention
        ###############################################################

        cross = self.cross_attention(
            query_features=spatial,
            context_features=lane_features,
            context_mask=lane_mask,
        )

        ###############################################################
        # Interaction Fusion
        ###############################################################

        fused = self.interaction_attention(
            spatial_features=spatial,
            cross_features=cross,
        )

        ###############################################################
        # Feed Forward
        ###############################################################

        ff = self.feed_forward(
            fused,
        )

        ###############################################################
        # Final Residual
        ###############################################################

        agent_output = self.norm(
            fused +
            self.dropout(ff)
        )

        ###############################################################
        # Lane features remain unchanged in this layer
        ###############################################################

        return (
            agent_output,
            lane_features,
        )

    def __repr__(self) -> str:

        return (
            "GSTA("
            f"hidden_dim={self.hidden_dim})"
        )
