"""
models.encoders.gsta

Global Spatio-Temporal Attention (GSTA)

GSTA performs an initial global aggregation of agent features
before the stacked TriATM refinement blocks.

Paper Note
----------
The DSTNet paper introduces GSTA as the first encoder stage,
but does not specify its exact internal implementation.

This implementation follows a standard Transformer encoder
block using global self-attention.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm
from models.model_types import GraphData
from models.model_types import RelativeFeatures


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Attention.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        #######################################################################
        # Global Self-Attention
        #######################################################################

        self.self_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        #######################################################################
        # Feed Forward
        #######################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=4,
            dropout=dropout,
        )

        #######################################################################
        # Output
        #######################################################################

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = LayerNorm(
            hidden_dim,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        relative: RelativeFeatures,
        graph: GraphData,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        agent_features
            (B,N,C)

        lane_features
            (B,L,C)

        Returns
        -------
        Tuple

            Updated agent features

            Lane features (unchanged)
        """

        #######################################################################
        # Currently unused
        #######################################################################

        _ = relative
        _ = graph
        _ = lane_mask

        #######################################################################
        # Global Self Attention
        #######################################################################

        residual = agent_features

        agent_features = self.self_attention(
            query=agent_features,
            key=agent_features,
            value=agent_features,
            mask=agent_mask,
        )

        agent_features = residual + self.dropout(
            agent_features,
        )

        agent_features = self.norm(
            agent_features,
        )

        #######################################################################
        # Feed Forward
        #######################################################################

        agent_features = self.feed_forward(
            agent_features,
        )

        #######################################################################
        # Return
        #######################################################################

        return (
            agent_features,
            lane_features,
        )

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "GSTA("
            f"hidden_dim={self.hidden_dim})"
        )
