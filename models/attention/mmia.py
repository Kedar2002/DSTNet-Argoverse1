"""
models.attention.mmia

Multi-Modal Interaction Attention (MMIA)

DSTNet
Section III-D

Pipeline

Agent Features
        │
Lane Features
        │
        ▼
Agent → Lane Attention
        │
        ▼
Lane → Agent Attention
        │
        ▼
Learnable Gated Fusion
        │
        ▼
Feed Forward
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm


###############################################################################
# MMIA
###############################################################################


class MMIA(nn.Module):
    """
    Multi-Modal Interaction Attention.

    Performs bidirectional interaction between

        • agent features

        • lane features

    using symmetric cross-attention followed by gated fusion.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        #######################################################################
        # Agent -> Lane
        #######################################################################

        self.agent_to_lane = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        #######################################################################
        # Lane -> Agent
        #######################################################################

        self.lane_to_agent = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        #######################################################################
        # Learnable Gates
        #######################################################################

        self.agent_gate = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.Sigmoid(),
        )

        self.lane_gate = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.Sigmoid(),
        )

        #######################################################################
        # Feed Forward
        #######################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # LayerNorm
        #######################################################################

        self.agent_norm = LayerNorm(
            hidden_dim,
        )

        self.lane_norm = LayerNorm(
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    ###########################################################################
    # Agent → Lane Attention
    ###########################################################################

    def _agent_to_lane_attention(
        self,
        agent_features: Tensor,
        lane_features: Tensor,
        lane_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Update agent features using lane context.

        Query : Agent features

        Key/Value : Lane features
        """

        residual = agent_features

        x = self.agent_norm(
            agent_features,
        )

        context = self.agent_to_lane(
            query=x,
            key=lane_features,
            value=lane_features,
            mask=lane_mask,
        )

        context = self.dropout(
            context,
        )

        return residual + context

    ###########################################################################
    # Lane → Agent Attention
    ###########################################################################

    def _lane_to_agent_attention(
        self,
        lane_features: Tensor,
        agent_features: Tensor,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Update lane features using agent context.

        Query : Lane features

        Key/Value : Agent features
        """

        residual = lane_features

        x = self.lane_norm(
            lane_features,
        )

        context = self.lane_to_agent(
            query=x,
            key=agent_features,
            value=agent_features,
            mask=agent_mask,
        )

        context = self.dropout(
            context,
        )

        return residual + context

    ###########################################################################
    # Learnable Gated Fusion
    ###########################################################################

    def _gated_fusion(
        self,
        original: Tensor,
        updated: Tensor,
        gate: nn.Module,
    ) -> Tensor:
        """
        Learnable gated residual fusion.

        gate = σ(W[x; y])

        output = gate ⊙ y + (1-gate) ⊙ x
        """

        fusion = torch.cat(
            (
                original,
                updated,
            ),
            dim=-1,
        )

        alpha = gate(
            fusion,
        )

        return (
            alpha * updated
            + (1.0 - alpha) * original
        )

