"""
models.encoders.gsta

Global Spatio-Temporal Attention (GSTA)

Paper:
DSTNet

This module constructs the global scene representation before
the Tri-Attention encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from models.encoders.relative_embedding import RelativeEmbedding
from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import build_normalization


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Attention.

    Inputs
    ------
    agent_features : (B, Na, C)

    lane_features : (B, Nl, C)

    positions : (B, Na, 2)

    headings : (B, Na)

    Output
    ------
    Scene representation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        normalization: str = "layernorm",
    ) -> None:

        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_heads = num_heads

        ###############################################################
        # Relative Embedding
        ###############################################################

        self.relative_embedding = RelativeEmbedding(
            hidden_dim,
        )

        ###############################################################
        # Agent Self Attention
        ###############################################################

        self.agent_attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout,
        )

        ###############################################################
        # Lane Self Attention
        ###############################################################

        self.lane_attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout,
        )

        ###############################################################
        # Cross Attention
        ###############################################################

        self.cross_attention = MultiHeadAttention(
            hidden_dim,
            num_heads,
            dropout,
        )

        self.norm = build_normalization(
            normalization,
            hidden_dim,
        )

        self.ffn = FeedForward(
            hidden_dim,
            dropout=dropout,
            normalization=normalization,
        )

        ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        positions: torch.Tensor,
        headings: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        #
        # Relative embedding
        #
        # NOTE:
        # The current implementation computes the embeddings so that
        # they are available for graph-aware attention. The paper does
        # not explicitly describe how they are injected into the
        # attention scores, so this integration will be completed once
        # the graph encoder is implemented.
        #

        _ = self.relative_embedding(
            positions,
            headings,
        )

        ###############################################################
        # Agent Self Attention
        ###############################################################

        agents = self.agent_attention(
            agent_features,
            agent_features,
            agent_features,
            agent_mask,
        )

        ###############################################################
        # Lane Self Attention
        ###############################################################

        lanes = self.lane_attention(
            lane_features,
            lane_features,
            lane_features,
            lane_mask,
        )

        ###############################################################
        # Agent ← Lane Cross Attention
        ###############################################################

        fused = self.cross_attention(
            agents,
            lanes,
            lanes,
            None,
        )

        fused = fused + agents

        fused = self.norm(
            fused,
        )

        fused = self.ffn(
            fused,
        )

        return fused

    def __repr__(
        self,
    ) -> str:

        return (
            "GSTA("
            f"hidden_dim={self._hidden_dim}, "
            f"num_heads={self._num_heads})"
        )

