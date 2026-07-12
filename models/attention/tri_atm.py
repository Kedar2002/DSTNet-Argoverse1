"""
models.attention.mhca

Multi-Head Cross Attention (MHCA)

Cross-attention block used by DSTNet.

Unlike MSPA, query and key/value come from different feature sets.

Pipeline
--------
Query Features
        │
Key/Value Features
        │
        ▼
Multi-Head Cross Attention
        ▼
Residual
        ▼
LayerNorm
        ▼
Feed Forward
        ▼
Residual
        ▼
LayerNorm
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm

class MHCA(nn.Module):
    """
    Multi-Head Cross Attention.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_heads
        Number of attention heads.

    dropout
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        ###################################################################
        # Cross Attention
        ###################################################################

        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
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
        query_features: torch.Tensor,
        context_features: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query_features
            Shape (B, Nq, C)

        context_features
            Shape (B, Nk, C)

        context_mask
            Shape (B, Nk)

        attention_bias
            Optional relative attention bias.

        Returns
        -------
        Tensor
            Shape (B, Nq, C)
        """

        ###############################################################
        # Cross Attention
        ###############################################################

        if return_attention:

            attended, weights = self.attention(
                query=query_features,
                key=context_features,
                value=context_features,
                mask=context_mask,
                attention_bias=attention_bias,
                return_attention=True,
            )

        else:

            attended = self.attention(
                query=query_features,
                key=context_features,
                value=context_features,
                mask=context_mask,
                attention_bias=attention_bias,
            )

        ###############################################################
        # Residual
        ###############################################################

        features = self.norm1(
            query_features +
            self.dropout(attended)
        )

        ###############################################################
        # Feed Forward
        ###############################################################

        ff = self.feed_forward(
            features,
        )

        ###############################################################
        # Residual
        ###############################################################

        output = self.norm2(
            features +
            self.dropout(ff)
        )

        if return_attention:

            return output, weights

        return output

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            "MHCA("
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads})"
        )

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

"""
models.attention.mspa

Multi-head Spatial Attention (MSPA)

This module performs spatial self-attention over agent features.

Pipeline
--------
Input
    ↓
Multi-Head Self Attention
    ↓
Residual
    ↓
LayerNorm
    ↓
Feed Forward
    ↓
Residual
    ↓
LayerNorm
"""

class MSPA(nn.Module):
    """
    Multi-head Spatial Attention.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_heads
        Number of attention heads.

    dropout
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        ###################################################################
        # Spatial Self-Attention
        ###################################################################

        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
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
        features: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features
            Shape (B, N, C)

        mask
            Shape (B, N)

        attention_bias
            Optional relative positional bias.

        Returns
        -------
        Tensor
            Shape (B, N, C)
        """

        ###################################################################
        # Self Attention
        ###################################################################

        if return_attention:

            attended, weights = self.attention(
                features,
                features,
                features,
                mask=mask,
                attention_bias=attention_bias,
                return_attention=True,
            )

        else:

            attended = self.attention(
                features,
                features,
                features,
                mask=mask,
                attention_bias=attention_bias,
            )

        ###################################################################
        # Residual + Norm
        ###################################################################

        features = self.norm1(
            features +
            self.dropout(attended)
        )

        ###################################################################
        # Feed Forward
        ###################################################################

        ff = self.feed_forward(
            features,
        )

        ###################################################################
        # Residual + Norm
        ###################################################################

        output = self.norm2(
            features +
            self.dropout(ff)
        )

        if return_attention:

            return output, weights

        return output

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            "MSPA("
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads})"
        )

###############################################################################
# Tri-Attention Spatio-Temporal Module
###############################################################################

class TriATM(nn.Module):
    """
    Tri-Attention Spatio-Temporal Module.

    Pipeline

        MSPA
          ↓
        MHCA
          ↓
        MMIA
          ↓
        FeedForward
          ↓
        LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        ###################################################################
        # Three Attention Branches
        ###################################################################

        self.mspa = MSPA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.mhca = MHCA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.mmia = MMIA(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        ###################################################################
        # Single Feed Forward
        ###################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=4,
            dropout=dropout,
        )

        self.norm = LayerNorm(
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
        *,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        ###############################################################
        # Multi-Scale Spatial Attention
        ###############################################################

        spatial = self.mspa(
            agent_features,
            mask=agent_mask,
            attention_bias=attention_bias,
        )

        ###############################################################
        # Historical Cross Attention
        ###############################################################

        cross = self.mhca(
            query_features=spatial,
            context_features=lane_features,
            context_mask=lane_mask,
        )

        ###############################################################
        # Multi-Modal Interaction
        ###############################################################

        fused = self.mmia(
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

        output = self.norm(
            fused +
            self.dropout(ff)
        )

        return (
            output,
            lane_features,
        )

    def __repr__(self) -> str:

        return (
            "TriATM("
            f"hidden_dim={self.hidden_dim})"
        )
