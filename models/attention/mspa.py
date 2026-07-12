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

from __future__ import annotations

import torch
from torch import nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm


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
