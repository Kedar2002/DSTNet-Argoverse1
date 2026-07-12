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

