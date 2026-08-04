"""
models/attention/mspa.py

Multi-head Spatial Pattern Attention (MSPA)

Implements the spatial attention component of the
Tri-Attention Module described in the DSTNet paper.

MSPA performs graph-aware spatial attention using:

    • graph adjacency
    • relative geometric embeddings
    • multi-head attention

Output shape is identical to input.

(B,N,D) -> (B,N,D)
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward


class MSPA(nn.Module):
    """
    Multi-head Spatial Pattern Attention.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_heads
        Number of attention heads.

    expansion
        FeedForward expansion ratio.

    dropout
        Dropout probability.
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

        ###################################################################
        # Spatial Attention
        ###################################################################

        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        ###################################################################
        # Pre-Norm
        ###################################################################

        self.norm = nn.LayerNorm(
            hidden_dim,
        )

        ###################################################################
        # Feed Forward
        ###################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        ###################################################################
        # Relative Bias Projection
        ###################################################################

        self.relative_bias = nn.Linear(
            hidden_dim,
            num_heads,
            bias=False,
        )

    ###########################################################################
    # Relative Attention Bias
    ###########################################################################

    def _build_attention_bias(
        self,
        graph: Any = None,
    ) -> Tensor | None:
        """
        Graph-aware attention bias.

        Temporarily disabled while integrating the
        end-to-end DSTNet pipeline.
        """

        return None

    ###########################################################################
    # Attention Mask
    ###########################################################################

    def _build_attention_mask(
        self,
        graph: Any = None,
    ) -> Tensor | None:
        """
        Graph attention mask.

        Temporarily disabled while integrating the
        end-to-end DSTNet pipeline.
        """

        return None

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        features: Tensor,
        graph: Any = None,
    ) -> Tensor:
        """
        Multi-head Spatial Pattern Attention.

        Parameters
        ----------
        features
            (B,N,D)

        graph
            GraphData

        Returns
        -------
        (B,N,D)
        """

        residual = features

        x = self.norm(
            features,
        )

        attention_mask = None

        attention_bias = None

        x = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attention_mask,
            attention_bias=attention_bias,
        )

        x = residual + x

        x = self.feed_forward(
            x,
        )

        return x

    ###########################################################################
    # Utilities
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}"
        )


