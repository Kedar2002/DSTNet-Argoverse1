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

from typing import Optional

from torch import Tensor, nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.model_types import GraphData


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
        graph: Optional[GraphData],
    ) -> Optional[Tensor]:
        """
        Convert relative feature embeddings into a per-head
        attention bias.

        Returns
        -------
        (B, H, N, N) or None
        """

        if graph is None:
            return None

        embedding = graph.edge_features.embedding

        if embedding is None:
            return None

        # (B,N,N,D) -> (B,N,N,H)
        bias = self.relative_bias(
            embedding,
        )

        # (B,N,N,H) -> (B,H,N,N)
        bias = bias.permute(
            0,
            3,
            1,
            2,
        )

        return bias

    ###########################################################################
    # Attention Mask
    ###########################################################################

    def _build_attention_mask(
        self,
        graph: Optional[GraphData],
    ) -> Optional[Tensor]:
        """
        Convert graph adjacency into an attention mask.

        Returns
        -------
        (B,N,N) or None
        """

        if graph is None:
            return None

        return graph.adjacency.bool()

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        features: Tensor,
        graph: Optional[GraphData] = None,
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

        attention_mask = self._build_attention_mask(
            graph,
        )

        attention_bias = self._build_attention_bias(
            graph,
        )

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


