"""
models.attention.mspa

Multi-Scale Spatial Perception Attention (MSPA)

Paper:
DSTNet: Dynamic Trajectory Prediction via Tri-Attention
Spatio-Temporal Module.

This module models spatial interactions between neighboring
agents using multiple spatial receptive fields.

Input
-----
x : (B, N, C)

Output
------
(B, N, C)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.layers.feed_forward import FeedForward
from models.layers.normalization import build_normalization


class MSPA(nn.Module):
    """
    Multi-Scale Spatial Perception Attention.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_heads
        Number of attention heads.

    dropout
        Dropout probability.

    normalization
        Normalization layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        normalization: str = "layernorm",
    ) -> None:

        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        #
        # Learnable scale weights (Eq. in paper)
        #
        self.scale_weights = nn.Parameter(
            torch.ones(num_heads)
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
    # Helpers
    ###########################################################################

    def _reshape(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert

            (B, N, C)

        into

            (B, H, N, D)
        """

        batch_size, num_tokens, _ = x.shape

        x = x.view(
            batch_size,
            num_tokens,
            self._num_heads,
            self._head_dim,
        )

        return x.transpose(1, 2)

    def _scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention.
        """

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self._head_dim,
        )

        if mask is not None:

            scores = scores.masked_fill(
                ~mask,
                float("-inf"),
            )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention,
        )

        return torch.matmul(
            attention,
            v,
        )

        ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Shape (B, N, C)

        mask
            Optional spatial neighborhood mask.

            Shape (B, H, N, N)
            or broadcast-compatible.
        """

        residual = x

        q = self._reshape(
            self.q_proj(x),
        )

        k = self._reshape(
            self.k_proj(x),
        )

        v = self._reshape(
            self.v_proj(x),
        )

        output = self._scaled_dot_product(
            q,
            k,
            v,
            mask,
        )

        #
        # Learnable per-head weighting
        #

        weights = torch.softmax(
            self.scale_weights,
            dim=0,
        )

        output = output * weights.view(
            1,
            self._num_heads,
            1,
            1,
        )

        output = output.transpose(
            1,
            2,
        )

        output = output.reshape(
            x.shape[0],
            x.shape[1],
            self._hidden_dim,
        )

        output = self.out_proj(
            output,
        )

        output = output + residual

        output = self.norm(
            output,
        )

        output = self.ffn(
            output,
        )

        return output

    def __repr__(
        self,
    ) -> str:

        return (
            "MSPA("
            f"hidden_dim={self._hidden_dim}, "
            f"num_heads={self._num_heads})"
        )


