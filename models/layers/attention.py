"""
models.layers.attention

Generic Multi-Head Attention used throughout DSTNet.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head Attention.

    Input
    -----
        Query : (B,N,C)

        Key   : (B,M,C)

        Value : (B,M,C)

    Output
    ------
        (B,N,C)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:

        super().__init__()

        if hidden_dim % num_heads != 0:

            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

        ###########################################################################
    # Helpers
    ###########################################################################

    def _reshape(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, tokens, _ = x.shape

        x = x.view(
            batch_size,
            tokens,
            self._num_heads,
            self._head_dim,
        )

        x = x.transpose(
            1,
            2,
        )

        return x

        ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q = self._reshape(
            self.q_proj(query),
        )

        k = self._reshape(
            self.k_proj(key),
        )

        v = self._reshape(
            self.v_proj(value),
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self._head_dim,
        )

        if mask is not None:

            scores = scores.masked_fill(
                mask == 0,
                float("-inf"),
            )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention,
        )

        output = torch.matmul(
            attention,
            v,
        )

        output = output.transpose(
            1,
            2,
        )

        output = output.reshape(
            query.shape[0],
            query.shape[1],
            self._hidden_dim,
        )

        output = self.out_proj(
            output,
        )

        return output

    def __repr__(
        self,
    ) -> str:

        return (
            "MultiHeadAttention("
            f"hidden_dim={self._hidden_dim}, "
            f"num_heads={self._num_heads})"
        )


