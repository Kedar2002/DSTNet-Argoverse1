"""
models.layers.feed_forward

Transformer Feed Forward Network used throughout DSTNet.
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.normalization import build_normalization


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Architecture

        Linear
            ↓
        GELU
            ↓
        Dropout
            ↓
        Linear
            ↓
        Dropout
            ↓
        Residual
            ↓
        LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.1,
        normalization: str = "layernorm",
    ) -> None:

        super().__init__()

        self._hidden_dim = hidden_dim
        self._expansion = expansion

        inner_dim = hidden_dim * expansion

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm = build_normalization(
            normalization,
            hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.ffn(x)

        x = x + residual

        x = self.norm(x)

        return x

    def __repr__(self) -> str:

        return (
            "FeedForward("
            f"hidden={self._hidden_dim}, "
            f"expansion={self._expansion})"
        )
