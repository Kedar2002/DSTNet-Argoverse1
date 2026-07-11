"""
models.layers.normalization

Normalization layers used throughout DSTNet.
"""

from __future__ import annotations

import torch
from torch import nn


###############################################################################
# LayerNorm
###############################################################################


class LayerNorm(nn.Module):
    """
    Wrapper around torch.nn.LayerNorm.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ) -> None:

        super().__init__()

        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.norm(x)


###############################################################################
# IdentityNorm
###############################################################################


class IdentityNorm(nn.Module):
    """
    Identity normalization.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return x


###############################################################################
# Factory
###############################################################################


def build_normalization(
    normalization: str,
    hidden_dim: int,
) -> nn.Module:
    """
    Factory for normalization layers.
    """

    normalization = normalization.lower()

    if normalization == "layernorm":

        return LayerNorm(hidden_dim)

    if normalization == "identity":

        return IdentityNorm()

    raise ValueError(
        f"Unknown normalization: {normalization}"
    )
