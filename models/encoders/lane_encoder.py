"""
models.encoders.lane_encoder

Lane encoder for DSTNet.

This module projects sampled lane centerlines into the latent
feature space using a two-layer MLP.

Input
-----
(B, L, P, 2)

Output
------
(B, L, hidden_dim)
"""

from __future__ import annotations

import torch
from torch import nn

from models.layers.mlp import MLP


class LaneEncoder(nn.Module):
    """
    Encode lane centerlines.

    Parameters
    ----------
    num_points
        Number of sampled centerline points.

    hidden_dim
        Output feature dimension.

    dropout
        Dropout probability.
    """

    def __init__(
        self,
        num_points: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self._num_points = num_points
        self._hidden_dim = hidden_dim
        self._input_dim = num_points * 2

        self.encoder = MLP(
            input_dim=self._input_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    ###########################################################################
    # Encoding
    ###########################################################################

    def encode(
        self,
        centerlines: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode lane centerlines.

        Parameters
        ----------
        centerlines
            Shape (B, L, P, 2)

        Returns
        -------
        torch.Tensor
            Shape (B, L, hidden_dim)
        """

        if centerlines.ndim != 4:
            raise ValueError(
                "Expected shape (B,L,P,2)."
            )

        batch_size, num_lanes, num_points, dims = centerlines.shape

        if num_points != self._num_points:
            raise ValueError(
                f"Expected {self._num_points} sampled points, "
                f"got {num_points}."
            )

        if dims != 2:
            raise ValueError(
                "Last dimension must equal 2."
            )

        x = centerlines.reshape(
            batch_size,
            num_lanes,
            self._input_dim,
        )

        return self.encoder(
            x,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        centerlines: torch.Tensor,
    ) -> torch.Tensor:

        return self.encode(
            centerlines,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            "LaneEncoder("
            f"num_points={self._num_points}, "
            f"hidden_dim={self._hidden_dim})"
        )
