"""
models.layers.mlp

Generic multilayer perceptron used throughout DSTNet.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class MLP(nn.Module):
    """
    Generic multilayer perceptron.

    Structure

        Linear
            ↓
        Activation
            ↓
        Dropout

    repeated for every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:

        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        dims = [
            input_dim,
            *hidden_dims,
            output_dim,
        ]

        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):

            layers.append(
                nn.Linear(
                    dims[i],
                    dims[i + 1],
                    bias=bias,
                )
            )

            if i != len(dims) - 2:

                layers.append(
                    activation()
                )

                if dropout > 0.0:

                    layers.append(
                        nn.Dropout(
                            dropout
                        )
                    )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)

    @property
    def input_dim(self) -> int:
        return self._input_dim


    @property
    def output_dim(self) -> int:
        return self._output_dim

    def __repr__(self) -> str:

        return (
            f"MLP("
            f"in={self.input_dim}, "
            f"out={self.output_dim})"
        )
