"""
models.encoders.agent_encoder

Agent trajectory encoder for DSTNet.

This module projects an observed trajectory into the latent
feature space using the two-layer MLP described in the paper.

Input
-----
(B, N, T_obs, 2)

Output
------
(B, N, hidden_dim)
"""

from __future__ import annotations
from models.layers.mlp import MLP

import torch
from torch import nn


class AgentEncoder(nn.Module):
    """
    Encode observed agent trajectories.

    Parameters
    ----------
    observation_steps
        Number of observed frames.

    hidden_dim
        Output feature dimension.

    dropout
        Dropout probability.
    """

    def __init__(
        self,
        observation_steps: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self._observation_steps = observation_steps
        self._hidden_dim = hidden_dim
        self._input_dim = observation_steps * 2

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
    def observation_steps(self) -> int:
        return self._observation_steps

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    ###########################################################################
    # Encoding
    ###########################################################################

    def encode(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observed trajectories.

        Parameters
        ----------
        trajectories
            Shape (B, N, T_obs, 2)

        Returns
        -------
        torch.Tensor
            Shape (B, N, hidden_dim)
        """

        if trajectories.ndim != 4:
            raise ValueError(
                "Expected shape (B,N,T,2)."
            )

        batch_size, num_agents, steps, dims = trajectories.shape

        if steps != self._observation_steps:
            raise ValueError(
                f"Expected {self._observation_steps} "
                f"observation steps, got {steps}."
            )

        if dims != 2:
            raise ValueError(
                "Last dimension must be 2."
            )

        x = trajectories.reshape(
            batch_size,
            num_agents,
            self._input_dim,
        )

        return self.encoder(x)

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        """

        return self.encode(
            trajectories,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            "AgentEncoder("
            f"observation_steps={self._observation_steps}, "
            f"hidden_dim={self._hidden_dim})"
        )


