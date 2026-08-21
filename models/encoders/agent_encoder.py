"""
models.encoders.agent_encoder

Agent Encoder for DSTNet.

The Agent Encoder independently projects every observed agent state
into the hidden feature space while preserving the temporal dimension.

Input
-----
Ea_input ∈ R^(B × N × H × 2)

Output
------
Ea ∈ R^(B × N × H × D)

where

    B : batch size
    N : number of agents
    H : observation history length
    D : hidden feature dimension
"""

from __future__ import annotations

import torch
from torch import nn


###############################################################################
# Agent Encoder
###############################################################################


class AgentEncoder(nn.Module):
    """
    Encode observed agent states independently.

    The same encoder weights are applied to every agent and every
    historical timestep.

    Architecture
    ------------

        (x, y)
           │
           ▼
        Linear
           │
           ▼
       LayerNorm
           │
           ▼
         GELU
           │
           ▼
        Linear
           │
           ▼
          D

    The temporal dimension is never pooled or collapsed.
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
        self._input_dim = 2

        #######################################################################
        # State projection
        #######################################################################

        self.input_projection = nn.Linear(
            self._input_dim,
            hidden_dim,
        )

        #######################################################################
        # Normalization
        #######################################################################

        self.norm = nn.LayerNorm(
            hidden_dim,
        )

        #######################################################################
        # Non-linearity
        #######################################################################

        self.activation = nn.GELU()

        #######################################################################
        # Output projection
        #######################################################################

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        #######################################################################
        # Dropout
        #######################################################################

        self.dropout = nn.Dropout(
            dropout,
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

    @property
    def input_dim(self) -> int:
        return self._input_dim

    ###########################################################################
    # Encoding
    ###########################################################################

    def encode(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observed agent trajectories.

        Parameters
        ----------
        trajectories
            Observed agent positions.

            Shape:

                (B, N, H, 2)

        Returns
        -------
        torch.Tensor

            Agent embeddings Ea.

            Shape:

                (B, N, H, D)
        """

        #######################################################################
        # Validate rank
        #######################################################################

        if trajectories.ndim != 4:

            raise ValueError(
                "Expected trajectories with shape "
                "(B,N,H,2)."
            )

        batch_size, num_agents, history_steps, dimensions = (
            trajectories.shape
        )

        #######################################################################
        # Validate temporal dimension
        #######################################################################

        if history_steps != self._observation_steps:

            raise ValueError(
                f"Expected {self._observation_steps} observation "
                f"steps, got {history_steps}."
            )

        #######################################################################
        # Validate coordinate dimension
        #######################################################################

        if dimensions != self._input_dim:

            raise ValueError(
                "The final trajectory dimension must be 2 "
                "(x,y)."
            )

        #######################################################################
        # Each state is encoded independently.
        #
        # nn.Linear operates only on the final dimension:
        #
        #     (B,N,H,2)
        #          ↓
        #     (B,N,H,D)
        #
        # Therefore no explicit flattening is necessary.
        #######################################################################

        x = self.input_projection(
            trajectories,
        )

        #######################################################################
        # LayerNorm
        #######################################################################

        x = self.norm(
            x,
        )

        #######################################################################
        # GELU
        #######################################################################

        x = self.activation(
            x,
        )

        #######################################################################
        # Output projection
        #######################################################################

        x = self.output_projection(
            x,
        )

        #######################################################################
        # Dropout
        #######################################################################

        x = self.dropout(
            x,
        )

        return x

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns
        -------
        Ea

            Shape:

                (B,N,H,D)
        """

        return self.encode(
            trajectories,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "AgentEncoder("
            f"observation_steps={self._observation_steps}, "
            f"hidden_dim={self._hidden_dim})"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "AgentEncoder",
]
