"""
models.attention.tri_atm

Tri-Attention Module (Tri-ATM)

DSTNet
Section III-D

Pipeline

Agent Features
Lane Features
        │
        ▼
MSPA
        │
        ▼
MHCA
        │
        ▼
MMIA
        │
        ▼
Updated Features
"""

from __future__ import annotations

from torch import Tensor, nn

from models.attention.mspa import MSPA
from models.attention.mhca import MHCA
from models.attention.mmia import MMIA

from typing import Any


class TriATM(nn.Module):
    """
    Tri-Attention Module.

    Sequentially applies

        1. Multi-head Spatial Pattern Attention (MSPA)

        2. Multi-head Historical Context Attention (MHCA)

        3. Multi-modal Interaction Attention (MMIA)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_sizes: tuple[int, ...] = (
            2,
            4,
            8,
        ),
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        #######################################################################
        # Spatial Pattern Attention
        #######################################################################

        self.mspa = MSPA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Historical Context Attention
        #######################################################################

        self.mhca = MHCA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_sizes=window_sizes,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Multi-modal Interaction Attention
        #######################################################################

        self.mmia = MMIA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        agent_features: Tensor,
        lane_features: Tensor,
        graph: Any | None = None,
        agent_mask: Tensor | None = None,
        lane_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        Parameters
        ----------
        agent_features
            Shape (B,N,C)

        lane_features
            Shape (B,L,C)

        Returns
        -------
        Updated

            agent_features

            lane_features
        """

        #######################################################################
        # MSPA
        #######################################################################

        agent_features = self.mspa(
            features=agent_features,
            graph=graph,
        )

        #######################################################################
        # MHCA
        #######################################################################
        #
        # MHCA operates on temporal features.
        # If temporal dimension already exists:
        #
        #     (B,N,T,C)
        #
        # it can be passed directly.
        #
        # Otherwise temporarily insert a singleton
        # temporal dimension.
        #######################################################################

        if agent_features.ndim == 3:

            temporal = agent_features.unsqueeze(2)

            temporal = self.mhca(
                temporal,
            )

            agent_features = temporal.squeeze(2)

        else:

            agent_features = self.mhca(
                agent_features,
            )

        #######################################################################
        # MMIA
        #######################################################################

        agent_features, lane_features = self.mmia(
            agent_features=agent_features,
            lane_features=lane_features,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )

        return (
            agent_features,
            lane_features,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}"
        )
