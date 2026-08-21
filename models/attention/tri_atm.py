"""
models.attention.tri_atm

Tri-Attention Spatio-temporal Module (Tri-ATM).

DSTNet
------
Section III-D

Pipeline

Z_scene
   ↓
MSPA
   ↓
Z^S
   ↓
MHCA
   ↓
Z^ST
   ↓
MMIA
   ↓
Z^STM
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.attention.mspa import MSPA
from models.attention.mhca import MHCA
from models.attention.mmia import MMIA


class TriATM(nn.Module):
    """
    Tri-Attention Spatio-temporal Module.

    The three attention mechanisms operate on different
    dimensions of the scene representation:

        MSPA
            agent dimension

        MHCA
            historical timestep dimension

        MMIA
            prediction-mode dimension
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        interaction_radius: float,
        window_sizes: tuple[int, ...] = (
            2,
            4,
            8,
        ),
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = int(
            hidden_dim
        )

        self.num_heads = int(
            num_heads
        )

        self.interaction_radius = float(
            interaction_radius
        )

        self.window_sizes = tuple(
            int(size)
            for size in window_sizes
        )

        #######################################################################
        # MSPA
        #######################################################################

        self.mspa = MSPA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            interaction_radius=interaction_radius,
            dropout=dropout,
        )

        #######################################################################
        # MHCA
        #######################################################################

        self.mhca = MHCA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_sizes=window_sizes,
            dropout=dropout,
        )

        #######################################################################
        # MMIA
        #######################################################################

        self.mmia = MMIA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        scene_embeddings: Tensor,
        positions: Tensor,
        *,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Apply Tri-ATM.

        Parameters
        ----------
        scene_embeddings
            Z_scene

            Shape:

                (B,N,H,K,D)

        positions
            Current agent positions.

            Shape:

                (B,N,2)

        agent_mask
            Valid-agent mask.

            Shape:

                (B,N)

        Returns
        -------
        Z^STM

            Shape:

                (B,N,H,K,D)
        """

        #######################################################################
        # Input validation
        #######################################################################

        if scene_embeddings.ndim != 5:
            raise ValueError(
                "scene_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        B, N, H, K_modes, D = (
            scene_embeddings.shape
        )

        if D != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {D}."
            )

        if positions.shape != (
            B,
            N,
            2,
        ):
            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        if agent_mask is not None:

            if agent_mask.shape != (
                B,
                N,
            ):
                raise ValueError(
                    "agent_mask must have shape (B,N)."
                )

        #######################################################################
        # 1. MSPA
        #
        # Z_scene → Z^S
        #######################################################################

        spatial_embeddings = self.mspa(
            scene_embeddings,
            positions,
            agent_mask=agent_mask,
        )

        if spatial_embeddings.shape != scene_embeddings.shape:
            raise RuntimeError(
                "MSPA changed the scene representation shape."
            )

        #######################################################################
        # 2. MHCA
        #
        # Z^S → Z^ST
        #######################################################################

        spatio_temporal_embeddings = self.mhca(
            spatial_embeddings,
            agent_mask=agent_mask,
        )

        if spatio_temporal_embeddings.shape != scene_embeddings.shape:
            raise RuntimeError(
                "MHCA changed the scene representation shape."
            )

        #######################################################################
        # 3. MMIA
        #
        # Z^ST → Z^STM
        #######################################################################

        multi_mode_embeddings = self.mmia(
            spatio_temporal_embeddings,
        )

        if multi_mode_embeddings.shape != scene_embeddings.shape:
            raise RuntimeError(
                "MMIA changed the scene representation shape."
            )

        return multi_mode_embeddings

    def extra_repr(self) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"interaction_radius={self.interaction_radius}, "
            f"window_sizes={self.window_sizes}"
        )


__all__ = ["TriATM"]
