"""
models/gsta.py

Global Spatio-Temporal Attention (GSTA)

Implements Fig.3 and Equations (3)-(9) of DSTNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.layers.transformer import TransformerBlock


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Attention.

    Inputs
    ------
    agent_feat : [N,H,D]
    map_feat   : [M,D]
    rel_feat   : [N,H,D]

    Output
    ------
    scene_feat : [N,H,K,D]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_modes: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_modes = num_modes

        ###################################
        # Temporal branch
        ###################################

        self.temporal_self = TransformerBlock(
            hidden_dim,
            num_heads,
            dropout,
        )

        ###################################
        # Spatial branch
        ###################################

        self.spatial_self = TransformerBlock(
            hidden_dim,
            num_heads,
            dropout,
        )

        ###################################
        # Cross Attention
        ###################################

        self.temporal_cross = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.spatial_cross = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ###################################
        # Learnable Queries
        ###################################

        self.query_embed = nn.Parameter(
            torch.randn(num_modes, hidden_dim)
        )

        ###################################
        # Query Attention
        ###################################

        self.query_temporal = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_spatial = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        agent_feat: torch.Tensor,
        map_feat: torch.Tensor,
        rel_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        agent_feat : [N,H,D]
        map_feat   : [M,D]
        rel_feat   : [N,H,D]

        Returns
        -------
        scene_embedding : [N,H,K,D]
        """

        ####################################
        # Fuse relative embedding
        ####################################

        temporal = agent_feat + rel_feat

        spatial = map_feat.unsqueeze(0)

        ####################################
        # Temporal Self Attention
        ####################################

        temporal = self.temporal_self(temporal)

        ####################################
        # Spatial Self Attention
        ####################################

        spatial = self.spatial_self(spatial)

        ####################################
        # Cross Attention
        ####################################

        temporal2, _ = self.temporal_cross(
            temporal,
            spatial,
            spatial,
        )

        temporal = self.norm_t(
            temporal + temporal2
        )

        spatial2, _ = self.spatial_cross(
            spatial,
            temporal,
            temporal,
        )

        spatial = self.norm_s(
            spatial + spatial2
        )

        ####################################
        # Learnable Queries
        ####################################

        N, H, D = temporal.shape

        query = (
            self.query_embed
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(N, H, self.num_modes, D)
        )

        query = query.reshape(
            N * H,
            self.num_modes,
            D,
        )

        temporal_memory = temporal.reshape(
            N * H,
            1,
            D,
        )

        spatial_memory = (
            spatial.expand(
                N * H,
                spatial.shape[1],
                D,
            )
        )

        ####################################
        # Query ← Temporal
        ####################################

        q_temporal, _ = self.query_temporal(
            query,
            temporal_memory,
            temporal_memory,
        )

        ####################################
        # Query ← Spatial
        ####################################

        q_spatial, _ = self.query_spatial(
            query,
            spatial_memory,
            spatial_memory,
        )

        ####################################
        # Equation (9)
        ####################################

        scene = q_temporal + q_spatial

        scene = scene.reshape(
            N,
            H,
            self.num_modes,
            D,
        )

        return scene
