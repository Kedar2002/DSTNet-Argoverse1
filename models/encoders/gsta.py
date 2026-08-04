"""
models/encoders/gsta.py

Global Spatio-Temporal Aggregation (GSTA)

Implements the encoder described in Section III-C of the DSTNet paper.

Pipeline
--------
Temporal Self-Attention
        ↓
Spatial Self-Attention
        ↓
Bidirectional Cross Attention
        ↓
Learnable Scene Queries
        ↓
Scene Embedding

The public API intentionally remains

    agent_features, lane_features = gsta(...)

so the existing encoder does not require modification.
"""

from __future__ import annotations

from typing import Optional, Any

import torch
from torch import Tensor, nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.mlp import MLP
from models.model_types import RelativeFeatures


###############################################################################
# Learnable Scene Queries
###############################################################################


class LearnableQueries(nn.Module):
    """
    Learnable scene queries used to aggregate a compact
    global scene representation.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_queries
        Number of learnable latent scene tokens.

    Output
    ------
    (B, Q, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(
            torch.empty(
                num_queries,
                hidden_dim,
            )
        )

        nn.init.xavier_uniform_(self.queries)

    def forward(
        self,
        batch_size: int,
    ) -> Tensor:

        return self.queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

    def extra_repr(self) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_queries={self.num_queries}"
        )


###############################################################################
# Global Spatio-Temporal Aggregation
###############################################################################


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Aggregation.

    Inputs
    ------
    agent_features
        (B, Na, D)

    lane_features
        (B, Nl, D)

    relative
        RelativeFeatures

    graph
        GraphData

    Returns
    -------
    updated_agent_features
        (B, Na, D)

    updated_lane_features
        (B, Nl, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_queries: int = 6,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries

        #######################################################################
        # Learnable Scene Queries
        #######################################################################

        self.scene_queries = LearnableQueries(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
        )

        #######################################################################
        # Temporal Self Attention
        #######################################################################

        self.temporal_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.temporal_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.temporal_ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Spatial Self Attention
        #######################################################################

        self.spatial_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.spatial_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.spatial_ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Agent → Lane Cross Attention
        #######################################################################

        self.agent_to_lane = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.agent_lane_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.agent_lane_ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Lane → Agent Cross Attention
        #######################################################################

        self.lane_to_agent = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.lane_agent_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.lane_agent_ffn = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Learnable Query Attention
        #######################################################################

        self.query_agent_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.query_lane_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        #######################################################################
        # Scene Projection
        #######################################################################

        self.scene_projection = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self._scene_embedding_cache: Optional[Tensor] = None

    ###########################################################################
    # Temporal Self Attention
    ###########################################################################

    def _temporal_self_attention(
        self,
        agent_features: Tensor,
    ) -> Tensor:
        """
        Global temporal aggregation over encoded agent features.

        Input
        -----
        (B, Na, D)

        Output
        ------
        (B, Na, D)
        """

        residual = agent_features

        x = self.temporal_norm(agent_features)

        x = self.temporal_attention(
            query=x,
            key=x,
            value=x,
        )

        x = residual + x

        x = self.temporal_ffn(x)

        return x

    ###########################################################################
    # Spatial Self Attention
    ###########################################################################

    def _spatial_self_attention(
        self,
        lane_features: Tensor,
    ) -> Tensor:
        """
        Global spatial aggregation over encoded lane features.

        Input
        -----
        (B, Nl, D)

        Output
        ------
        (B, Nl, D)
        """

        residual = lane_features

        x = self.spatial_norm(lane_features)

        x = self.spatial_attention(
            query=x,
            key=x,
            value=x,
        )

        x = residual + x

        x = self.spatial_ffn(x)

        return x

    ###########################################################################
    # Agent -> Lane Cross Attention
    ###########################################################################

    def _agent_to_lane_cross_attention(
        self,
        agent_features: Tensor,
        lane_features: Tensor,
    ) -> Tensor:
        """
        Update lane features using agent context.

        Queries
        -------
        Lane features

        Keys / Values
        -------------
        Agent features
        """

        residual = lane_features

        query = self.agent_lane_norm(lane_features)

        key = self.agent_lane_norm(agent_features)

        x = self.agent_to_lane(
            query=query,
            key=key,
            value=key,
        )

        x = residual + x

        x = self.agent_lane_ffn(x)

        return x

    ###########################################################################
    # Lane -> Agent Cross Attention
    ###########################################################################

    def _lane_to_agent_cross_attention(
        self,
        agent_features: Tensor,
        lane_features: Tensor,
    ) -> Tensor:
        """
        Update agent features using lane context.

        Queries
        -------
        Agent features

        Keys / Values
        -------------
        Lane features
        """

        residual = agent_features

        query = self.lane_agent_norm(agent_features)

        key = self.lane_agent_norm(lane_features)

        x = self.lane_to_agent(
            query=query,
            key=key,
            value=key,
        )

        x = residual + x

        x = self.lane_agent_ffn(x)

        return x

    ###########################################################################
    # Query -> Agent Attention
    ###########################################################################

    def _temporal_query_attention(
        self,
        agent_features: Tensor,
    ) -> Tensor:
        """
        Aggregate global temporal scene information using
        learnable scene queries.

        Input
        -----
        agent_features
            (B, Na, D)

        Returns
        -------
        scene_tokens
            (B, Q, D)
        """

        batch_size = agent_features.size(0)

        queries = self.scene_queries(
            batch_size,
        )

        scene_tokens = self.query_agent_attention(
            query=queries,
            key=agent_features,
            value=agent_features,
        )

        return scene_tokens

    ###########################################################################
    # Query -> Lane Attention
    ###########################################################################

    def _spatial_query_attention(
        self,
        lane_features: Tensor,
    ) -> Tensor:
        """
        Aggregate global spatial scene information using
        learnable scene queries.

        Input
        -----
        lane_features
            (B, Nl, D)

        Returns
        -------
        scene_tokens
            (B, Q, D)
        """

        batch_size = lane_features.size(0)

        queries = self.scene_queries(
            batch_size,
        )

        scene_tokens = self.query_lane_attention(
            query=queries,
            key=lane_features,
            value=lane_features,
        )

        return scene_tokens

    ###########################################################################
    # Scene Embedding
    ###########################################################################

    def _scene_embedding(
        self,
        temporal_scene: Tensor,
        spatial_scene: Tensor,
    ) -> Tensor:
        """
        Fuse temporal and spatial scene representations.

        Input
        -----
        temporal_scene
            (B, Q, D)

        spatial_scene
            (B, Q, D)

        Returns
        -------
        scene_embedding
            (B, Q, D)
        """

        scene = temporal_scene + spatial_scene

        scene = self.scene_projection(
            scene,
        )

        return scene

    ###########################################################################
    # Scene Embedding Accessor
    ###########################################################################

    @property
    def scene_embedding(
        self,
    ) -> Optional[Tensor]:
        """
        Returns the scene embedding produced during the most
        recent forward pass.

        Returns
        -------
        Tensor
            (B, Q, D)

        or

        None
        """

        return self._scene_embedding_cache

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        agent_features: Tensor,
        lane_features: Tensor,
        relative: RelativeFeatures | None = None,
        graph: Any = None,
        agent_mask: Tensor | None = None,
        lane_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        Global Spatio-Temporal Aggregation.

        Parameters
        ----------
        agent_features
            Shape (B, Na, D)

        lane_features
            Shape (B, Nl, D)

        Returns
        -------
        Updated

            agent_features

            lane_features
        """

        #######################################################################
        # 1. Temporal Self-Attention
        #######################################################################

        agent_features = self._temporal_self_attention(
            agent_features,
        )

        #######################################################################
        # 2. Spatial Self-Attention
        #######################################################################

        lane_features = self._spatial_self_attention(
            lane_features,
        )

        #######################################################################
        # 3. Bidirectional Cross Attention
        #######################################################################

        lane_features = self._agent_to_lane_cross_attention(
            agent_features,
            lane_features,
        )

        agent_features = self._lane_to_agent_cross_attention(
            agent_features,
            lane_features,
        )

        #######################################################################
        # 4. Global Scene Tokens
        #######################################################################

        temporal_scene = self._temporal_query_attention(
            agent_features,
        )

        spatial_scene = self._spatial_query_attention(
            lane_features,
        )

        #######################################################################
        # 5. Scene Embedding
        #######################################################################

        scene_embedding = self._scene_embedding(
            temporal_scene,
            spatial_scene,
        )

        self._scene_embedding_cache = scene_embedding

        #######################################################################
        # 6. Broadcast Scene Context
        #######################################################################

        scene_context = scene_embedding.mean(
            dim=1,
            keepdim=True,
        )

        agent_features = agent_features + scene_context

        lane_features = lane_features + scene_context

        #######################################################################
        # Output
        #######################################################################

        return (
            agent_features,
            lane_features,
        )
