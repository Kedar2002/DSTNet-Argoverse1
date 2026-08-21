"""
models.encoders.relative_spatiotemporal_embeddings

Relative Spatio-Temporal Embedding for DSTNet.

Paper
-----
DSTNet, Section III-C, Eq. (2)

The paper represents:

    • agent states
    • lane segments

as graph nodes.

Relative spatial and temporal relationships between connected
scene elements are transformed into edge embeddings.

For dynamic-dynamic relations, Eq. (2) uses:

    ||p_j^s - p_i^t||_2
    sin(theta_j^s - theta_i^t)
    cos(theta_j^s - theta_i^t)
    sin(beta_j^s - beta_i^t)
    cos(beta_j^s - beta_i^t)
    s - t

For relations involving time-invariant map elements, the
temporal term is omitted.

The resulting representation is:

    Er ∈ R^(U × D)

where:

    U = number of connected edges
    D = hidden feature dimension

This module performs no attention.

It only constructs Er.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from datasets.scene_graph_builder import SceneGraph
from models.layers.mlp import MLP
from models.model_types import RelativeSpatioTemporalEmbedding


###############################################################################
# Edge Types
###############################################################################

TEMPORAL_EDGE_TYPE = 0
SPATIAL_EDGE_TYPE = 1
AGENT_MAP_EDGE_TYPE = 2
MAP_MAP_EDGE_TYPE = 3


###############################################################################
# Relative Spatio-Temporal Embedding
###############################################################################


class RelativeSpatioTemporalEmbeddingModule(nn.Module):
    """
    Construct the relative spatio-temporal embedding Er.

    Dynamic-dynamic edges
    ---------------------
    Six input features:

        distance
        sin(delta_heading)
        cos(delta_heading)
        sin(delta_beta)
        cos(delta_beta)
        delta_time

    Map-related edges
    -----------------
    Five input features:

        distance
        sin(delta_heading)
        cos(delta_heading)
        sin(delta_beta)
        cos(delta_beta)

    Separate MLPs are used because time-invariant map relations
    do not contain the temporal factor.

    The graph edge direction is interpreted as:

        source -> target

and therefore:

        source = i at time t
        target = j at time s

so that the paper's temporal term:

        s - t

becomes:

        target_time - source_time
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.hidden_dim = int(
            hidden_dim
        )

        #######################################################################
        # Dynamic relation projection
        #######################################################################

        self.dynamic_embedding = MLP(
            input_dim=6,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Map relation projection
        #######################################################################

        self.map_embedding = MLP(
            input_dim=5,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        scene_graph: SceneGraph,
    ) -> RelativeSpatioTemporalEmbedding:
        """
        Construct Er from a SceneGraph.

        Returns
        -------
        RelativeSpatioTemporalEmbedding

        edge_index
            Shape (2,U).

            Agent-state nodes use indices:

                [0, Ns)

            Map nodes are shifted to:

                [Ns, Ns+Nm)

        embeddings
            Shape (U,D).

        edge_type
            Shape (U,).
        """

        if not isinstance(
            scene_graph,
            SceneGraph,
        ):
            raise TypeError(
                "scene_graph must be a SceneGraph."
            )

        scene_graph.validate()

        #######################################################################
        # Construct the four edge groups
        #######################################################################

        temporal_index, temporal_features = (
            self._build_dynamic_edge_group(
                scene_graph,
                scene_graph.temporal_edges,
            )
        )

        spatial_index, spatial_features = (
            self._build_dynamic_edge_group(
                scene_graph,
                scene_graph.spatial_edges,
            )
        )

        agent_map_index, agent_map_features = (
            self._build_agent_map_edge_group(
                scene_graph,
                scene_graph.agent_map_edges,
            )
        )

        map_map_index, map_map_features = (
            self._build_map_map_edge_group(
                scene_graph,
                scene_graph.map_map_edges,
            )
        )

        #######################################################################
        # Concatenate in deterministic order
        #
        # This order MUST remain synchronized with edge_type.
        #
        # temporal
        # spatial
        # agent-map
        # map-map
        #######################################################################

        edge_indices = (
            temporal_index,
            spatial_index,
            agent_map_index,
            map_map_index,
        )

        embeddings = (
            temporal_features,
            spatial_features,
            agent_map_features,
            map_map_features,
        )

        edge_index = self._concatenate_edge_indices(
            edge_indices
        )

        edge_embeddings = self._concatenate_embeddings(
            embeddings
        )

        edge_type = self._build_edge_types(
            scene_graph=scene_graph,
            device=edge_embeddings.device,
        )

        #######################################################################
        # Final consistency checks
        #######################################################################

        if edge_index.shape[1] != edge_embeddings.shape[0]:
            raise RuntimeError(
                "edge_index and embeddings contain different "
                "numbers of edges."
            )

        if edge_type.shape[0] != edge_embeddings.shape[0]:
            raise RuntimeError(
                "edge_type and embeddings contain different "
                "numbers of edges."
            )

        return RelativeSpatioTemporalEmbedding(
            edge_index=edge_index,
            embeddings=edge_embeddings,
            edge_type=edge_type,
        )

    ###########################################################################
    # Dynamic-Dynamic Relations
    ###########################################################################

    def _build_dynamic_edge_group(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct Er for dynamic-dynamic relations.

        This is used for:

            temporal edges
            spatial edges
        """

        if edges.shape[1] == 0:

            return (
                self._empty_edge_index(),
                self._empty_features(),
            )

        features = self._compute_dynamic_features(
            scene_graph=scene_graph,
            edges=edges,
        )

        edge_index = torch.as_tensor(
            edges,
            dtype=torch.long,
            device=features.device,
        )

        embeddings = self.dynamic_embedding(
            features
        )

        return (
            edge_index,
            embeddings,
        )

    ###########################################################################
    # Agent → Map Relations
    ###########################################################################

    def _build_agent_map_edge_group(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct Er for agent-state → map-node relations.

        Map node indices are shifted by Ns in the unified edge index:

            map_index_unified = Ns + map_index
        """

        if edges.shape[1] == 0:

            return (
                self._empty_edge_index(),
                self._empty_features(),
            )

        features = self._compute_agent_map_features(
            scene_graph=scene_graph,
            edges=edges,
        )

        source = edges[0].astype(
            np.int64,
            copy=False,
        )

        target = (
            scene_graph.num_agent_states
            + edges[1]
        ).astype(
            np.int64,
            copy=False,
        )

        unified_edges = np.stack(
            (
                source,
                target,
            ),
            axis=0,
        )

        edge_index = torch.as_tensor(
            unified_edges,
            dtype=torch.long,
            device=features.device,
        )

        embeddings = self.map_embedding(
            features
        )

        return (
            edge_index,
            embeddings,
        )

    ###########################################################################
    # Map → Map Relations
    ###########################################################################

    def _build_map_map_edge_group(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct Er for map-node → map-node relations.

        Both endpoints are static map nodes, so no temporal
        feature is included.

        Both map indices are shifted by Ns.
        """

        if edges.shape[1] == 0:

            return (
                self._empty_edge_index(),
                self._empty_features(),
            )

        features = self._compute_map_features(
            scene_graph=scene_graph,
            edges=edges,
        )

        offset = scene_graph.num_agent_states

        unified_edges = (
            edges
            + offset
        ).astype(
            np.int64,
            copy=False,
        )

        edge_index = torch.as_tensor(
            unified_edges,
            dtype=torch.long,
            device=features.device,
        )

        embeddings = self.map_embedding(
            features
        )

        return (
            edge_index,
            embeddings,
        )

    ###########################################################################
    # Dynamic Feature Computation
    ###########################################################################

    def _compute_dynamic_features(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute the six-dimensional relation in Eq. (2).

        For:

            source = i at time t
            target = j at time s

        compute:

            ||p_j^s - p_i^t||_2

            sin(theta_j^s - theta_i^t)

            cos(theta_j^s - theta_i^t)

            sin(beta_j^s - beta_i^t)

            cos(beta_j^s - beta_i^t)

            s - t
        """

        source = edges[0]
        target = edges[1]

        positions = scene_graph.state_positions

        headings = scene_graph.state_headings

        timestamps = scene_graph.timestamps

        #######################################################################
        # Relative position
        #######################################################################

        delta_position = (
            positions[target]
            - positions[source]
        )

        distance = np.linalg.norm(
            delta_position,
            axis=1,
        )

        #######################################################################
        # Relative heading
        #######################################################################

        delta_heading = (
            headings[target]
            - headings[source]
        )

        sin_delta_heading = np.sin(
            delta_heading
        )

        cos_delta_heading = np.cos(
            delta_heading
        )

        #######################################################################
        # Relative azimuth beta
        #
        # Paper definition:
        #
        # beta_i^t is the angle from the current instance i
        # toward the other instance, measured relative to the
        # current instance heading.
        #
        # For source:
        #
        #   vector source -> target
        #
        # For target:
        #
        #   vector target -> source
        #######################################################################

        source_to_target_angle = np.arctan2(
            delta_position[:, 1],
            delta_position[:, 0],
        )

        target_to_source_angle = (
            source_to_target_angle
            + np.pi
        )

        beta_source = (
            source_to_target_angle
            - headings[source]
        )

        beta_target = (
            target_to_source_angle
            - headings[target]
        )

        delta_beta = (
            beta_target
            - beta_source
        )

        sin_delta_beta = np.sin(
            delta_beta
        )

        cos_delta_beta = np.cos(
            delta_beta
        )

        #######################################################################
        # Temporal difference
        #######################################################################

        delta_time = (
            timestamps[target]
            - timestamps[source]
        )

        #######################################################################
        # Eq. (2)
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_delta_heading,
                cos_delta_heading,
                sin_delta_beta,
                cos_delta_beta,
                delta_time,
            ),
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        return torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self._parameter_device,
        )

    ###########################################################################
    # Agent → Map Feature Computation
    ###########################################################################

    def _compute_agent_map_features(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute relative spatial features for:

            agent-state -> map-node

        The temporal factor is omitted because the map node is
        time-invariant.
        """

        state_index = edges[0]
        map_index = edges[1]

        agent_positions = scene_graph.state_positions[
            state_index
        ]

        agent_headings = scene_graph.state_headings[
            state_index
        ]

        map_positions = scene_graph.map_positions[
            map_index
        ]

        map_headings = scene_graph.map_headings[
            map_index
        ]

        #######################################################################
        # Relative position
        #######################################################################

        delta_position = (
            map_positions
            - agent_positions
        )

        distance = np.linalg.norm(
            delta_position,
            axis=1,
        )

        #######################################################################
        # Relative heading
        #######################################################################

        delta_heading = (
            map_headings
            - agent_headings
        )

        sin_delta_heading = np.sin(
            delta_heading
        )

        cos_delta_heading = np.cos(
            delta_heading
        )

        #######################################################################
        # Relative azimuth
        #######################################################################

        geometric_angle = np.arctan2(
            delta_position[:, 1],
            delta_position[:, 0],
        )

        beta_agent = (
            geometric_angle
            - agent_headings
        )

        beta_map = (
            geometric_angle
            + np.pi
            - map_headings
        )

        delta_beta = (
            beta_map
            - beta_agent
        )

        sin_delta_beta = np.sin(
            delta_beta
        )

        cos_delta_beta = np.cos(
            delta_beta
        )

        #######################################################################
        # Five-dimensional map relation
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_delta_heading,
                cos_delta_heading,
                sin_delta_beta,
                cos_delta_beta,
            ),
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        return torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self._parameter_device,
        )

    ###########################################################################
    # Map → Map Feature Computation
    ###########################################################################

    def _compute_map_features(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute relative spatial features for map-node pairs.

        The temporal factor is omitted because both nodes are
        time-invariant.
        """

        source = edges[0]

        target = edges[1]

        source_positions = scene_graph.map_positions[
            source
        ]

        target_positions = scene_graph.map_positions[
            target
        ]

        source_headings = scene_graph.map_headings[
            source
        ]

        target_headings = scene_graph.map_headings[
            target
        ]

        #######################################################################
        # Relative position
        #######################################################################

        delta_position = (
            target_positions
            - source_positions
        )

        distance = np.linalg.norm(
            delta_position,
            axis=1,
        )

        #######################################################################
        # Relative heading
        #######################################################################

        delta_heading = (
            target_headings
            - source_headings
        )

        sin_delta_heading = np.sin(
            delta_heading
        )

        cos_delta_heading = np.cos(
            delta_heading
        )

        #######################################################################
        # Relative azimuth
        #######################################################################

        geometric_angle = np.arctan2(
            delta_position[:, 1],
            delta_position[:, 0],
        )

        beta_source = (
            geometric_angle
            - source_headings
        )

        beta_target = (
            geometric_angle
            + np.pi
            - target_headings
        )

        delta_beta = (
            beta_target
            - beta_source
        )

        sin_delta_beta = np.sin(
            delta_beta
        )

        cos_delta_beta = np.cos(
            delta_beta
        )

        #######################################################################
        # Five-dimensional map relation
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_delta_heading,
                cos_delta_heading,
                sin_delta_beta,
                cos_delta_beta,
            ),
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        return torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self._parameter_device,
        )

    ###########################################################################
    # Edge Types
    ###########################################################################

    def _build_edge_types(
        self,
        scene_graph: SceneGraph,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build edge-type labels in exactly the same ordering as
        the concatenated edge embeddings.
        """

        return torch.cat(
            (
                torch.full(
                    (
                        scene_graph.num_temporal_edges,
                    ),
                    TEMPORAL_EDGE_TYPE,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (
                        scene_graph.num_spatial_edges,
                    ),
                    SPATIAL_EDGE_TYPE,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (
                        scene_graph.num_agent_map_edges,
                    ),
                    AGENT_MAP_EDGE_TYPE,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (
                        scene_graph.num_map_map_edges,
                    ),
                    MAP_MAP_EDGE_TYPE,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=0,
        )

    ###########################################################################
    # Concatenation Helpers
    ###########################################################################

    def _concatenate_edge_indices(
        self,
        edge_indices: tuple[
            torch.Tensor,
            ...,
        ],
    ) -> torch.Tensor:

        non_empty = [
            edge_index
            for edge_index in edge_indices
            if edge_index.shape[1] > 0
        ]

        if not non_empty:

            return self._empty_edge_index()

        return torch.cat(
            non_empty,
            dim=1,
        )

    def _concatenate_embeddings(
        self,
        embeddings: tuple[
            torch.Tensor,
            ...,
        ],
    ) -> torch.Tensor:

        non_empty = [
            embedding
            for embedding in embeddings
            if embedding.shape[0] > 0
        ]

        if not non_empty:

            return self._empty_features()

        return torch.cat(
            non_empty,
            dim=0,
        )

    ###########################################################################
    # Empty Tensor Helpers
    ###########################################################################

    def _empty_edge_index(
        self,
    ) -> torch.Tensor:

        return torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
            device=self._parameter_device,
        )

    def _empty_features(
        self,
    ) -> torch.Tensor:

        return torch.empty(
            (
                0,
                self.hidden_dim,
            ),
            dtype=torch.float32,
            device=self._parameter_device,
        )

    ###########################################################################
    # Device
    ###########################################################################

    @property
    def _parameter_device(
        self,
    ) -> torch.device:

        return next(
            self.parameters()
        ).device

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "RelativeSpatioTemporalEmbeddingModule("
            f"hidden_dim={self.hidden_dim})"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "RelativeSpatioTemporalEmbeddingModule",
    "TEMPORAL_EDGE_TYPE",
    "SPATIAL_EDGE_TYPE",
    "AGENT_MAP_EDGE_TYPE",
    "MAP_MAP_EDGE_TYPE",
]
