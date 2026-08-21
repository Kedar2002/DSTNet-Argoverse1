"""
models.encoders.relative_embedding

Relative Spatio-Temporal Embedding (Er) for DSTNet.

This module implements Eq. (2) of the DSTNet paper.

Agent states and lane segments are represented as nodes in the
scene graph. Relative spatial-temporal relationships between
connected nodes are represented as edge features.

For a connected pair of scene elements, the paper defines:

    [ distance,
      sin(delta_heading),
      cos(delta_heading),
      sin(delta_azimuth),
      cos(delta_azimuth),
      delta_time ]

which is projected into the hidden feature dimension using an MLP.

The graph connectivity itself is constructed during preprocessing
and is provided through SceneGraph.

This module performs feature construction and embedding only.
It does not perform attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from datasets.scene_graph_builder import SceneGraph
from models.layers.mlp import MLP


###############################################################################
# Output container
###############################################################################


@dataclass(slots=True)
class RelativeFeatures:
    """
    Relative spatio-temporal embedding associated with graph edges.

    Attributes
    ----------
    edge_index
        Unified graph edge indices.

        Shape:
            (2, U)

    features
        Raw geometric edge features before the MLP.

        Shape:
            (U, 6)

        Feature order:

            0 : distance
            1 : sin(delta_heading)
            2 : cos(delta_heading)
            3 : sin(delta_azimuth)
            4 : cos(delta_azimuth)
            5 : delta_time

    embedding
        Learned relative embedding.

        Shape:
            (U, D)

    edge_type
        Integer edge-type identifier.

        Shape:
            (U,)
    """

    edge_index: torch.Tensor
    features: torch.Tensor
    embedding: torch.Tensor
    edge_type: torch.Tensor


###############################################################################
# Relative Embedding
###############################################################################


class RelativeEmbedding(nn.Module):
    """
    DSTNet Relative Spatio-Temporal Embedding.

    Implements Eq. (2) of the DSTNet paper.

    The module operates on the precomputed SceneGraph rather than
    constructing a dense pairwise tensor.

    Parameters
    ----------
    hidden_dim
        Output embedding dimension.

    dropout
        Dropout probability used by the MLP.
    """

    ###########################################################################
    # Construction
    ###########################################################################

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        #######################################################################
        # Eq. (2)
        #
        # [distance,
        #  sin(delta_heading),
        #  cos(delta_heading),
        #  sin(delta_azimuth),
        #  cos(delta_azimuth),
        #  delta_time]
        #
        # The paper states that these features are passed through an MLP
        # to produce Ef in R^(U x D).
        #######################################################################

        self.embedding = MLP(
            input_dim=6,
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
    ) -> RelativeFeatures:
        """
        Construct the relative spatio-temporal embedding.

        Parameters
        ----------
        scene_graph
            Precomputed DSTNet scene graph.

        Returns
        -------
        RelativeFeatures
            Edge-indexed relative geometric features and their
            learned embeddings.

        Shapes
        -------
        edge_index
            (2, U)

        features
            (U, 6)

        embedding
            (U, hidden_dim)

        edge_type
            (U,)
        """

        if not isinstance(
            scene_graph,
            SceneGraph,
        ):
            raise TypeError(
                "scene_graph must be an instance of "
                "datasets.scene_graph_builder.SceneGraph."
            )

        #######################################################################
        # Build unified edge representation
        #######################################################################

        (
            edge_index,
            features,
            edge_type,
        ) = self._build_edge_features(
            scene_graph,
        )

        #######################################################################
        # Learned embedding
        #######################################################################

        embedding = self.embedding(
            features,
        )

        return RelativeFeatures(
            edge_index=edge_index,
            features=features,
            embedding=embedding,
            edge_type=edge_type,
        )

    ###########################################################################
    # Edge Construction
    ###########################################################################

    def _build_edge_features(
        self,
        scene_graph: SceneGraph,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct relative features for every graph edge.

        Four edge sets are currently represented by SceneGraph:

            temporal
            spatial
            agent-map
            map-map

        The first three contain at least one dynamic agent state.
        Map-map edges are time-invariant.

        Returns
        -------
        edge_index
            Shape (2,U)

        features
            Shape (U,6)

        edge_type
            Shape (U,)
        """

        #######################################################################
        # Individual edge sets
        #######################################################################

        temporal = self._agent_state_edges(
            scene_graph,
            scene_graph.temporal_edges,
            edge_type=0,
            include_time=True,
        )

        spatial = self._agent_state_edges(
            scene_graph,
            scene_graph.spatial_edges,
            edge_type=1,
            include_time=True,
        )

        agent_map = self._agent_map_edges(
            scene_graph,
        )

        map_map = self._map_map_edges(
            scene_graph,
        )

        #######################################################################
        # Concatenate
        #######################################################################

        edge_indices = [
            temporal[0],
            spatial[0],
            agent_map[0],
            map_map[0],
        ]

        features = [
            temporal[1],
            spatial[1],
            agent_map[1],
            map_map[1],
        ]

        edge_types = [
            temporal[2],
            spatial[2],
            agent_map[2],
            map_map[2],
        ]

        edge_index = torch.cat(
            edge_indices,
            dim=1,
        )

        relative_features = torch.cat(
            features,
            dim=0,
        )

        edge_type_tensor = torch.cat(
            edge_types,
            dim=0,
        )

        return (
            edge_index,
            relative_features,
            edge_type_tensor,
        )

    ###########################################################################
    # Agent-State Edges
    ###########################################################################

    def _agent_state_edges(
        self,
        scene_graph: SceneGraph,
        edges: np.ndarray,
        edge_type: int,
        include_time: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct relative features for agent-state edges.

        Both temporal and spatial edges connect agent-state nodes.

        Parameters
        ----------
        edges
            Shape (2,E)

        edge_type
            Integer relation identifier.

        include_time
            Whether the temporal difference is included.
        """

        if edges.shape[1] == 0:

            return self._empty_result()

        source = edges[0]
        target = edges[1]

        source_positions = scene_graph.state_positions[
            source
        ]

        target_positions = scene_graph.state_positions[
            target
        ]

        source_headings = scene_graph.state_headings[
            source
        ]

        target_headings = scene_graph.state_headings[
            target
        ]

        source_times = scene_graph.timestamps[
            source
        ]

        target_times = scene_graph.timestamps[
            target
        ]

        #######################################################################
        # Relative displacement
        #
        # The paper uses the Euclidean norm as the first feature.
        #######################################################################

        relative_position = (
            target_positions
            - source_positions
        )

        distance = np.linalg.norm(
            relative_position,
            axis=-1,
        )

        #######################################################################
        # Heading difference
        #######################################################################

        heading_difference = self._angle_difference(
            target_headings,
            source_headings,
        )

        sin_heading = np.sin(
            heading_difference,
        )

        cos_heading = np.cos(
            heading_difference,
        )

        #######################################################################
        # Relative azimuth
        #
        # For the source node:
        #
        #     beta_source =
        #         atan2(p_source - p_target)
        #         - theta_source
        #
        # For the target node:
        #
        #     beta_target =
        #         atan2(p_target - p_source)
        #         - theta_target
        #
        # Eq. (2) then uses:
        #
        #     sin(beta_target - beta_source)
        #     cos(beta_target - beta_source)
        #######################################################################

        source_azimuth = self._compute_azimuth(
            source_positions,
            target_positions,
            source_headings,
        )

        target_azimuth = self._compute_azimuth(
            target_positions,
            source_positions,
            target_headings,
        )

        azimuth_difference = self._angle_difference(
            target_azimuth,
            source_azimuth,
        )

        sin_azimuth = np.sin(
            azimuth_difference,
        )

        cos_azimuth = np.cos(
            azimuth_difference,
        )

        #######################################################################
        # Temporal difference
        #######################################################################

        if include_time:

            delta_time = (
                target_times
                - source_times
            )

        else:

            delta_time = np.zeros_like(
                distance,
                dtype=np.float32,
            )

        #######################################################################
        # Eq. (2) feature vector
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_heading,
                cos_heading,
                sin_azimuth,
                cos_azimuth,
                delta_time,
            ),
            axis=-1,
        )

        #######################################################################
        # Unified edge index
        #
        # Agent-state nodes occupy the first portion of the unified
        # node index space, so no offset is required here.
        #######################################################################

        edge_index = np.asarray(
            edges,
            dtype=np.int64,
        )

        edge_type_tensor = np.full(
            (
                edges.shape[1],
            ),
            edge_type,
            dtype=np.int64,
        )

        return (
            torch.as_tensor(
                edge_index,
                dtype=torch.long,
            ),
            torch.as_tensor(
                features,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                edge_type_tensor,
                dtype=torch.long,
            ),
        )

    ###########################################################################
    # Agent → Map Edges
    ###########################################################################

    def _agent_map_edges(
        self,
        scene_graph: SceneGraph,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct relative features for agent-map edges.

        Agent nodes occupy:

            [0, num_agent_states)

        Map nodes occupy:

            [num_agent_states,
             num_agent_states + num_map_nodes)

        Map elements are time-invariant, therefore:

            delta_time = 0

        for these edges.
        """

        edges = scene_graph.agent_map_edges

        if edges.shape[1] == 0:

            return self._empty_result()

        source = edges[0]
        target = edges[1]

        source_positions = scene_graph.state_positions[
            source
        ]

        target_positions = scene_graph.map_positions[
            target
        ]

        source_headings = scene_graph.state_headings[
            source
        ]

        target_headings = scene_graph.map_headings[
            target
        ]

        #######################################################################
        # Distance
        #######################################################################

        relative_position = (
            target_positions
            - source_positions
        )

        distance = np.linalg.norm(
            relative_position,
            axis=-1,
        )

        #######################################################################
        # Heading difference
        #######################################################################

        heading_difference = self._angle_difference(
            target_headings,
            source_headings,
        )

        sin_heading = np.sin(
            heading_difference,
        )

        cos_heading = np.cos(
            heading_difference,
        )

        #######################################################################
        # Azimuth difference
        #######################################################################

        source_azimuth = self._compute_azimuth(
            source_positions,
            target_positions,
            source_headings,
        )

        target_azimuth = self._compute_azimuth(
            target_positions,
            source_positions,
            target_headings,
        )

        azimuth_difference = self._angle_difference(
            target_azimuth,
            source_azimuth,
        )

        sin_azimuth = np.sin(
            azimuth_difference,
        )

        cos_azimuth = np.cos(
            azimuth_difference,
        )

        #######################################################################
        # Map elements are time invariant.
        #######################################################################

        delta_time = np.zeros_like(
            distance,
            dtype=np.float32,
        )

        #######################################################################
        # Feature vector
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_heading,
                cos_heading,
                sin_azimuth,
                cos_azimuth,
                delta_time,
            ),
            axis=-1,
        )

        #######################################################################
        # Offset map indices into the unified node space.
        #######################################################################

        unified_edges = np.asarray(
            edges,
            dtype=np.int64,
        ).copy()

        unified_edges[1] += (
            scene_graph.num_agent_states
        )

        edge_type = np.full(
            (
                edges.shape[1],
            ),
            2,
            dtype=np.int64,
        )

        return (
            torch.as_tensor(
                unified_edges,
                dtype=torch.long,
            ),
            torch.as_tensor(
                features,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                edge_type,
                dtype=torch.long,
            ),
        )

    ###########################################################################
    # Map → Map Edges
    ###########################################################################

    def _map_map_edges(
        self,
        scene_graph: SceneGraph,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct relative features for map-map edges.

        Both nodes are time-invariant map elements, so the temporal
        component is zero.
        """

        edges = scene_graph.map_map_edges

        if edges.shape[1] == 0:

            return self._empty_result()

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
        # Distance
        #######################################################################

        relative_position = (
            target_positions
            - source_positions
        )

        distance = np.linalg.norm(
            relative_position,
            axis=-1,
        )

        #######################################################################
        # Heading difference
        #######################################################################

        heading_difference = self._angle_difference(
            target_headings,
            source_headings,
        )

        sin_heading = np.sin(
            heading_difference,
        )

        cos_heading = np.cos(
            heading_difference,
        )

        #######################################################################
        # Azimuth difference
        #######################################################################

        source_azimuth = self._compute_azimuth(
            source_positions,
            target_positions,
            source_headings,
        )

        target_azimuth = self._compute_azimuth(
            target_positions,
            source_positions,
            target_headings,
        )

        azimuth_difference = self._angle_difference(
            target_azimuth,
            source_azimuth,
        )

        sin_azimuth = np.sin(
            azimuth_difference,
        )

        cos_azimuth = np.cos(
            azimuth_difference,
        )

        #######################################################################
        # Time-invariant map pair
        #######################################################################

        delta_time = np.zeros_like(
            distance,
            dtype=np.float32,
        )

        #######################################################################
        # Feature vector
        #######################################################################

        features = np.stack(
            (
                distance,
                sin_heading,
                cos_heading,
                sin_azimuth,
                cos_azimuth,
                delta_time,
            ),
            axis=-1,
        )

        #######################################################################
        # Offset both map nodes into unified node space.
        #######################################################################

        unified_edges = np.asarray(
            edges,
            dtype=np.int64,
        ).copy()

        unified_edges += (
            scene_graph.num_agent_states
        )

        edge_type = np.full(
            (
                edges.shape[1],
            ),
            3,
            dtype=np.int64,
        )

        return (
            torch.as_tensor(
                unified_edges,
                dtype=torch.long,
            ),
            torch.as_tensor(
                features,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                edge_type,
                dtype=torch.long,
            ),
        )

    ###########################################################################
    # Geometry Helpers
    ###########################################################################

    @staticmethod
    def _compute_azimuth(
        source_positions: np.ndarray,
        reference_positions: np.ndarray,
        source_headings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute heading-relative azimuth.

        The paper defines the azimuth of a node relative to the
        other node as the angle of the displacement vector minus
        the node's own heading.

        Parameters
        ----------
        source_positions
            Positions of the node whose azimuth is being computed.

        reference_positions
            Positions of the other node.

        source_headings
            Heading of the source node.

        Returns
        -------
        np.ndarray
            Heading-relative azimuth.
        """

        displacement = (
            source_positions
            - reference_positions
        )

        geometric_angle = np.arctan2(
            displacement[:, 1],
            displacement[:, 0],
        )

        azimuth = (
            geometric_angle
            - source_headings
        )

        return RelativeEmbedding._wrap_angle(
            azimuth,
        )

    ###########################################################################

    @staticmethod
    def _angle_difference(
        target: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        """
        Wrapped angular difference:

            target - source

        mapped to [-pi, pi].
        """

        difference = (
            target
            - source
        )

        return RelativeEmbedding._wrap_angle(
            difference,
        )

    ###########################################################################

    @staticmethod
    def _wrap_angle(
        angle: np.ndarray,
    ) -> np.ndarray:
        """
        Wrap angles to [-pi, pi].
        """

        return (
            np.arctan2(
                np.sin(angle),
                np.cos(angle),
            )
        )

    ###########################################################################
    # Empty Graph Helper
    ###########################################################################

    @staticmethod
    def _empty_result() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Return correctly shaped empty edge tensors.
        """

        return (
            torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
            torch.empty(
                (0, 6),
                dtype=torch.float32,
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
            ),
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "RelativeEmbedding("
            f"hidden_dim={self.hidden_dim}, "
            "input_dim=6)"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "RelativeEmbedding",
    "RelativeFeatures",
]
