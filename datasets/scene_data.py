"""
datasets.scene_data

Canonical processed scene representation for DSTNet.

Pipeline
--------
RawScene
    ↓
ScenePreprocessor
    ↓
SceneGraphBuilder
    ↓
SceneData
    ↓
Collate
    ↓
DSTNet
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datasets.scene_graph_builder import SceneGraph


###############################################################################
# SceneData
###############################################################################


@dataclass(slots=True)
class SceneData:
    """
    Canonical processed scene.

    All geometric quantities are expressed in the target-agent
    local reference frame.

    The SceneGraph contains the complete interaction graph used by
    the encoder, while this class stores the processed numerical
    features required by the model.
    """

    ###########################################################################
    # Scene Metadata
    ###########################################################################

    sequence_id: str

    city: str

    ###########################################################################
    # Local Reference Frame
    ###########################################################################

    origin: np.ndarray
    """
    Shape (2,)
    """

    heading: float

    ###########################################################################
    # Processed Dynamic Agents
    ###########################################################################

    agents: list[dict[str, Any]]

    ###########################################################################
    # Processed Map Elements
    ###########################################################################

    maps: list[dict[str, Any]]

    ###########################################################################
    # Scene Graph
    ###########################################################################

    scene_graph: SceneGraph

    ###########################################################################
    # Validation
    ###########################################################################

    def __post_init__(
        self,
    ) -> None:

        if self.origin.shape != (2,):
            raise ValueError(
                "origin must have shape (2,)."
            )

        if not isinstance(
            self.heading,
            (
                float,
                np.floating,
            ),
        ):
            raise TypeError(
                "heading must be a float."
            )

        #######################################################################
        # Validate graph consistency
        #######################################################################

        self.scene_graph.validate()

    ###########################################################################
    # Basic Properties
    ###########################################################################

    @property
    def num_agents(
        self,
    ) -> int:

        return len(
            self.agents
        )

    @property
    def num_maps(
        self,
    ) -> int:

        return len(
            self.maps
        )

    @property
    def num_agent_states(
        self,
    ) -> int:

        return self.scene_graph.num_agent_states

    @property
    def num_temporal_edges(
        self,
    ) -> int:

        return self.scene_graph.temporal_edges.shape[1]

    @property
    def num_spatial_edges(
        self,
    ) -> int:

        return self.scene_graph.spatial_edges.shape[1]

    @property
    def num_agent_map_edges(
        self,
    ) -> int:

        return self.scene_graph.agent_map_edges.shape[1]

    @property
    def num_map_map_edges(
        self,
    ) -> int:

        return self.scene_graph.map_map_edges.shape[1]

    ###########################################################################
    # Agent Utilities
    ###########################################################################

    @property
    def target_agent(
        self,
    ) -> dict[str, Any]:
        """
        Return the prediction target agent.
        """

        for agent in self.agents:

            if agent["category"].upper() == "AGENT":
                return agent

        raise RuntimeError(
            "Prediction target not found."
        )

    @property
    def av_agent(
        self,
    ) -> dict[str, Any] | None:
        """
        Return the autonomous vehicle if present.
        """

        for agent in self.agents:

            if agent["object_type"].upper() == "AV":
                return agent

        return None

    ###########################################################################
    # Scene Statistics
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a concise summary of the processed scene.
        """

        return {

            "sequence_id": self.sequence_id,

            "city": self.city,

            "num_agents": self.num_agents,

            "num_maps": self.num_maps,

            "num_agent_states": self.num_agent_states,

            "temporal_edges": self.num_temporal_edges,

            "spatial_edges": self.num_spatial_edges,

            "agent_map_edges": self.num_agent_map_edges,

            "map_map_edges": self.num_map_map_edges,

            "spatial_radius": self.scene_graph.spatial_radius,

            "map_radius": self.scene_graph.map_radius,
        }

    ###########################################################################
    # Serialization
    ###########################################################################

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert SceneData into a dictionary.
        """

        return {

            "sequence_id": self.sequence_id,

            "city": self.city,

            "origin": self.origin,

            "heading": self.heading,

            "agents": self.agents,

            "maps": self.maps,

            "scene_graph": self.scene_graph,
        }

    ###########################################################################
    # Standard Methods
    ###########################################################################

    def __len__(
        self,
    ) -> int:

        return self.num_agents

    def __repr__(
        self,
    ) -> str:

        return (

            "SceneData("

            f"sequence='{self.sequence_id}', "

            f"agents={self.num_agents}, "

            f"maps={self.num_maps}, "

            f"agent_states={self.num_agent_states}, "

            f"temporal_edges={self.num_temporal_edges}, "

            f"spatial_edges={self.num_spatial_edges})"
        )
