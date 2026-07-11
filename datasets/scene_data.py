"""
datasets.scene_data

Processed scene representation for DSTNet.

This module defines the canonical data structure exchanged between
the preprocessing pipeline and the neural network.

Pipeline
--------
RawScene
    ↓
ScenePreprocessor
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

from datasets.graph_builder import GraphData


###############################################################################
# SceneData
###############################################################################

@dataclass(slots=True)
class SceneData:
    """
    Fully processed scene.

    All coordinates are already normalized into the target-agent
    reference frame.
    """

    ###########################################################################
    # Metadata
    ###########################################################################

    sequence_id: str

    city: str

    ###########################################################################
    # Reference Frame
    ###########################################################################

    origin: np.ndarray
    """
    Shape (2,)
    """

    heading: float

    ###########################################################################
    # Processed Agents
    ###########################################################################

    agents: list[dict[str, Any]]

    ###########################################################################
    # Processed Lanes
    ###########################################################################

    lanes: list[dict[str, Any]]

    ###########################################################################
    # Graph
    ###########################################################################

    graph: GraphData

    ###########################################################################
    # Validation
    ###########################################################################

    def __post_init__(self) -> None:

        if self.origin.shape != (2,):
            raise ValueError(
                "origin must have shape (2,)."
            )

        if not isinstance(
            self.heading,
            (float, np.floating),
        ):
            raise TypeError(
                "heading must be a float."
            )

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def num_lanes(self) -> int:
        return len(self.lanes)

    @property
    def num_agent_edges(self) -> int:
        return self.graph.agent_agent_edges.shape[1]

    @property
    def num_lane_edges(self) -> int:
        return self.graph.lane_lane_edges.shape[1]

    @property
    def num_lane_agent_edges(self) -> int:
        return self.graph.lane_agent_edges.shape[1]

        ###########################################################################
    # Agent Utilities
    ###########################################################################

    @property
    def target_agent(self) -> dict[str, Any]:
        """
        Return the prediction target (AGENT).
        """

        for agent in self.agents:

            if agent["category"] == "AGENT":
                return agent

        raise RuntimeError(
            "Target agent not found."
        )

    @property
    def av_agent(self) -> dict[str, Any] | None:
        """
        Return the autonomous vehicle.
        """

        for agent in self.agents:

            if agent["category"] == "AV":
                return agent

        return None

    ###########################################################################
    # Utilities
    ###########################################################################

    def summary(self) -> dict[str, Any]:
        """
        Return scene statistics.
        """

        return {
            "sequence_id": self.sequence_id,
            "city": self.city,
            "num_agents": self.num_agents,
            "num_lanes": self.num_lanes,
            "agent_edges": self.num_agent_edges,
            "lane_edges": self.num_lane_edges,
            "lane_agent_edges": self.num_lane_agent_edges,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the scene into a dictionary.
        """

        return {
            "sequence_id": self.sequence_id,
            "city": self.city,
            "origin": self.origin,
            "heading": self.heading,
            "agents": self.agents,
            "lanes": self.lanes,
            "graph": self.graph,
        }

    def __len__(self) -> int:
        return self.num_agents

    def __repr__(self) -> str:

        return (
            "SceneData("
            f"sequence='{self.sequence_id}', "
            f"agents={self.num_agents}, "
            f"lanes={self.num_lanes}, "
            f"agent_edges={self.num_agent_edges})"
        )


