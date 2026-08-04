"""
datasets.graph_builder

Graph construction utilities for DSTNet.

This module builds the geometric interaction graph used by the
DSTNet encoder. Only geometric connectivity is computed here.

No learned features are produced in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.spatial import cKDTree

import numpy as np

from datasets.raw_scene import RawScene
from datasets.geometry import (
    compute_heading,
)


###############################################################################
# Graph Container
###############################################################################


@dataclass(slots=True)
class GraphData:
    """
    Geometric graph representation for a single scene.

    All arrays are NumPy arrays.

    Attributes
    ----------
    agent_agent_edges
        Shape (2,Eaa)

    lane_lane_edges
        Shape (2,Ell)

    lane_agent_edges
        Shape (2,Ela)

    agent_positions
        Shape (Na,2)

    lane_positions
        Shape (Nl,2)

    agent_headings
        Shape (Na,)

    lane_headings
        Shape (Nl,)
    """

    agent_agent_edges: np.ndarray

    lane_lane_edges: np.ndarray

    lane_agent_edges: np.ndarray

    agent_positions: np.ndarray

    lane_positions: np.ndarray

    agent_headings: np.ndarray

    lane_headings: np.ndarray

    ###########################################################################
    # Interaction Radii
    ###########################################################################

    agent_radius: float

    lane_radius: float


###############################################################################
# Graph Builder
###############################################################################


class GraphBuilder:
    """
    Build the geometric interaction graph for one scene.

    This class only computes geometric connectivity and does not
    perform any neural feature extraction.
    """

    def __init__(
        self,
        agent_radius: float,
        lane_radius: float,
    ) -> None:

        self.agent_radius = float(agent_radius)

        self.lane_radius = float(lane_radius)

    ###########################################################################
    # Public API
    ###########################################################################

    def build(
        self,
        scene: RawScene,
    ) -> GraphData:
        """
        Build the geometric graph for one scene.
        """

        #######################################################################
        # Collect agent geometry
        #######################################################################

        agents = list(scene.tracks.values())

        if agents:

            agent_positions = np.stack(
                [
                    agent.last_position.astype(np.float32)
                    for agent in agents
                ],
                axis=0,
            )

            agent_headings = np.asarray(
                [
                    compute_heading(agent.positions)
                    for agent in agents
                ],
                dtype=np.float32,
            )

        else:

            agent_positions = np.empty(
                (0, 2),
                dtype=np.float32,
            )

            agent_headings = np.empty(
                (0,),
                dtype=np.float32,
            )

        #######################################################################
        # Collect lane geometry
        #######################################################################

        lanes = list(scene.lanes.values())

        if lanes:

            lane_positions = np.stack(
                [
                    lane.centerline.mean(axis=0).astype(np.float32)
                    for lane in lanes
                ],
                axis=0,
            )

            lane_headings = np.asarray(
                [
                    compute_heading(lane.centerline)
                    for lane in lanes
                ],
                dtype=np.float32,
            )

        else:

            lane_positions = np.empty(
                (0, 2),
                dtype=np.float32,
            )

            lane_headings = np.empty(
                (0,),
                dtype=np.float32,
            )

        #######################################################################
        # Connectivity
        #######################################################################

        aa = self._build_agent_graph(agent_positions)

        ll = self._build_lane_graph(lane_positions)

        la = self._build_lane_agent_graph(
            agent_positions,
            lane_positions,
        )

        #######################################################################
        # Return graph
        #######################################################################

        return GraphData(
            agent_agent_edges=aa,
            lane_lane_edges=ll,
            lane_agent_edges=la,
            agent_positions=agent_positions,
            lane_positions=lane_positions,
            agent_headings=agent_headings,
            lane_headings=lane_headings,
            agent_radius=self.agent_radius,
            lane_radius=self.lane_radius,
        )

    ###########################################################################
    # Agent Graph
    ###########################################################################

    def _build_agent_graph(
        self,
        agent_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the agent-agent interaction graph.

        Two agents are connected if their Euclidean distance is within the
        configured interaction radius.
        """

        if len(agent_positions) == 0:
            return np.empty((2, 0), dtype=np.int64)

        tree = cKDTree(agent_positions)

        edge_list: list[tuple[int, int]] = []

        for source_idx, position in enumerate(agent_positions):

            neighbors = tree.query_ball_point(
                position,
                r=self.agent_radius,
                p=2.0,
            )

            for target_idx in neighbors:

                if source_idx == target_idx:
                    continue

                edge_list.append(
                    (source_idx, target_idx)
                )

        if not edge_list:
            return np.empty((2, 0), dtype=np.int64)

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T

    ###########################################################################
    # Lane Graph
    ###########################################################################

    def _build_lane_graph(
        self,
        lane_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the lane-lane interaction graph.
        """

        if len(lane_positions) == 0:
            return np.empty((2, 0), dtype=np.int64)

        tree = cKDTree(lane_positions)

        edge_list: list[tuple[int, int]] = []

        for source_idx, center in enumerate(lane_positions):

            neighbors = tree.query_ball_point(
                center,
                r=self.lane_radius,
                p=2.0,
            )

            for target_idx in neighbors:

                if source_idx == target_idx:
                    continue

                edge_list.append(
                    (source_idx, target_idx)
                )

        if not edge_list:
            return np.empty((2, 0), dtype=np.int64)

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T

    ###########################################################################
    # Lane-Agent Graph
    ###########################################################################

    def _build_lane_agent_graph(
        self,
        agent_positions: np.ndarray,
        lane_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the lane-agent interaction graph.

        A lane is connected to an agent when the lane centroid is inside the
        configured interaction radius of the agent.
        """

        if (
            len(agent_positions) == 0
            or len(lane_positions) == 0
        ):
            return np.empty((2, 0), dtype=np.int64)

        lane_tree = cKDTree(lane_positions)

        edge_list: list[tuple[int, int]] = []

        for agent_idx, position in enumerate(agent_positions):

            nearby_lanes = lane_tree.query_ball_point(
                position,
                r=self.lane_radius,
                p=2.0,
            )

            for lane_idx in nearby_lanes:

                edge_list.append(
                    (lane_idx, agent_idx)
                )

        if not edge_list:
            return np.empty((2, 0), dtype=np.int64)

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T

