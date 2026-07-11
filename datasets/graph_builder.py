"""
datasets.graph_builder

Graph construction utilities for DSTNet.

This module converts a RawScene into graph connectivity
used by the encoder.
"""

from __future__ import annotations
from scipy.spatial import cKDTree

from dataclasses import dataclass

import numpy as np

from datasets.raw_scene import RawScene


###############################################################################
# Graph Container
###############################################################################


@dataclass(slots=True)
class GraphData:
    """
    Graph representation for one scene.

    Attributes
    ----------
    agent_agent_edges
        Shape (2, Eaa)

    lane_lane_edges
        Shape (2, Ell)

    lane_agent_edges
        Shape (2, Ela)
    """

    agent_agent_edges: np.ndarray

    lane_lane_edges: np.ndarray

    lane_agent_edges: np.ndarray


###############################################################################
# Graph Builder
###############################################################################


class GraphBuilder:
    """
    Build graph connectivity from a RawScene.
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
        Build all graph connections.
        """

        aa = self._build_agent_graph(scene)

        ll = self._build_lane_graph(scene)

        la = self._build_lane_agent_graph(scene)

        return GraphData(
            agent_agent_edges=aa,
            lane_lane_edges=ll,
            lane_agent_edges=la,
        )

        ###########################################################################
    # Agent Graph
    ###########################################################################

    def _build_agent_graph(
        self,
        scene: RawScene,
    ) -> np.ndarray:
        """
        Construct the agent-agent interaction graph using a KD-tree.
        """

        agents = list(scene.tracks.values())

        if not agents:
            return np.empty((2, 0), dtype=np.int64)

        positions = np.stack(
            [agent.last_position for agent in agents],
            axis=0,
        )

        tree = cKDTree(positions)

        edge_list: list[tuple[int, int]] = []

        for source_idx, position in enumerate(positions):

            neighbors = tree.query_ball_point(
                position,
                r=self.agent_radius,
                p = 2.0,
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
        scene: RawScene,
    ) -> np.ndarray:
        """
        Construct the lane-lane graph using KD-tree search.
        """

        lanes = list(scene.lanes.values())

        if not lanes:
            return np.empty((2, 0), dtype=np.int64)

        centers = np.stack(
            [
                lane.centerline.mean(axis=0)
                for lane in lanes
            ],
            axis=0,
        )

        tree = cKDTree(centers)

        edge_list: list[tuple[int, int]] = []

        for source_idx, center in enumerate(centers):

            neighbors = tree.query_ball_point(
                center,
                r=self.lane_radius,
                p = 2.0,
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
    # Agent-Lane Graph
    ###########################################################################

    def _build_lane_agent_graph(
        self,
        scene: RawScene,
    ) -> np.ndarray:
        """
        Construct the lane-agent interaction graph.

        An edge is created if an agent's final observed position is
        within the lane interaction radius of the lane centroid.
        """

        agents = list(scene.tracks.values())
        lanes = list(scene.lanes.values())

        if not agents or not lanes:
            return np.empty((2, 0), dtype=np.int64)

        agent_positions = np.stack(
            [agent.last_position for agent in agents],
            axis=0,
        )

        lane_centers = np.stack(
            [
                lane.centerline.mean(axis=0)
                for lane in lanes
            ],
            axis=0,
        )

        lane_tree = cKDTree(lane_centers)

        edge_list: list[tuple[int, int]] = []

        for agent_idx, position in enumerate(agent_positions):

            nearby_lanes = lane_tree.query_ball_point(
                position,
                r=self.lane_radius,
                p = 2.0,
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
