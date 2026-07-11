"""
datasets.preprocess

Preprocessing pipeline for DSTNet.

Converts RawScene into normalized numerical features.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from datasets.geometry import (
    compute_acceleration,
    compute_heading,
    compute_headings,
    compute_speed,
    compute_velocity,
    sample_centerline,
    transform_points,
)
from datasets.scene_data import SceneData
from datasets.graph_builder import GraphBuilder
from datasets.raw_scene import RawLane, RawScene


class ScenePreprocessor:
    """
    Preprocess one RawScene.
    """

    def __init__(
        self,
        observation_steps: int,
        prediction_steps: int,
        lane_sample_points: int,
        agent_radius: float,
        lane_radius: float,
    ) -> None:

        self.observation_steps = observation_steps

        self.prediction_steps = prediction_steps

        self.lane_sample_points = lane_sample_points

        self.graph_builder = GraphBuilder(
            agent_radius=agent_radius,
            lane_radius=lane_radius,
        )

    ###########################################################################
    # Public API
    ###########################################################################

    def preprocess(
        self,
        scene: RawScene,
    ) -> SceneData:
        """
        Preprocess one scene.

        Returns
        -------
        dict
            Intermediate processed representation.
        """

        origin, heading = self._reference_frame(scene)

        agents = self._process_agents(
            scene,
            origin,
            heading,
        )

        lanes = self._process_lanes(
            scene,
            origin,
            heading,
        )

        graph = self.graph_builder.build(scene)

        return SceneData(
            sequence_id=scene.metadata.sequence_id,
            city=scene.metadata.city,
            origin=origin,
            heading=heading,
            agents=agents,
            lanes=lanes,
            graph=graph,
        )

        ###########################################################################
    # Reference Frame
    ###########################################################################

    def _reference_frame(
        self,
        scene: RawScene,
    ) -> tuple[np.ndarray, float]:
        """
        Compute the local reference frame.

        The reference frame is centered at the target agent's
        last observed position and aligned with its heading.
        """

        target = scene.target_track

        observed = target.positions[: self.observation_steps]

        origin = observed[-1]

        heading = compute_heading(observed)

        return origin.astype(np.float32), heading

    ###########################################################################
    # Agent Processing
    ###########################################################################

    def _process_agents(
        self,
        scene: RawScene,
        origin: np.ndarray,
        heading: float,
    ) -> list[dict[str, Any]]:
        """
        Process all agent trajectories.
        """

        processed_agents: list[dict[str, Any]] = []

        for track in scene.tracks.values():

            trajectory = track.positions.copy()

            trajectory = transform_points(
                trajectory,
                origin,
                heading,
            )

            observed = trajectory[: self.observation_steps]

            future = trajectory[
                self.observation_steps:
                self.observation_steps + self.prediction_steps
            ]

            velocity = compute_velocity(
                observed,
            )

            speed = compute_speed(
                observed,
            )

            acceleration = compute_acceleration(
                observed,
            )

            headings = compute_headings(
                observed,
            )

            processed_agents.append(
                {
                    "track_id": track.track_id,
                    "object_type": track.object_type,
                    "category": track.category,
                    "observed": observed,
                    "future": future,
                    "velocity": velocity,
                    "speed": speed,
                    "acceleration": acceleration,
                    "heading": headings,
                }
            )

        return processed_agents

        ###########################################################################
    # Lane Processing
    ###########################################################################

    def _process_lanes(
        self,
        scene: RawScene,
        origin: np.ndarray,
        heading: float,
    ) -> list[dict[str, Any]]:
        """
        Normalize and sample lane centerlines.

        Returns
        -------
        list[dict]
            Processed lane representations.
        """

        processed_lanes: list[dict[str, Any]] = []

        for lane in scene.lanes.values():

            centerline = transform_points(
                lane.centerline,
                origin,
                heading,
            )

            centerline = sample_centerline(
                centerline,
                self.lane_sample_points,
            )

            direction = np.zeros(
                (
                    self.lane_sample_points,
                    2,
                ),
                dtype=np.float32,
            )

            if self.lane_sample_points > 1:

                delta = (
                    centerline[1:]
                    - centerline[:-1]
                )

                norms = np.linalg.norm(
                    delta,
                    axis=1,
                    keepdims=True,
                )

                norms = np.maximum(
                    norms,
                    1e-8,
                )

                direction[:-1] = delta / norms

                direction[-1] = direction[-2]

            processed_lanes.append(
                {
                    "lane_id": lane.lane_id,
                    "centerline": centerline,
                    "direction": direction,
                    "is_intersection": lane.is_intersection,
                    "turn_direction": lane.turn_direction,
                    "has_traffic_control": lane.has_traffic_control,
                }
            )

        return processed_lanes


