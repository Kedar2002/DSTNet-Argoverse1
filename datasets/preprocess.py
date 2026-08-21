"""
datasets.preprocess

Preprocessing pipeline for DSTNet.

Transforms a RawScene into the canonical SceneData representation.

Pipeline
--------
RawScene
    ↓
Reference Frame Normalization
    ↓
Agent Feature Extraction
Map Feature Extraction
    ↓
SceneGraphBuilder
    ↓
SceneData

Important
---------
The SceneGraph is constructed from the SAME processed agent/map
representations that are supplied to the neural network.

Therefore:

    AgentEncoder
        (N,H,2)
            ↓
        (N,H,D)

and:

    SceneGraph
        N × H agent-state nodes

refer to exactly the same observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from datasets.geometry import (
    compute_acceleration,
    compute_heading,
    compute_headings,
    compute_speed,
    compute_velocity,
    sample_centerline,
    transform_points,
)

from datasets.raw_scene import RawScene
from datasets.scene_data import SceneData
from datasets.scene_graph_builder import SceneGraphBuilder


###############################################################################
# Scene Preprocessor
###############################################################################


class ScenePreprocessor:
    """
    Convert a RawScene into a SceneData object.

    All geometric quantities are transformed into the target-agent
    local reference frame before constructing the SceneGraph.

    The graph is built from the processed observation history,
    not directly from the raw CSV states.
    """

    def __init__(
        self,
        observation_steps: int,
        prediction_steps: int,
        map_sample_points: int,
        spatial_radius: float,
        map_radius: float,
    ) -> None:

        if observation_steps <= 0:
            raise ValueError(
                "observation_steps must be positive."
            )

        if prediction_steps < 0:
            raise ValueError(
                "prediction_steps cannot be negative."
            )

        if map_sample_points <= 0:
            raise ValueError(
                "map_sample_points must be positive."
            )

        self.observation_steps = int(
            observation_steps
        )

        self.prediction_steps = int(
            prediction_steps
        )

        self.map_sample_points = int(
            map_sample_points
        )

        #######################################################################
        # Scene Graph Builder
        #######################################################################

        self.graph_builder = SceneGraphBuilder(
            spatial_radius=spatial_radius,
            map_radius=map_radius,
        )

    ###########################################################################
    # Public API
    ###########################################################################

    def preprocess(
        self,
        scene: RawScene,
    ) -> SceneData:
        """
        Preprocess one RawScene.

        Returns
        -------
        SceneData
            Canonical processed scene representation.
        """

        #######################################################################
        # Reference frame
        #######################################################################

        origin, heading = self._reference_frame(
            scene,
        )

        #######################################################################
        # Process agents
        #######################################################################

        agents = self._process_agents(
            scene=scene,
            origin=origin,
            heading=heading,
        )

        #######################################################################
        # Process maps
        #######################################################################

        maps = self._process_maps(
            scene=scene,
            origin=origin,
            heading=heading,
        )

        #######################################################################
        # Build SceneGraph
        #
        # IMPORTANT:
        #
        # The graph is constructed from the processed representation.
        #
        # Therefore:
        #
        #     graph positions == model input positions
        #
        # and:
        #
        #     graph states == AgentEncoder states
        #######################################################################

        scene_graph = self.graph_builder.build(
            agents=agents,
            maps=maps,
        )

        #######################################################################
        # Explicit graph/model state invariant
        #######################################################################

        expected_agent_states = (
            len(agents)
            * self.observation_steps
        )

        actual_agent_states = (
            scene_graph.num_agent_states
        )

        if actual_agent_states != expected_agent_states:

            raise ValueError(
                "SceneGraph agent-state count does not match "
                "the AgentEncoder observation space: "
                f"expected {len(agents)} × "
                f"{self.observation_steps} = "
                f"{expected_agent_states}, "
                f"got {actual_agent_states}."
            )

        #######################################################################
        # Construct SceneData
        #######################################################################

        return SceneData(
            sequence_id=scene.metadata.sequence_id,
            city=scene.metadata.city,
            origin=origin,
            heading=heading,
            agents=agents,
            maps=maps,
            scene_graph=scene_graph,
        )

    ###########################################################################
    # Local Reference Frame
    ###########################################################################

    def _reference_frame(
        self,
        scene: RawScene,
    ) -> tuple[np.ndarray, float]:
        """
        Compute the target-agent local coordinate frame.

        Origin
        ------
        Last observed position of the prediction target.

        Heading
        -------
        Heading of the prediction target over the observed history.
        """

        target = scene.target_track

        #######################################################################
        # Ensure enough target observations exist
        #######################################################################

        if len(target.positions) < self.observation_steps:

            raise ValueError(
                "Prediction target does not contain enough "
                "observations for the configured observation window: "
                f"required {self.observation_steps}, "
                f"available {len(target.positions)}."
            )

        #######################################################################
        # Observed target trajectory
        #######################################################################

        observed = target.positions[
            : self.observation_steps
        ]

        #######################################################################
        # Local-frame origin
        #######################################################################

        origin = observed[-1]

        #######################################################################
        # Target heading
        #######################################################################

        heading = compute_heading(
            observed,
        )

        return (
            origin.astype(
                np.float32,
            ),
            float(
                heading,
            ),
        )

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
        Process all dynamic agents.

        Every agent contributes exactly ``observation_steps`` states
        to the SceneGraph.

        The processed representation contains:

            observed
            future
            velocity
            acceleration
            speed
            heading
            timestamps

        All positions are expressed in the target-agent local frame.
        """

        processed_agents: list[
            dict[str, Any]
        ] = []

        #######################################################################
        # Iterate over tracks
        #######################################################################

        for track in scene.tracks.values():

            ###################################################################
            # Validate observation availability
            ###################################################################

            if len(track.positions) < self.observation_steps:

                # A track that does not exist for the complete observation
                # window cannot produce the fixed N × H representation
                # required by the current DSTNet implementation.
                continue

            ###################################################################
            # Normalize trajectory
            ###################################################################

            trajectory = transform_points(
                track.positions,
                origin,
                heading,
            ).astype(
                np.float32,
            )

            ###################################################################
            # Observation trajectory
            ###################################################################

            observed = trajectory[
                : self.observation_steps
            ]

            ###################################################################
            # Future trajectory
            ###################################################################

            future = trajectory[
                self.observation_steps:
                self.observation_steps
                + self.prediction_steps
            ]

            ###################################################################
            # Observed timestamps
            ####################################################################

            timestamps = np.asarray(
                track.timestamps[
                    : self.observation_steps
                ],
                dtype=np.float32,
            )

            if len(timestamps) != self.observation_steps:

                raise ValueError(
                    f"Track '{track.track_id}' contains "
                    f"{len(timestamps)} observed timestamps, "
                    f"expected {self.observation_steps}."
                )

            ###################################################################
            # Motion features
            ###################################################################

            velocity = compute_velocity(
                observed,
            ).astype(
                np.float32,
            )

            acceleration = compute_acceleration(
                observed,
            ).astype(
                np.float32,
            )

            speed = compute_speed(
                observed,
            ).astype(
                np.float32,
            )

            headings = compute_headings(
                observed,
            ).astype(
                np.float32,
            )

            ###################################################################
            # Processed agent representation
            ###################################################################

            processed_agents.append(
                {
                    ################################################################
                    # Identity
                    ################################################################

                    "track_id": track.track_id,

                    "object_type": track.object_type,

                    "category": track.category,

                    ################################################################
                    # Trajectory
                    ################################################################

                    "observed": observed,

                    "future": future.astype(
                        np.float32,
                    ),

                    ################################################################
                    # Temporal information
                    ################################################################

                    "timestamps": timestamps,

                    ################################################################
                    # Motion features
                    ################################################################

                    "velocity": velocity,

                    "acceleration": acceleration,

                    "speed": speed,

                    ################################################################
                    # Heading
                    #
                    # Singular key intentionally used here.
                    #
                    # SceneGraphBuilder consumes:
                    #
                    #     agent["heading"]
                    #
                    ################################################################

                    "headings": headings,

                    ################################################################
                    # Convenience current state
                    ################################################################

                    "last_position": observed[-1].copy(),

                    "last_heading": float(
                        headings[-1]
                    ),
                }
            )

        #######################################################################
        # Validate agent collection
        #######################################################################

        if not processed_agents:

            raise ValueError(
                "No agents contain the complete configured "
                "observation history."
            )

        return processed_agents

    ###########################################################################
    # Map Processing
    ###########################################################################

    def _process_maps(
        self,
        scene: RawScene,
        origin: np.ndarray,
        heading: float,
    ) -> list[dict[str, Any]]:
        """
        Normalize and sample map centerlines.

        The resulting map representations are expressed in the
        target-agent local reference frame.
        """

        processed_maps: list[
            dict[str, Any]
        ] = []

        #######################################################################
        # Iterate over lanes
        #######################################################################

        for lane in scene.lanes.values():

            ###################################################################
            # Normalize centerline
            ###################################################################

            centerline = transform_points(
                lane.centerline,
                origin,
                heading,
            )

            ###################################################################
            # Uniform sampling
            ###################################################################

            centerline = sample_centerline(
                centerline,
                self.map_sample_points,
            ).astype(
                np.float32,
            )

            ###################################################################
            # Unit tangent vectors
            ###################################################################

            direction = np.zeros(
                (
                    self.map_sample_points,
                    2,
                ),
                dtype=np.float32,
            )

            if self.map_sample_points > 1:

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

                direction[:-1] = (
                    delta / norms
                )

                direction[-1] = (
                    direction[-2]
                )

            ###################################################################
            # Lane heading
            ###################################################################

            if self.map_sample_points > 1:

                lane_heading = float(
                    np.arctan2(
                        direction[0, 1],
                        direction[0, 0],
                    )
                )

            else:

                lane_heading = 0.0

            ###################################################################
            # Processed map representation
            ###################################################################

            processed_maps.append(
                {
                    "lane_id": int(
                        lane.lane_id
                    ),

                    "centerline": centerline,

                    "centroid": centerline.mean(
                        axis=0
                    ).astype(
                        np.float32,
                    ),

                    "direction": direction,

                    "heading": lane_heading,

                    "is_intersection": (
                        lane.is_intersection
                    ),

                    "turn_direction": (
                        lane.turn_direction
                    ),

                    "has_traffic_control": (
                        lane.has_traffic_control
                    ),
                }
            )

        return processed_maps

    ###########################################################################
    # Callable Interface
    ###########################################################################

    def __call__(
        self,
        scene: RawScene,
    ) -> SceneData:
        """
        Allow:

            preprocessor(scene)

        syntax.
        """

        return self.preprocess(
            scene,
        )


###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "ScenePreprocessor",
]
