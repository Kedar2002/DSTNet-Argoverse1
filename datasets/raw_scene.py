"""
datasets/raw_scene.py

Raw scene representation for the Argoverse 1 Motion Forecasting dataset.

This module defines lightweight dataclasses used immediately after parsing
CSV files. They represent the scene exactly as stored in the dataset before
any preprocessing or tensor conversion.

Pipeline
--------
CSV
    ↓
SceneParser
    ↓
RawScene
    ↓
Geometry
    ↓
Graph Builder
    ↓
SceneData
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


###############################################################################
# Scene Metadata
###############################################################################


@dataclass(slots=True)
class SceneMetadata:
    """
    Metadata describing one Argoverse forecasting scene.
    """

    sequence_id: str
    city: str
    focal_track_id: str

    def __post_init__(self) -> None:

        if not self.sequence_id:
            raise ValueError("sequence_id cannot be empty.")

        if not self.city:
            raise ValueError("city cannot be empty.")

        if not self.focal_track_id:
            raise ValueError("focal_track_id cannot be empty.")


###############################################################################
# Agent Track
###############################################################################


@dataclass(slots=True)
class RawTrack:
    """
    Raw trajectory for one tracked object.

    Coordinates are stored exactly as read from the CSV.
    """

    track_id: str

    object_type: str

    category: str

    timestamps: np.ndarray

    positions: np.ndarray

    def __post_init__(self) -> None:

        if self.timestamps.ndim != 1:
            raise ValueError(
                "timestamps must be 1D."
            )

        if self.positions.ndim != 2:
            raise ValueError(
                "positions must be 2D."
            )

        if self.positions.shape[1] != 2:
            raise ValueError(
                "positions must have shape (N,2)."
            )

        if len(self.timestamps) != len(self.positions):
            raise ValueError(
                "timestamps and positions length mismatch."
            )

    @property
    def length(self) -> int:
        """
        Number of trajectory points.
        """
        return len(self.timestamps)

    @property
    def first_position(self) -> np.ndarray:
        return self.positions[0]

    @property
    def last_position(self) -> np.ndarray:
        return self.positions[-1]

    @property
    def is_target(self) -> bool:
        return self.category.upper() == "AGENT"

    @property
    def is_av(self) -> bool:
        return self.object_type.upper() == "AV"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "object_type": self.object_type,
            "category": self.category,
            "timestamps": self.timestamps,
            "positions": self.positions,
        }

    def to_agent_states(self) -> list[AgentState]:
        """
        Convert this trajectory into a sequence of AgentState objects.

        One AgentState is created for every observed timestamp.
        """

        from datasets.geometry import compute_heading

        states: list[AgentState] = []

        num_points = len(self.positions)

        for timestep in range(num_points):

            if timestep == 0:

                heading = compute_heading(
                    self.positions[:2]
                )

            else:

                heading = compute_heading(
                    self.positions[
                        max(0, timestep - 1): timestep + 1
                    ]
                )

            states.append(

                AgentState(

                    track_id=self.track_id,

                    object_type=self.object_type,

                    category=self.category,

                    timestep=timestep,

                    timestamp=float(self.timestamps[timestep]),

                    position=self.positions[timestep].astype(np.float32),

                    heading=float(heading),
                )

            )

        return states

###############################################################################
# Agent State
###############################################################################


@dataclass(slots=True)
class AgentState:
    """
    One observed state of one moving agent.

    This corresponds to a single node in the Scene Graph used by
    the Global Spatio-Temporal Attention (GSTA) module.

    Unlike RawTrack, which represents an entire trajectory,
    AgentState represents exactly one observed timestamp.

    Paper notation
    --------------
    p_i^s
        Position of agent i at timestep s.

    θ_i^s
        Heading of agent i at timestep s.

    β_i^s
        Timestamp of agent i at timestep s.
    """

    ###########################################################################
    # Identity
    ###########################################################################

    track_id: str

    object_type: str

    category: str

    ###########################################################################
    # Temporal information
    ###########################################################################

    timestep: int

    timestamp: float

    ###########################################################################
    # Geometry
    ###########################################################################

    position: np.ndarray

    heading: float

    def __post_init__(self) -> None:

        if self.position.shape != (2,):
            raise ValueError(
                "position must have shape (2,)."
            )

    @property
    def is_target(self) -> bool:

        return self.category.upper() == "AGENT"

    @property
    def is_av(self) -> bool:

        return self.object_type.upper() == "AV"

    def to_dict(self) -> dict:

        return {

            "track_id": self.track_id,

            "object_type": self.object_type,

            "category": self.category,

            "timestep": self.timestep,

            "timestamp": self.timestamp,

            "position": self.position,

            "heading": self.heading,
        }


###############################################################################
# Lane Centerline
###############################################################################


@dataclass(slots=True)
class RawLane:
    """
    Lane centerline before feature extraction.
    """

    lane_id: int

    centerline: np.ndarray

    is_intersection: bool = False

    turn_direction: str = "NONE"

    has_traffic_control: bool = False

    def __post_init__(self) -> None:

        if self.centerline.ndim != 2:
            raise ValueError(
                "centerline must be 2D."
            )

        if self.centerline.shape[1] != 2:
            raise ValueError(
                "centerline must have shape (N,2)."
            )

    @property
    def num_points(self) -> int:
        return self.centerline.shape[0]

    def to_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "centerline": self.centerline,
            "is_intersection": self.is_intersection,
            "turn_direction": self.turn_direction,
            "has_traffic_control": self.has_traffic_control,
        }


###############################################################################
# Complete Scene
###############################################################################


@dataclass(slots=True)
class RawScene:
    """
    Complete parsed Argoverse scene.

    This class is the output of SceneParser.
    """

    metadata: SceneMetadata

    tracks: Dict[str, RawTrack] = field(
        default_factory=dict
    )

    lanes: Dict[int, RawLane] = field(
        default_factory=dict
    )

    def add_track(
        self,
        track: RawTrack,
    ) -> None:

        self.tracks[track.track_id] = track

    def add_lane(
        self,
        lane: RawLane,
    ) -> None:

        self.lanes[lane.lane_id] = lane

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    @property
    def num_lanes(self) -> int:
        return len(self.lanes)

    @property
    def target_track(self) -> RawTrack:
        """
        Return the focal prediction target.
        """

        return self.tracks[
            self.metadata.focal_track_id
        ]

    @property
    def agent_tracks(self) -> List[RawTrack]:
        """
        Return all dynamic agents.
        """

        return [
            track
            for track in self.tracks.values()
            if track.category.upper() == "AGENT"
        ]

    @property
    def av_track(self) -> Optional[RawTrack]:
        """
        Return autonomous vehicle track.
        """

        for track in self.tracks.values():

            if track.is_av:
                return track

        return None

    @property
    def agent_states(self) -> list[AgentState]:
        """
        Flatten every trajectory into AgentState nodes.

        Returns
        -------
        list[AgentState]
        """

        states: list[AgentState] = []

        for track in self.tracks.values():

            states.extend(
                track.to_agent_states()
            )

        return states

    def clear_lanes(self) -> None:
        self.lanes.clear()

    def clear_tracks(self) -> None:
        self.tracks.clear()

    def summary(self) -> dict:
        """
        Scene statistics.
        """

        return {
            "sequence_id": self.metadata.sequence_id,
            "city": self.metadata.city,
            "tracks": self.num_tracks,
            "lanes": self.num_lanes,
        }

    def __len__(self) -> int:
        return self.num_tracks

    def __repr__(self) -> str:

        return (
            f"RawScene("
            f"sequence={self.metadata.sequence_id}, "
            f"tracks={self.num_tracks}, "
            f"lanes={self.num_lanes})"
        )
