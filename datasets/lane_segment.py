"""
datasets.lane_segment

Lane segment representation for the custom Argoverse-1 HD Map loader.

This module replaces the LaneSegment class provided by the deprecated
argoverse-api while exposing only the functionality required by DSTNet.

Pipeline
--------
HD Map XML
        ↓
MapLoader
        ↓
LaneSegment
        ↓
SceneParser
        ↓
ScenePreprocessor
        ↓
LaneEncoder
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


###############################################################################
# Lane Segment
###############################################################################


@dataclass(slots=True)
class LaneSegment:
    """
    One lane segment in the Argoverse HD Map.

    Parameters
    ----------
    lane_id
        Unique lane identifier.

    centerline
        Shape (N,2).

    predecessors
        Previous connected lane IDs.

    successors
        Next connected lane IDs.

    left_neighbor_id
        Adjacent lane on the left.

    right_neighbor_id
        Adjacent lane on the right.

    is_intersection
        True if lane belongs to an intersection.

    turn_direction
        LEFT / RIGHT / NONE.

    has_traffic_control
        Whether traffic control exists.
    """

    lane_id: int

    centerline: np.ndarray

    predecessors: list[int] = field(default_factory=list)

    successors: list[int] = field(default_factory=list)

    left_neighbor_id: int | None = None

    right_neighbor_id: int | None = None

    is_intersection: bool = False

    turn_direction: str = "NONE"

    has_traffic_control: bool = False

    ###########################################################################
    # Validation
    ###########################################################################

    def __post_init__(self) -> None:

        self.centerline = np.asarray(
            self.centerline,
            dtype=np.float32,
        )

        if self.centerline.ndim != 2:
            raise ValueError(
                "centerline must have shape (N,2)."
            )

        if self.centerline.shape[1] != 2:
            raise ValueError(
                "centerline must have shape (N,2)."
            )

        self.turn_direction = (
            self.turn_direction.upper()
        )

    ###########################################################################
    # Geometry
    ###########################################################################

    @property
    def num_points(
        self,
    ) -> int:
        """
        Number of centerline points.
        """

        return self.centerline.shape[0]

    @property
    def start(
        self,
    ) -> np.ndarray:
        """
        First point.
        """

        return self.centerline[0]

    @property
    def end(
        self,
    ) -> np.ndarray:
        """
        Last point.
        """

        return self.centerline[-1]

    @property
    def centroid(
        self,
    ) -> np.ndarray:
        """
        Lane centroid.
        """

        return self.centerline.mean(
            axis=0,
        )

    @property
    def length(
        self,
    ) -> float:
        """
        Approximate lane length.
        """

        if self.num_points < 2:
            return 0.0

        delta = np.diff(
            self.centerline,
            axis=0,
        )

        return float(
            np.linalg.norm(
                delta,
                axis=1,
            ).sum()
        )

    @property
    def direction(
        self,
    ) -> np.ndarray:
        """
        Unit direction vector.
        """

        if self.num_points < 2:

            return np.zeros(
                2,
                dtype=np.float32,
            )

        direction = (
            self.end
            - self.start
        )

        norm = np.linalg.norm(
            direction,
        )

        if norm < 1e-8:

            return np.zeros(
                2,
                dtype=np.float32,
            )

        return (
            direction / norm
        ).astype(
            np.float32,
        )

    ###########################################################################
    # Topology
    ###########################################################################

    @property
    def has_predecessor(
        self,
    ) -> bool:

        return len(
            self.predecessors
        ) > 0

    @property
    def has_successor(
        self,
    ) -> bool:

        return len(
            self.successors
        ) > 0

    @property
    def has_left_neighbor(
        self,
    ) -> bool:

        return (
            self.left_neighbor_id
            is not None
        )

    @property
    def has_right_neighbor(
        self,
    ) -> bool:

        return (
            self.right_neighbor_id
            is not None
        )

    ###########################################################################
    # Utilities
    ###########################################################################

    def bounding_box(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute axis-aligned bounding box.

        Returns
        -------
        minimum
            Shape (2,)

        maximum
            Shape (2,)
        """

        return (

            self.centerline.min(
                axis=0,
            ),

            self.centerline.max(
                axis=0,
            ),

        )

    def to_dict(
        self,
    ) -> dict:

        return {

            "lane_id": self.lane_id,

            "centerline": self.centerline,

            "predecessors": self.predecessors,

            "successors": self.successors,

            "left_neighbor_id": self.left_neighbor_id,

            "right_neighbor_id": self.right_neighbor_id,

            "is_intersection": self.is_intersection,

            "turn_direction": self.turn_direction,

            "has_traffic_control": self.has_traffic_control,

        }

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (

            "LaneSegment("

            f"id={self.lane_id}, "

            f"points={self.num_points}, "

            f"length={self.length:.2f}m, "

            f"turn={self.turn_direction}, "

            f"intersection={self.is_intersection})"

        )


