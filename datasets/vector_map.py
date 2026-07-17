"""
datasets.vector_map

Vector map representation for the custom Argoverse-1 HD maps.

This module stores all LaneSegment objects belonging to one city and
provides efficient lookup and spatial indexing utilities.

Pipeline
--------
HD Map XML
      ↓
MapLoader
      ↓
VectorMap
      ↓
SceneParser
      ↓
ScenePreprocessor
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from datasets.lane_segment import LaneSegment


###############################################################################
# Vector Map
###############################################################################


class VectorMap:
    """
    Vector representation of one Argoverse HD map.

    Parameters
    ----------
    city_name
        Name of the city (e.g. MIA, PIT).

    map_path
        Optional path to the source XML.
    """

    ###########################################################################
    # Construction
    ###########################################################################

    def __init__(
        self,
        city_name: str,
        map_path: str | Path | None = None,
    ) -> None:

        self.city_name = city_name

        self.map_path = (
            Path(map_path)
            if map_path is not None
            else None
        )

        #######################################################################
        # Lane storage
        #######################################################################

        self._lanes: dict[int, LaneSegment] = {}

        #######################################################################
        # Spatial index
        #######################################################################

        self._tree: cKDTree | None = None

        #######################################################################
        # Cached arrays
        #######################################################################

        self._lane_ids: np.ndarray = np.empty(
            0,
            dtype=np.int64,
        )

        self._centroids: np.ndarray = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        self._bbox_min: np.ndarray = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        self._bbox_max: np.ndarray = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        #######################################################################
        # Compatibility dictionary
        #
        # This matches the interface expected by SceneParser:
        #
        # map.city_lane_centerlines_dict[city][lane_id]
        #######################################################################

        self.city_lane_centerlines_dict: dict[
            str,
            dict[int, LaneSegment],
        ] = {

            self.city_name: self._lanes

        }

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def num_lanes(
        self,
    ) -> int:
        """
        Number of lane segments.
        """

        return len(
            self._lanes,
        )

    @property
    def lane_ids(
        self,
    ) -> list[int]:
        """
        Sorted lane IDs.
        """

        return sorted(
            self._lanes.keys(),
        )

    @property
    def lanes(
        self,
    ) -> dict[int, LaneSegment]:
        """
        Underlying lane dictionary.
        """

        return self._lanes

    @property
    def indexed(
        self,
    ) -> bool:
        """
        Whether the KD-tree has been built.
        """

        return self._tree is not None

    ###########################################################################
    # Container API
    ###########################################################################

    def __len__(
        self,
    ) -> int:

        return self.num_lanes

    def __contains__(
        self,
        lane_id: int,
    ) -> bool:

        return lane_id in self._lanes

    def __getitem__(
        self,
        lane_id: int,
    ) -> LaneSegment:

        return self._lanes[lane_id]

    def __iter__(
        self,
    ) -> Iterator[LaneSegment]:

        return iter(
            self._lanes.values(),
        )

    ###########################################################################
    # Lookup
    ###########################################################################

    def get_lane(
        self,
        lane_id: int,
    ) -> LaneSegment | None:
        """
        Safe lane lookup.
        """

        return self._lanes.get(
            lane_id,
        )

    ###########################################################################
    # Lane Management
    ###########################################################################

    def add_lane(
        self,
        lane: LaneSegment,
    ) -> None:
        """
        Add one lane segment.

        Parameters
        ----------
        lane
            LaneSegment object.
        """

        if lane.lane_id in self._lanes:
            raise ValueError(
                f"Duplicate lane id: {lane.lane_id}"
            )

        self._lanes[lane.lane_id] = lane

    def add_lanes(
        self,
        lanes: list[LaneSegment],
    ) -> None:
        """
        Add multiple lane segments.
        """

        for lane in lanes:
            self.add_lane(lane)

    def remove_lane(
        self,
        lane_id: int,
    ) -> None:
        """
        Remove one lane.
        """

        self._lanes.pop(
            lane_id,
            None,
        )

    def clear(
        self,
    ) -> None:
        """
        Remove all lanes and cached geometry.
        """

        self._lanes.clear()

        self._tree = None

        self._lane_ids = np.empty(
            0,
            dtype=np.int64,
        )

        self._centroids = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        self._bbox_min = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        self._bbox_max = np.empty(
            (0, 2),
            dtype=np.float32,
        )

    ###########################################################################
    # KDTree Construction
    ###########################################################################

    def build_index(
        self,
    ) -> None:
        """
        Build the spatial index.

        This should be called once after all lanes
        have been loaded.
        """

        if not self._lanes:

            self._tree = None

            return

        lane_ids: list[int] = []

        centroids: list[np.ndarray] = []

        bbox_min: list[np.ndarray] = []

        bbox_max: list[np.ndarray] = []

        #######################################################################
        # Cache geometry
        #######################################################################

        for lane in self._lanes.values():

            lane_ids.append(
                lane.lane_id,
            )

            centroids.append(
                lane.centroid,
            )

            minimum, maximum = lane.bounding_box()

            bbox_min.append(
                minimum,
            )

            bbox_max.append(
                maximum,
            )

        #######################################################################
        # Convert to contiguous arrays
        #######################################################################

        self._lane_ids = np.asarray(
            lane_ids,
            dtype=np.int64,
        )

        self._centroids = np.asarray(
            centroids,
            dtype=np.float32,
        )

        self._bbox_min = np.asarray(
            bbox_min,
            dtype=np.float32,
        )

        self._bbox_max = np.asarray(
            bbox_max,
            dtype=np.float32,
        )

        #######################################################################
        # Build KDTree
        #######################################################################

        self._tree = cKDTree(
            self._centroids,
        )

    ###########################################################################
    # Cached Geometry
    ###########################################################################

    @property
    def centroids(
        self,
    ) -> np.ndarray:
        """
        Cached lane centroids.

        Shape
        -----
        (num_lanes,2)
        """

        return self._centroids

    @property
    def bounding_boxes(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cached lane bounding boxes.

        Returns
        -------
        minimum
            (num_lanes,2)

        maximum
            (num_lanes,2)
        """

        return (
            self._bbox_min,
            self._bbox_max,
        )

    ###########################################################################
    # Spatial Queries
    ###########################################################################

    def get_lane_ids_in_xy_bbox(
        self,
        x: float,
        y: float,
        query_search_range_manhattan: float = 50.0,
    ) -> list[int]:
        """
        Return nearby lane IDs.

        This mirrors the interface of the deprecated
        ArgoverseMap.get_lane_ids_in_xy_bbox().

        Parameters
        ----------
        x, y
            Query point in city coordinates.

        query_search_range_manhattan
            Search radius (meters).

        Returns
        -------
        list[int]
            Nearby lane IDs.
        """

        if self._tree is None or self.num_lanes == 0:
            return []

        indices = self._tree.query_ball_point(
            [x, y],
            r=query_search_range_manhattan,
            p=2.0,
        )

        return [
            int(self._lane_ids[index])
            for index in indices
        ]

    ###########################################################################
    # Topology
    ###########################################################################

    def predecessors(
        self,
        lane_id: int,
    ) -> list[LaneSegment]:
        """
        Return predecessor lane segments.
        """

        lane = self.get_lane(lane_id)

        if lane is None:
            return []

        return [
            self._lanes[idx]
            for idx in lane.predecessors
            if idx in self._lanes
        ]

    def successors(
        self,
        lane_id: int,
    ) -> list[LaneSegment]:
        """
        Return successor lane segments.
        """

        lane = self.get_lane(lane_id)

        if lane is None:
            return []

        return [
            self._lanes[idx]
            for idx in lane.successors
            if idx in self._lanes
        ]

    def left_neighbor(
        self,
        lane_id: int,
    ) -> LaneSegment | None:
        """
        Return left neighboring lane.
        """

        lane = self.get_lane(lane_id)

        if (
            lane is None
            or lane.left_neighbor_id is None
        ):
            return None

        return self.get_lane(
            lane.left_neighbor_id,
        )

    def right_neighbor(
        self,
        lane_id: int,
    ) -> LaneSegment | None:
        """
        Return right neighboring lane.
        """

        lane = self.get_lane(lane_id)

        if (
            lane is None
            or lane.right_neighbor_id is None
        ):
            return None

        return self.get_lane(
            lane.right_neighbor_id,
        )

    ###########################################################################
    # Utilities
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, object]:
        """
        Return map statistics.
        """

        return {

            "city": self.city_name,

            "map_path": (
                str(self.map_path)
                if self.map_path is not None
                else None
            ),

            "num_lanes": self.num_lanes,

            "indexed": self.indexed,

        }

    def validate(
        self,
    ) -> None:
        """
        Perform lightweight consistency checks.
        """

        for lane in self._lanes.values():

            for predecessor in lane.predecessors:

                if predecessor not in self._lanes:

                    raise ValueError(
                        f"Missing predecessor lane "
                        f"{predecessor} for lane "
                        f"{lane.lane_id}."
                    )

            for successor in lane.successors:

                if successor not in self._lanes:

                    raise ValueError(
                        f"Missing successor lane "
                        f"{successor} for lane "
                        f"{lane.lane_id}."
                    )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (

            "VectorMap("

            f"city='{self.city_name}', "

            f"lanes={self.num_lanes}, "

            f"indexed={self.indexed})"

        )

    
