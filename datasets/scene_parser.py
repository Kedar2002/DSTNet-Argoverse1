"""
datasets.scene_parser

Parser for the Argoverse 1 Motion Forecasting dataset.

This module converts one CSV file into a RawScene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typing import Any

from datasets.raw_scene import (
    RawLane,
    RawScene,
    RawTrack,
    SceneMetadata,
)


class SceneParser:
    def __init__(self, map_api: Any) -> None:
        self._map = map_api

    ###########################################################################
    # Public API
    ###########################################################################

    def parse(
        self,
        csv_path: str | Path,
    ) -> RawScene:
        """
        Parse a single CSV file.

        Parameters
        ----------
        csv_path
            Path to one Argoverse sequence.

        Returns
        -------
        RawScene
        """

        csv_path = Path(csv_path)

        dataframe = pd.read_csv(csv_path)

        metadata = self._parse_metadata(
            dataframe,
            csv_path,
        )

        scene = RawScene(metadata)

        self._parse_tracks(
            dataframe,
            scene,
        )

        # Lane parsing will be implemented using our custom
        # map loader instead of the deprecated argoverse-api.
        if self._map is not None:
            self._parse_lanes(scene)

        return scene

    ###########################################################################
    # Metadata
    ###########################################################################

    def _parse_metadata(
        self,
        dataframe: pd.DataFrame,
        csv_path: Path,
    ) -> SceneMetadata:
        """
        Parse scene metadata.
        """

        city = dataframe["CITY_NAME"].iloc[0]

        agent_row = dataframe[
            dataframe["OBJECT_TYPE"] == "AGENT"
        ].iloc[0]

        return SceneMetadata(
            sequence_id=csv_path.stem,
            city=city,
            focal_track_id=str(
                agent_row["TRACK_ID"]
            ),
        )

    ###########################################################################
    # Track Parsing
    ###########################################################################

    def _parse_tracks(
        self,
        dataframe: pd.DataFrame,
        scene: RawScene,
    ) -> None:
        """
        Parse all trajectories in the scene.
        """

        grouped = dataframe.groupby("TRACK_ID", sort=False)

        for track_id, track_df in grouped:

            track = self._build_track(
                str(track_id),
                track_df,
            )

            scene.add_track(track)

    def _build_track(
        self,
        track_id: str,
        dataframe: pd.DataFrame,
    ) -> RawTrack:
        """
        Construct a RawTrack from one grouped dataframe.
        """

        dataframe = dataframe.sort_values(
            "TIMESTAMP"
        )

        timestamps = dataframe[
            "TIMESTAMP"
        ].to_numpy(
            dtype=np.float64
        )

        positions = dataframe[
            ["X", "Y"]
        ].to_numpy(
            dtype=np.float32
        )

        object_type = str(
            dataframe["OBJECT_TYPE"].iloc[0]
        )

        #
        # Argoverse-1 category mapping
        #
        # AGENT
        # AV
        # OTHERS
        #

        if object_type == "AGENT":

            category = "AGENT"

        elif object_type == "AV":

            category = "AV"

        else:

            category = "OTHERS"

        return RawTrack(
            track_id=track_id,
            object_type=object_type,
            category=category,
            timestamps=timestamps,
            positions=positions,
        )

        ###########################################################################
    # Lane Parsing
    ###########################################################################

    def _parse_lanes(
        self,
        scene: RawScene,
    ) -> None:
        """
        Parse lane centerlines around the target agent.
        """

        target = scene.target_track

        query_point = target.last_position

        city = scene.metadata.city

        lane_ids = self._get_local_lane_ids(
            query_point,
            city,
        )

        for lane_id in lane_ids:

            lane = self._build_lane(
                lane_id,
                city,
            )

            if lane is not None:

                scene.add_lane(lane)

    def _get_local_lane_ids(
        self,
        position: np.ndarray,
        city: str,
    ) -> list[int]:
        """
        Query nearby lane IDs.
        """

        lane_ids = self._map.get_lane_ids_in_xy_bbox(
            position[0],
            position[1],
            city,
            query_search_range_manhattan=50.0,
        )

        return list(lane_ids)

    def _build_lane(
        self,
        lane_id: int,
        city: str,
    ) -> RawLane | None:
        """
        Build one RawLane from the map.
        """

        lane_segment = self._map.city_lane_centerlines_dict[
            city
        ].get(lane_id)

        if lane_segment is None:
            return None

        centerline = np.asarray(
            lane_segment.centerline,
            dtype=np.float32,
        )

        return RawLane(
            lane_id=lane_id,
            centerline=centerline,
            is_intersection=lane_segment.is_intersection,
            turn_direction=lane_segment.turn_direction,
            has_traffic_control=lane_segment.has_traffic_control,
        )

    def __call__(
        self,
        csv_path: str | Path,
    ) -> RawScene:
        """
        Callable interface.
        """

        return self.parse(csv_path)

