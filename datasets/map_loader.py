"""
datasets.map_loader

Custom HD map loader for Argoverse 1 Motion Forecasting.

This module replaces the deprecated argoverse.map_representation
API by parsing the official Argoverse OSM-style vector map XML files.

Responsibilities
----------------
- Discover HD map XML files
- Parse OSM nodes
- Parse lane segments (ways)
- Build VectorMap objects
- Build spatial indices
- Provide ArgoverseMap-compatible query API

Pipeline
--------
HD Map XML
      │
      ▼
MapLoader
      │
      ▼
VectorMap
      │
      ▼
SceneParser
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from datasets.lane_segment import LaneSegment
from datasets.vector_map import VectorMap


###############################################################################
# XML Constants
###############################################################################

NODE_TAG = "node"

WAY_TAG = "way"

TAG_TAG = "tag"

ND_TAG = "nd"


###############################################################################
# XML Helper Structures
###############################################################################


@dataclass(slots=True)
class OSMNode:
    """
    Temporary OSM node.

    Parameters
    ----------
    node_id
        OSM node identifier.

    x
        Global x coordinate.

    y
        Global y coordinate.
    """

    node_id: int

    x: float

    y: float

    @property
    def position(
        self,
    ) -> np.ndarray:

        return np.asarray(
            (
                self.x,
                self.y,
            ),
            dtype=np.float32,
        )


###############################################################################
# Map Loader
###############################################################################


class MapLoader:
    """
    Custom Argoverse HD Map loader.

    Parameters
    ----------
    map_root

        Directory containing

            pruned_argoverse_*_vector_map.xml
    """

    ###########################################################################
    # Construction
    ###########################################################################

    def __init__(
        self,
        map_root: str | Path,
    ) -> None:

        self.map_root = Path(
            map_root,
        )

        if not self.map_root.exists():

            raise FileNotFoundError(
                self.map_root,
            )

        #######################################################################
        # Loaded vector maps
        #######################################################################

        self._maps: dict[
            str,
            VectorMap,
        ] = {}

        #######################################################################
        # SceneParser compatibility
        #######################################################################

        self.city_lane_centerlines_dict: dict[
            str,
            dict[int, LaneSegment],
        ] = {}

        #######################################################################
        # OSM node cache
        #
        # Used while parsing one XML.
        #######################################################################

        self._nodes: dict[
            int,
            OSMNode,
        ] = {}

        #######################################################################
        # Automatically load every map
        #######################################################################

        self.load_all_maps()

    ###########################################################################
    # Map Discovery
    ###########################################################################

    def discover_maps(
        self,
    ) -> list[Path]:
        """
        Discover every HD map XML.
        """

        xml_files = sorted(

            self.map_root.glob(
                "pruned_argoverse_*_vector_map.xml",
            )

        )

        if not xml_files:

            raise RuntimeError(

                "No vector map XML files found in "

                f"{self.map_root}"

            )

        return xml_files

    ###########################################################################
    # Utilities
    ###########################################################################

    @staticmethod
    def infer_city_name(
        xml_path: Path,
    ) -> str:
        """
        Infer city name from filename.

        Example
        -------

        pruned_argoverse_MIA_10316_vector_map.xml

        →

        MIA
        """

        match = re.search(

            r"argoverse_([A-Za-z]+)_",

            xml_path.name,

        )

        if match is None:

            raise RuntimeError(

                f"Cannot determine city "

                f"from {xml_path.name}"

            )

        return match.group(
            1,
        ).upper()

    ###########################################################################
    # Public API
    ###########################################################################

    @property
    def cities(
        self,
    ) -> list[str]:
        """
        Loaded city names.
        """

        return sorted(
            self._maps.keys(),
        )

    def get_vector_map(
        self,
        city: str,
    ) -> VectorMap:
        """
        Retrieve the VectorMap for one city.
        """

        city = city.upper()

        if city not in self._maps:

            raise KeyError(
                f"Unknown city '{city}'."
            )

        return self._maps[city]

    def has_city(
        self,
        city: str,
    ) -> bool:

        return city.upper() in self._maps

    ###########################################################################
    # Loading
    ###########################################################################

    def load_all_maps(
        self,
    ) -> None:
        """
        Load every available Argoverse HD map.
        """

        self._maps.clear()

        self.city_lane_centerlines_dict.clear()

        for xml_path in self.discover_maps():

            city = self.infer_city_name(
                xml_path,
            )

            vector_map = self.load_city(
                city=city,
                xml_path=xml_path,
            )

            self._maps[city] = vector_map

            ###################################################################
            # Compatibility with SceneParser
            ###################################################################

            self.city_lane_centerlines_dict[
                city
            ] = vector_map.lanes

    ###########################################################################

    def load_city(
        self,
        *,
        city: str,
        xml_path: Path,
    ) -> VectorMap:
        """
        Load one city's HD map.
        """

        #######################################################################
        # Reset temporary node cache
        #######################################################################

        self._nodes.clear()

        #######################################################################
        # Parse XML
        #######################################################################

        tree = ET.parse(
            xml_path,
        )

        root = tree.getroot()

        #######################################################################
        # Parse nodes
        #######################################################################

        self._parse_nodes(
            root,
        )

        #######################################################################
        # Parse lane segments
        #######################################################################

        vector_map = VectorMap(
            city_name=city,
            map_path=xml_path,
        )

        for element in root.findall(
            WAY_TAG,
        ):

            lane = self._parse_lane(
                element,
            )

            vector_map.add_lane(
                lane,
            )

        #######################################################################
        # Build KD-tree
        #######################################################################

        vector_map.build_index()

        #######################################################################
        # Validate topology
        #######################################################################

        vector_map.validate()

        return vector_map

    ###########################################################################
    # Node Parsing
    ###########################################################################

    def _parse_nodes(
        self,
        root: ET.Element,
    ) -> None:
        """
        Parse every OSM node.

        Each node stores one global (x,y) coordinate.
        """

        self._nodes.clear()

        for node in root.findall(
            NODE_TAG,
        ):

            node_id = int(
                node.attrib["id"]
            )

            x = float(
                node.attrib["x"]
            )

            y = float(
                node.attrib["y"]
            )

            self._nodes[
                node_id
            ] = OSMNode(
                node_id=node_id,
                x=x,
                y=y,
            )

    ###########################################################################
    # Node Utilities
    ###########################################################################

    def _node_position(
        self,
        node_id: int,
    ) -> np.ndarray:
        """
        Retrieve one node position.

        Raises
        ------
        KeyError
            If the node is missing.
        """

        try:

            return self._nodes[
                node_id
            ].position

        except KeyError as exc:

            raise KeyError(
                f"Node {node_id} not found "
                "in current map."
            ) from exc

    ###########################################################################
    # Lane Parsing
    ###########################################################################

    def _parse_lane(
        self,
        way: ET.Element,
    ) -> LaneSegment:
        """
        Parse one OSM <way> element into a LaneSegment.
        """

        #######################################################################
        # Lane ID
        #######################################################################

        if "lane_id" not in way.attrib:
            raise ValueError(
                "Way element missing lane_id attribute."
            )

        lane_id = int(
            way.attrib["lane_id"]
        )

        #######################################################################
        # Parse child elements
        #######################################################################

        node_ids: list[int] = []

        tags: dict[str, str] = {}

        for child in way:

            ###############################################################
            # Centerline nodes
            ###############################################################

            if child.tag == ND_TAG:

                ref = child.attrib.get("ref")

                if ref is None:
                    continue

                node_ids.append(
                    int(ref)
                )

                continue

            ###############################################################
            # Semantic tags
            ###############################################################

            if child.tag == TAG_TAG:

                key = child.attrib.get("k")

                value = child.attrib.get("v")

                if key is None or value is None:
                    continue

                tags[key] = value

        #######################################################################
        # Build lane geometry
        #######################################################################

        centerline = self._parse_centerline(
            node_ids,
        )

        #######################################################################
        # Build topology
        #######################################################################

        predecessors = self._parse_id_list(
            tags.get("predecessor")
        )

        successors = self._parse_id_list(
            tags.get("successor")
        )

        left_neighbor = self._parse_optional_id(
            tags.get("l_neighbor_id")
        )

        right_neighbor = self._parse_optional_id(
            tags.get("r_neighbor_id")
        )

        #######################################################################
        # Semantic attributes
        #######################################################################

        turn_direction = tags.get(
            "turn_direction",
            "NONE",
        ).upper()

        is_intersection = self._parse_bool(
            tags.get(
                "is_intersection",
                "False",
            )
        )

        has_traffic_control = self._parse_bool(
            tags.get(
                "has_traffic_control",
                "False",
            )
        )

        #######################################################################
        # Construct LaneSegment
        #######################################################################

        return LaneSegment(

            lane_id=lane_id,

            centerline=centerline,

            predecessors=predecessors,

            successors=successors,

            left_neighbor_id=left_neighbor,

            right_neighbor_id=right_neighbor,

            is_intersection=is_intersection,

            turn_direction=turn_direction,

            has_traffic_control=has_traffic_control,
        )

    ###########################################################################
    # Centerline Parsing
    ###########################################################################

    def _parse_centerline(
        self,
        node_ids: list[int],
    ) -> np.ndarray:
        """
        Convert node references into a centerline.
        """

        if len(node_ids) < 2:

            raise ValueError(
                "Lane must contain at least two nodes."
            )

        coordinates = [

            self._node_position(
                node_id,
            )

            for node_id in node_ids

        ]

        return np.asarray(
            coordinates,
            dtype=np.float32,
        )

    ###########################################################################
    # XML Helper Parsers
    ###########################################################################

    @staticmethod
    def _parse_bool(
        value: str | None,
    ) -> bool:
        """
        Parse an XML boolean.

        Accepted values
        ---------------
        True
        False
        true
        false
        1
        0
        yes
        no
        """

        if value is None:
            return False

        value = value.strip().lower()

        return value in (
            "true",
            "1",
            "yes",
        )

    ###########################################################################

    @staticmethod
    def _parse_optional_id(
        value: str | None,
    ) -> int | None:
        """
        Parse an optional integer lane ID.

        Examples
        --------
        "12345" -> 12345

        "None" -> None

        "" -> None
        """

        if value is None:
            return None

        value = value.strip()

        if value == "":
            return None

        if value.lower() == "none":
            return None

        return int(value)

    ###########################################################################

    @staticmethod
    def _parse_id_list(
        value: str | None,
    ) -> list[int]:
        """
        Parse predecessor/successor IDs.

        Supported formats
        -----------------

        None

        ""

        "123"

        "123,456"

        "123 456"

        "123;456"
        """

        if value is None:
            return []

        value = value.strip()

        if value == "":
            return []

        if value.lower() == "none":
            return []

        #######################################################################
        # Normalize delimiters
        #######################################################################

        value = (
            value
            .replace(";", ",")
            .replace(" ", ",")
        )

        ids: list[int] = []

        for token in value.split(","):

            token = token.strip()

            if token == "":
                continue

            ids.append(
                int(token)
            )

        return ids

    ###########################################################################
    # Lane Utilities
    ###########################################################################

    def get_lane_segment(
        self,
        lane_id: int,
        city: str,
    ) -> LaneSegment:
        """
        Retrieve one lane segment.

        Parameters
        ----------
        lane_id

        city
        """

        vector_map = self.get_vector_map(
            city,
        )

        lane = vector_map.get_lane(
            lane_id,
        )

        if lane is None:
            raise KeyError(
                f"Lane {lane_id} not found in {city}."
            )

        return lane

    ###########################################################################

    def has_lane(
        self,
        lane_id: int,
        city: str,
    ) -> bool:
        """
        Check if a lane exists.
        """

        return (
            self.get_lane_segment(
                lane_id,
                city,
            )
            is not None
        )

    ###########################################################################

    def lane_centerline(
        self,
        lane_id: int,
        city: str,
    ) -> np.ndarray:
        """
        Return lane centerline.
        """

        lane = self.get_lane_segment(
            lane_id,
            city,
        )

        return lane.centerline

    ###########################################################################
    # Spatial Queries
    ###########################################################################

    def get_lane_ids_in_xy_bbox(
        self,
        x: float,
        y: float,
        city: str,
        query_search_range_manhattan: float = 50.0,
    ) -> list[int]:
        """
        Return nearby lane IDs.

        This function intentionally mirrors the public API of the
        deprecated ArgoverseMap.

        Parameters
        ----------
        x
            Global x coordinate.

        y
            Global y coordinate.

        city
            City name.

        query_search_range_manhattan
            Search radius (meters).

        Returns
        -------
        list[int]
        """

        vector_map = self.get_vector_map(
            city,
        )

        return vector_map.get_lane_ids_in_xy_bbox(
            x=x,
            y=y,
            query_search_range_manhattan=query_search_range_manhattan,
        )

    ###########################################################################

    def get_nearest_lane(
        self,
        x: float,
        y: float,
        city: str,
        max_distance: float | None = None,
    ) -> LaneSegment | None:
        """
        Return the nearest lane segment.

        Parameters
        ----------
        x, y
            Query point.

        city
            City name.

        max_distance
            Optional maximum allowed distance.

        Returns
        -------
        LaneSegment | None
        """

        vector_map = self.get_vector_map(
            city,
        )

        if not vector_map.indexed:
            return None

        candidate_ids = vector_map.get_lane_ids_in_xy_bbox(
            x=x,
            y=y,
            query_search_range_manhattan=(
                max_distance
                if max_distance is not None
                else 100.0
            ),
        )

        if not candidate_ids:
            return None

        query = np.asarray(
            [x, y],
            dtype=np.float32,
        )

        best_lane: LaneSegment | None = None

        best_distance = float("inf")

        for lane_id in candidate_ids:

            lane = vector_map.get_lane(
                lane_id,
            )

            if lane is None:
                continue

            distance = np.linalg.norm(
                lane.centroid - query,
            )

            if distance < best_distance:

                best_distance = distance

                best_lane = lane

        if (
            max_distance is not None
            and best_distance > max_distance
        ):
            return None

        return best_lane

    ###########################################################################

    def get_lane_centerlines(
        self,
        city: str,
    ) -> list[np.ndarray]:
        """
        Return all lane centerlines for one city.
        """

        vector_map = self.get_vector_map(
            city,
        )

        return [

            lane.centerline

            for lane in vector_map

        ]

    ###########################################################################

    def get_lane_segments(
        self,
        city: str,
    ) -> list[LaneSegment]:
        """
        Return all lane segments.
        """

        vector_map = self.get_vector_map(
            city,
        )

        return list(
            vector_map,
        )

    ###########################################################################

    def num_lanes(
        self,
        city: str,
    ) -> int:
        """
        Number of lane segments in one city.
        """

        return self.get_vector_map(
            city,
        ).num_lanes

    ###########################################################################

    def summary(
        self,
    ) -> dict[str, object]:
        """
        Return loader summary.
        """

        return {

            "map_root": str(
                self.map_root,
            ),

            "cities": self.cities,

            "num_cities": len(
                self.cities,
            ),

            "num_lanes": {

                city: self.num_lanes(
                    city,
                )

                for city in self.cities

            },

        }

    ###########################################################################
    # Validation
    ###########################################################################

    def validate(
        self,
    ) -> None:
        """
        Validate every loaded city map.

        Raises
        ------
        RuntimeError
            If no maps have been loaded.

        ValueError
            If a VectorMap fails validation.
        """

        if not self._maps:

            raise RuntimeError(
                "No maps have been loaded."
            )

        for city, vector_map in self._maps.items():

            try:

                vector_map.validate()

            except Exception as exc:

                raise ValueError(
                    f"Validation failed for city "
                    f"{city}."
                ) from exc

    ###########################################################################
    # Reload
    ###########################################################################

    def reload(
        self,
    ) -> None:
        """
        Reload every map from disk.
        """

        self._maps.clear()

        self.city_lane_centerlines_dict.clear()

        self._nodes.clear()

        self.load_all_maps()

    ###########################################################################
    # Cache Management
    ###########################################################################

    def clear(
        self,
    ) -> None:
        """
        Remove every loaded map from memory.
        """

        self._maps.clear()

        self.city_lane_centerlines_dict.clear()

        self._nodes.clear()

    ###########################################################################

    def is_loaded(
        self,
    ) -> bool:
        """
        Returns
        -------
        bool
            True if at least one city has been loaded.
        """

        return len(self._maps) > 0

    ###########################################################################
    # Statistics
    ###########################################################################

    @property
    def num_cities(
        self,
    ) -> int:
        """
        Number of loaded cities.
        """

        return len(
            self._maps,
        )

    @property
    def total_num_lanes(
        self,
    ) -> int:
        """
        Total number of lane segments across all cities.
        """

        return sum(

            vector_map.num_lanes

            for vector_map in self._maps.values()

        )

    ###########################################################################

    def city_summary(
        self,
        city: str,
    ) -> dict[str, object]:
        """
        Summary for one city.
        """

        vector_map = self.get_vector_map(
            city,
        )

        return {

            "city": city.upper(),

            "num_lanes": vector_map.num_lanes,

            "indexed": vector_map.indexed,

            "map_path": (
                str(vector_map.map_path)
                if vector_map.map_path is not None
                else None
            ),

        }

    ###########################################################################

    def all_summaries(
        self,
    ) -> list[dict[str, object]]:
        """
        Summary of every loaded city.
        """

        return [

            self.city_summary(city)

            for city in self.cities

        ]

    ###########################################################################
    # Export
    ###########################################################################

    def to_dict(
        self,
    ) -> dict[str, object]:
        """
        Serialize loader metadata.
        """

        return {

            "map_root": str(
                self.map_root,
            ),

            "cities": self.cities,

            "num_cities": self.num_cities,

            "total_num_lanes": self.total_num_lanes,

        }

    ###########################################################################
    # Debug / Inspection Utilities
    ###########################################################################

    def list_lane_ids(
        self,
        city: str,
    ) -> list[int]:
        """
        Return all lane IDs for a city.
        """

        return self.get_vector_map(
            city,
        ).lane_ids

    ###########################################################################

    def lane_exists(
        self,
        lane_id: int,
        city: str,
    ) -> bool:
        """
        Check whether a lane exists.
        """

        return (
            lane_id
            in self.get_vector_map(city)
        )

    ###########################################################################

    def inspect_lane(
        self,
        lane_id: int,
        city: str,
    ) -> dict[str, object]:
        """
        Return detailed information for one lane.
        """

        lane = self.get_lane_segment(
            lane_id,
            city,
        )

        return {

            "lane_id": lane.lane_id,

            "num_points": lane.num_points,

            "length": lane.length,

            "centroid": lane.centroid,

            "start": lane.start,

            "end": lane.end,

            "turn_direction": lane.turn_direction,

            "intersection": lane.is_intersection,

            "traffic_control": lane.has_traffic_control,

            "predecessors": lane.predecessors,

            "successors": lane.successors,

            "left_neighbor": lane.left_neighbor_id,

            "right_neighbor": lane.right_neighbor_id,

        }

    ###########################################################################

    def print_summary(
        self,
    ) -> None:
        """
        Print a readable summary.
        """

        print()

        print("=" * 72)

        print("Argoverse HD Maps")

        print("=" * 72)

        print(
            f"Map Root : {self.map_root}"
        )

        print(
            f"Cities   : {self.num_cities}"
        )

        print(
            f"Total Lanes : {self.total_num_lanes}"
        )

        print()

        for city in self.cities:

            vm = self.get_vector_map(
                city,
            )

            print(

                f"{city:>6}"

                f" | lanes={vm.num_lanes:6d}"

                f" | indexed={vm.indexed}"

            )

        print("=" * 72)

    ###########################################################################

    def check_topology(
        self,
        city: str,
    ) -> bool:
        """
        Verify predecessor/successor references.

        Returns
        -------
        bool
            True if topology is consistent.
        """

        vector_map = self.get_vector_map(
            city,
        )

        for lane in vector_map:

            ###############################################################
            # predecessors
            ###############################################################

            for predecessor in lane.predecessors:

                if predecessor not in vector_map:

                    return False

            ###############################################################
            # successors
            ###############################################################

            for successor in lane.successors:

                if successor not in vector_map:

                    return False

            ###############################################################
            # neighbors
            ###############################################################

            if (

                lane.left_neighbor_id is not None

                and

                lane.left_neighbor_id
                not in vector_map

            ):

                return False

            if (

                lane.right_neighbor_id is not None

                and

                lane.right_neighbor_id
                not in vector_map

            ):

                return False

        return True

    ###########################################################################

    def statistics(
        self,
    ) -> dict[str, object]:
        """
        Dataset-wide statistics.
        """

        return {

            "cities": self.num_cities,

            "total_lanes": self.total_num_lanes,

            "average_lanes_per_city": (

                self.total_num_lanes
                / max(1, self.num_cities)

            ),

            "city_names": self.cities,

        }

    ###########################################################################
    # Container Interface
    ###########################################################################

    def __len__(
        self,
    ) -> int:
        """
        Number of loaded city maps.
        """

        return self.num_cities

    def __contains__(
        self,
        city: object,
    ) -> bool:
        """
        Membership test.

        Example
        -------
        "MIA" in loader
        """

        if not isinstance(
            city,
            str,
        ):
            return False

        return city.upper() in self._maps

    def __getitem__(
        self,
        city: str,
    ) -> VectorMap:
        """
        Dictionary-style access.

        Example
        -------
        loader["MIA"]
        """

        return self.get_vector_map(
            city,
        )

    def __iter__(
        self,
    ) -> Iterator[VectorMap]:
        """
        Iterate over loaded VectorMaps.
        """

        return iter(
            self._maps.values(),
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (

            "MapLoader("

            f"cities={self.num_cities}, "

            f"total_lanes={self.total_num_lanes}, "

            f"map_root='{self.map_root}')"

        )


