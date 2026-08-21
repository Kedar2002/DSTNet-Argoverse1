"""
datasets.scene_graph_builder

Scene graph construction for DSTNet.

The DSTNet paper treats:

    • observed agent states
    • lane segments

as graph nodes.

This module constructs the geometric graph from the processed
scene representation.

Node space
----------

For N agents and H observed states:

    N × H agent-state nodes

For M lane segments:

    M map nodes

The graph therefore contains:

    agent-state nodes
    map nodes

and four relation sets:

    temporal
    spatial
    agent-map
    map-map

Important
---------
The DSTNet paper explicitly defines the node representation and
the relative edge features, but does not fully specify the exact
neighborhood construction algorithm.

Therefore this implementation uses configurable geometric
neighborhood radii for spatial, agent-map, and map-map connectivity.
Those radii are implementation parameters, not claims about an
explicit value specified by the paper.

All geometry is expected to already be expressed in the
target-agent local coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


###############################################################################
# Scene Graph
###############################################################################


@dataclass(slots=True)
class SceneGraph:
    """
    Geometric scene graph used by DSTNet.

    Agent-state nodes
    -----------------
    One node corresponds to one observed state of one agent.

    Map nodes
    ---------
    One node corresponds to one lane segment.

    Node indexing
    -------------
    Agent-state nodes occupy:

        [0, N_state)

    Map nodes occupy:

        [0, N_map)

    within their respective arrays.

    RelativeSpatioTemporalEmbeddingModule is responsible for
    converting map indices into the unified node-index space when
    constructing Er.edge_index.
    """

    ###########################################################################
    # Agent-State Nodes
    ###########################################################################

    state_ids: np.ndarray
    """
    Shape: (Ns,)

    Sequential graph-state identifier.
    """

    track_indices: np.ndarray
    """
    Shape: (Ns,)

    Integer trajectory identifier.

    Each unique string track_id from the processed scene is mapped
    to one integer index.

    dtype: int64
    """

    timesteps: np.ndarray
    """
    Shape: (Ns,)

    Observation timestep for each state.

    Expected range:

        0 ... H-1
    """

    timestamps: np.ndarray
    """
    Shape: (Ns,)

    Temporal value associated with each state.

    The original timestamp is preserved when available. The
    timestep index is used as a fallback.
    """

    state_positions: np.ndarray
    """
    Shape: (Ns, 2)

    Agent-state positions in target-agent local coordinates.
    """

    state_headings: np.ndarray
    """
    Shape: (Ns,)

    Agent-state headings in target-agent local coordinates.
    """

    ###########################################################################
    # Map Nodes
    ###########################################################################

    map_ids: np.ndarray
    """
    Shape: (Nm,)

    Original lane identifiers.

    These are metadata identifiers and are NOT required to be
    contiguous.
    """

    map_positions: np.ndarray
    """
    Shape: (Nm, 2)

    Position of each lane-segment node in local coordinates.

    Following the paper's description of lane position/orientation
    from the lane centerline entrance and exit, the node position is
    represented by the midpoint of the first and last centerline
    points.
    """

    map_headings: np.ndarray
    """
    Shape: (Nm,)

    Orientation of each lane segment derived from its entrance-to-exit
    direction.
    """

    ###########################################################################
    # Edge Sets
    ###########################################################################

    temporal_edges: np.ndarray
    """
    Shape: (2, Et)

    Directed temporal edges:

        state(t) -> state(t+1)

    for the same track.
    """

    spatial_edges: np.ndarray
    """
    Shape: (2, Es)

    Directed agent-state interaction edges.

    Edges are restricted to states occurring at the same timestep
    and within spatial_radius.
    """

    agent_map_edges: np.ndarray
    """
    Shape: (2, Eam)

    Directed:

        agent-state -> map-node
    """

    map_map_edges: np.ndarray
    """
    Shape: (2, Emm)

    Directed map-node interaction edges within map_radius.
    """

    ###########################################################################
    # Connectivity Parameters
    ###########################################################################

    spatial_radius: float

    map_radius: float

    ###########################################################################
    # Basic Properties
    ###########################################################################

    @property
    def num_agent_states(self) -> int:
        return int(self.state_ids.shape[0])

    @property
    def num_map_nodes(self) -> int:
        return int(self.map_ids.shape[0])

    @property
    def num_temporal_edges(self) -> int:
        return int(self.temporal_edges.shape[1])

    @property
    def num_spatial_edges(self) -> int:
        return int(self.spatial_edges.shape[1])

    @property
    def num_agent_map_edges(self) -> int:
        return int(self.agent_map_edges.shape[1])

    @property
    def num_map_map_edges(self) -> int:
        return int(self.map_map_edges.shape[1])

    @property
    def num_edges(self) -> int:
        return (
            self.num_temporal_edges
            + self.num_spatial_edges
            + self.num_agent_map_edges
            + self.num_map_map_edges
        )

    ###########################################################################
    # Validation
    ###########################################################################

    def validate(self) -> None:
        """
        Validate graph structure, dimensions, dtypes and indices.
        """

        num_states = self.num_agent_states
        num_maps = self.num_map_nodes

        #######################################################################
        # Agent-state node arrays
        #######################################################################

        if self.state_ids.ndim != 1:
            raise ValueError(
                "state_ids must have shape (Ns,)."
            )

        if self.track_indices.shape != (num_states,):
            raise ValueError(
                "track_indices must have shape (Ns,)."
            )

        if self.timesteps.shape != (num_states,):
            raise ValueError(
                "timesteps must have shape (Ns,)."
            )

        if self.timestamps.shape != (num_states,):
            raise ValueError(
                "timestamps must have shape (Ns,)."
            )

        if self.state_positions.shape != (
            num_states,
            2,
        ):
            raise ValueError(
                "state_positions must have shape (Ns,2)."
            )

        if self.state_headings.shape != (num_states,):
            raise ValueError(
                "state_headings must have shape (Ns,)."
            )

        #######################################################################
        # Map node arrays
        #######################################################################

        if self.map_ids.ndim != 1:
            raise ValueError(
                "map_ids must have shape (Nm,)."
            )

        if self.map_positions.shape != (
            num_maps,
            2,
        ):
            raise ValueError(
                "map_positions must have shape (Nm,2)."
            )

        if self.map_headings.shape != (num_maps,):
            raise ValueError(
                "map_headings must have shape (Nm,)."
            )

        #######################################################################
        # Dtypes
        #######################################################################

        integer_arrays = (
            ("state_ids", self.state_ids),
            ("track_indices", self.track_indices),
            ("timesteps", self.timesteps),
        )

        for name, array in integer_arrays:

            if not np.issubdtype(
                array.dtype,
                np.integer,
            ):
                raise TypeError(
                    f"{name} must contain integers."
                )

        if not np.issubdtype(
            self.map_ids.dtype,
            np.integer,
        ):
            raise TypeError(
                "map_ids must contain integers."
            )

        #######################################################################
        # Edge shapes and bounds
        #######################################################################

        self._validate_edge_index(
            self.temporal_edges,
            upper_bound=num_states,
            name="temporal_edges",
        )

        self._validate_edge_index(
            self.spatial_edges,
            upper_bound=num_states,
            name="spatial_edges",
        )

        if self.agent_map_edges.ndim != 2:
            raise ValueError(
                "agent_map_edges must have shape (2,Eam)."
            )

        if self.agent_map_edges.shape[0] != 2:
            raise ValueError(
                "agent_map_edges must have shape (2,Eam)."
            )

        if self.agent_map_edges.size:

            state_indices = self.agent_map_edges[0]
            map_indices = self.agent_map_edges[1]

            if state_indices.min() < 0:
                raise ValueError(
                    "agent_map_edges contains a negative state index."
                )

            if state_indices.max() >= num_states:
                raise ValueError(
                    "agent_map_edges contains an invalid state index."
                )

            if map_indices.min() < 0:
                raise ValueError(
                    "agent_map_edges contains a negative map index."
                )

            if map_indices.max() >= num_maps:
                raise ValueError(
                    "agent_map_edges contains an invalid map index."
                )

        self._validate_edge_index(
            self.map_map_edges,
            upper_bound=num_maps,
            name="map_map_edges",
        )

        #######################################################################
        # Numeric validity
        #######################################################################

        numeric_arrays = (
            ("state_positions", self.state_positions),
            ("state_headings", self.state_headings),
            ("map_positions", self.map_positions),
            ("map_headings", self.map_headings),
            ("timestamps", self.timestamps),
        )

        for name, array in numeric_arrays:

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        #######################################################################
        # State IDs must be unique
        #######################################################################

        if len(np.unique(self.state_ids)) != num_states:
            raise ValueError(
                "state_ids must be unique."
            )

        #######################################################################
        # Track indices must be non-negative
        #######################################################################

        if num_states and self.track_indices.min() < 0:
            raise ValueError(
                "track_indices must be non-negative."
            )

        #######################################################################
        # Timesteps must be non-negative
        #######################################################################

        if num_states and self.timesteps.min() < 0:
            raise ValueError(
                "timesteps must be non-negative."
            )

    @staticmethod
    def _validate_edge_index(
        edges: np.ndarray,
        upper_bound: int,
        name: str,
    ) -> None:
        """
        Validate an edge-index array with shape (2,E).
        """

        if edges.ndim != 2:
            raise ValueError(
                f"{name} must have shape (2,E)."
            )

        if edges.shape[0] != 2:
            raise ValueError(
                f"{name} must have shape (2,E)."
            )

        if not edges.size:
            return

        if edges.min() < 0:
            raise ValueError(
                f"{name} contains negative indices."
            )

        if edges.max() >= upper_bound:
            raise ValueError(
                f"{name} contains invalid indices."
            )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(self) -> str:

        return (
            "SceneGraph("
            f"agent_states={self.num_agent_states}, "
            f"map_nodes={self.num_map_nodes}, "
            f"temporal_edges={self.num_temporal_edges}, "
            f"spatial_edges={self.num_spatial_edges}, "
            f"agent_map_edges={self.num_agent_map_edges}, "
            f"map_map_edges={self.num_map_map_edges})"
        )


###############################################################################
# Scene Graph Builder
###############################################################################


class SceneGraphBuilder:
    """
    Construct a SceneGraph from processed agents and map elements.

    The processed representation is expected to contain only the
    observation history for graph construction.

    Required agent fields
    ----------------------
    track_id
    observed
    headings

    Optional agent field
    --------------------
    timestamps

    Required map fields
    -------------------
    lane_id
    centerline

    The graph construction itself is geometric and does not produce
    learned features.
    """

    def __init__(
        self,
        spatial_radius: float,
        map_radius: float,
    ) -> None:

        if spatial_radius <= 0.0:
            raise ValueError(
                "spatial_radius must be positive."
            )

        if map_radius <= 0.0:
            raise ValueError(
                "map_radius must be positive."
            )

        self.spatial_radius = float(
            spatial_radius
        )

        self.map_radius = float(
            map_radius
        )

    ###########################################################################
    # Public API
    ###########################################################################

    def build(
        self,
        agents: Sequence[dict[str, Any]],
        maps: Sequence[dict[str, Any]],
    ) -> SceneGraph:
        """
        Build a SceneGraph from processed agents and maps.
        """

        (
            state_ids,
            track_indices,
            timesteps,
            timestamps,
            state_positions,
            state_headings,
        ) = self._collect_agent_states(
            agents
        )

        (
            map_ids,
            map_positions,
            map_headings,
        ) = self._collect_map_nodes(
            maps
        )

        temporal_edges = self._build_temporal_edges(
            track_indices=track_indices,
            timesteps=timesteps,
        )

        spatial_edges = self._build_spatial_edges(
            state_positions=state_positions,
            timesteps=timesteps,
        )

        agent_map_edges = self._build_agent_map_edges(
            state_positions=state_positions,
            map_positions=map_positions,
        )

        map_map_edges = self._build_map_map_edges(
            map_positions=map_positions,
        )

        graph = SceneGraph(
            state_ids=state_ids,
            track_indices=track_indices,
            timesteps=timesteps,
            timestamps=timestamps,
            state_positions=state_positions,
            state_headings=state_headings,
            map_ids=map_ids,
            map_positions=map_positions,
            map_headings=map_headings,
            temporal_edges=temporal_edges,
            spatial_edges=spatial_edges,
            agent_map_edges=agent_map_edges,
            map_map_edges=map_map_edges,
            spatial_radius=self.spatial_radius,
            map_radius=self.map_radius,
        )

        graph.validate()

        return graph

    ###########################################################################
    # Agent-State Nodes
    ###########################################################################

    def _collect_agent_states(
        self,
        agents: Sequence[dict[str, Any]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Flatten observed trajectories into graph nodes.

        Ordering
        --------
        Agent 0:
            t=0 ... H-1

        Agent 1:
            t=0 ... H-1

        ...

        This ordering is deliberately aligned with the natural
        flattening:

            (N,H,D) -> (N*H,D)

        used by downstream model components.
        """

        state_ids: list[int] = []
        track_indices: list[int] = []
        timesteps: list[int] = []
        timestamps: list[float] = []
        positions: list[np.ndarray] = []
        headings: list[float] = []

        track_lookup: dict[str, int] = {}

        next_state_id = 0

        for agent in agents:

            if "track_id" not in agent:
                raise KeyError(
                    "Processed agent is missing 'track_id'."
                )

            if "observed" not in agent:
                raise KeyError(
                    "Processed agent is missing 'observed'."
                )

            if "headings" not in agent:
                raise KeyError(
                    "Processed agent is missing 'headings'."
                )

            track_id = str(
                agent["track_id"]
            )

            observed = np.asarray(
                agent["observed"],
                dtype=np.float32,
            )

            agent_headings = np.asarray(
                agent["headings"],
                dtype=np.float32,
            )

            if observed.ndim != 2:
                raise ValueError(
                    "agent['observed'] must have shape (H,2)."
                )

            if observed.shape[1] != 2:
                raise ValueError(
                    "agent['observed'] must have shape (H,2)."
                )

            if agent_headings.ndim != 1:
                raise ValueError(
                    "agent['headings'] must have shape (H,)."
                )

            if len(agent_headings) != len(observed):
                raise ValueError(
                    "Agent observed trajectory and headings "
                    "must have equal length."
                )

            if len(observed) == 0:
                continue

            ###################################################################
            # Integer track identifier
            ###################################################################

            if track_id not in track_lookup:

                track_lookup[track_id] = len(
                    track_lookup
                )

            track_index = track_lookup[
                track_id
            ]

            ###################################################################
            # Optional timestamps
            ###################################################################

            raw_timestamps = agent.get(
                "timestamps",
                None,
            )

            if raw_timestamps is None:

                agent_timestamps = np.arange(
                    len(observed),
                    dtype=np.float32,
                )

            else:

                agent_timestamps = np.asarray(
                    raw_timestamps,
                    dtype=np.float64,
                )

                if agent_timestamps.ndim != 1:
                    raise ValueError(
                        "agent['timestamps'] must have shape (H,)."
                    )

                if len(agent_timestamps) != len(observed):
                    raise ValueError(
                        "Agent timestamps and observed trajectory "
                        "must have equal length."
                    )

            ###################################################################
            # Append one node per observed state
            ###################################################################

            for timestep in range(
                len(observed)
            ):

                state_ids.append(
                    next_state_id
                )

                track_indices.append(
                    track_index
                )

                timesteps.append(
                    timestep
                )

                timestamps.append(
                    float(
                        agent_timestamps[timestep]
                    )
                )

                positions.append(
                    observed[timestep]
                )

                headings.append(
                    float(
                        agent_headings[timestep]
                    )
                )

                next_state_id += 1

        #######################################################################
        # Convert
        #######################################################################

        if state_ids:

            state_positions = np.asarray(
                positions,
                dtype=np.float32,
            )

            state_headings = np.asarray(
                headings,
                dtype=np.float32,
            )

        else:

            state_positions = np.empty(
                (0, 2),
                dtype=np.float32,
            )

            state_headings = np.empty(
                (0,),
                dtype=np.float32,
            )

        return (
            np.asarray(
                state_ids,
                dtype=np.int64,
            ),
            np.asarray(
                track_indices,
                dtype=np.int64,
            ),
            np.asarray(
                timesteps,
                dtype=np.int64,
            ),
            np.asarray(
                timestamps,
                dtype=np.float32,
            ),
            state_positions,
            state_headings,
        )

    ###########################################################################
    # Map Nodes
    ###########################################################################

    def _collect_map_nodes(
        self,
        maps: Sequence[dict[str, Any]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Construct one graph node for every lane segment.

        The paper describes the lane-segment position and orientation
        using the entrance and exit of the lane centerline.

        Therefore:

            position =
                (centerline[0] + centerline[-1]) / 2

            heading =
                atan2(
                    y_exit - y_entry,
                    x_exit - x_entry
                )
        """

        map_ids: list[int] = []
        positions: list[np.ndarray] = []
        headings: list[float] = []

        for map_index, map_element in enumerate(
            maps
        ):

            if "lane_id" not in map_element:
                raise KeyError(
                    "Processed map element is missing 'lane_id'."
                )

            if "centerline" not in map_element:
                raise KeyError(
                    "Processed map element is missing 'centerline'."
                )

            centerline = np.asarray(
                map_element["centerline"],
                dtype=np.float32,
            )

            if centerline.ndim != 2:
                raise ValueError(
                    "map centerline must have shape (P,2)."
                )

            if centerline.shape[1] != 2:
                raise ValueError(
                    "map centerline must have shape (P,2)."
                )

            if len(centerline) == 0:
                raise ValueError(
                    "map centerline cannot be empty."
                )

            lane_id = int(
                map_element["lane_id"]
            )

            ###################################################################
            # Lane position and orientation
            ###################################################################

            if len(centerline) == 1:

                position = centerline[0]

                direction = np.array(
                    [1.0, 0.0],
                    dtype=np.float32,
                )

            else:

                entrance = centerline[0]

                exit_point = centerline[-1]

                position = (
                    0.5
                    * (
                        entrance
                        + exit_point
                    )
                )

                direction = (
                    exit_point
                    - entrance
                )

                norm = np.linalg.norm(
                    direction
                )

                if norm <= 1e-8:

                    direction = np.array(
                        [1.0, 0.0],
                        dtype=np.float32,
                    )

                else:

                    direction = (
                        direction / norm
                    )

            lane_heading = float(
                np.arctan2(
                    direction[1],
                    direction[0],
                )
            )

            map_ids.append(
                lane_id
            )

            positions.append(
                np.asarray(
                    position,
                    dtype=np.float32,
                )
            )

            headings.append(
                lane_heading
            )

        if positions:

            map_positions = np.asarray(
                positions,
                dtype=np.float32,
            )

            map_headings = np.asarray(
                headings,
                dtype=np.float32,
            )

        else:

            map_positions = np.empty(
                (0, 2),
                dtype=np.float32,
            )

            map_headings = np.empty(
                (0,),
                dtype=np.float32,
            )

        return (
            np.asarray(
                map_ids,
                dtype=np.int64,
            ),
            map_positions,
            map_headings,
        )

    ###########################################################################
    # Temporal Edges
    ###########################################################################

    def _build_temporal_edges(
        self,
        track_indices: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """
        Connect consecutive observed states of the same agent.

        For each trajectory:

            t -> t+1

        is added when both states exist.
        """

        if len(track_indices) == 0:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        state_lookup: dict[
            tuple[int, int],
            int,
        ] = {}

        for state_index in range(
            len(track_indices)
        ):

            key = (
                int(track_indices[state_index]),
                int(timesteps[state_index]),
            )

            if key in state_lookup:
                raise ValueError(
                    "Duplicate agent state for "
                    f"track_index={key[0]}, "
                    f"timestep={key[1]}."
                )

            state_lookup[key] = state_index

        edges: list[
            tuple[int, int]
        ] = []

        for (
            track_index,
            timestep,
        ), source_index in sorted(
            state_lookup.items()
        ):

            target_index = state_lookup.get(
                (
                    track_index,
                    timestep + 1,
                )
            )

            if target_index is None:
                continue

            edges.append(
                (
                    source_index,
                    target_index,
                )
            )

        if not edges:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        return np.asarray(
            edges,
            dtype=np.int64,
        ).T

    ###########################################################################
    # Spatial Agent-State Edges
    ###########################################################################

    def _build_spatial_edges(
        self,
        state_positions: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        """
        Construct same-timestep spatial interaction edges.

        Only states with identical timesteps are compared.

        Self-edges are excluded.

        Both directions are retained:

            i -> j
            j -> i
        """

        num_states = len(
            state_positions
        )

        if num_states == 0:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        edge_list: list[
            tuple[int, int]
        ] = []

        #######################################################################
        # Group states by timestep
        #######################################################################

        timestep_groups: dict[
            int,
            list[int],
        ] = {}

        for state_index, timestep in enumerate(
            timesteps
        ):

            timestep_groups.setdefault(
                int(timestep),
                [],
            ).append(
                state_index
            )

        #######################################################################
        # Radius search within each timestep
        #######################################################################

        for timestep in sorted(
            timestep_groups
        ):

            indices = timestep_groups[
                timestep
            ]

            if len(indices) <= 1:
                continue

            positions = state_positions[
                indices
            ]

            tree = cKDTree(
                positions
            )

            for local_source, position in enumerate(
                positions
            ):

                neighbors = tree.query_ball_point(
                    position,
                    r=self.spatial_radius,
                    p=2.0,
                )

                source_index = indices[
                    local_source
                ]

                for local_target in neighbors:

                    target_index = indices[
                        local_target
                    ]

                    if (
                        source_index
                        == target_index
                    ):
                        continue

                    edge_list.append(
                        (
                            source_index,
                            target_index,
                        )
                    )

        if not edge_list:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T

    ###########################################################################
    # Agent → Map Edges
    ###########################################################################

    def _build_agent_map_edges(
        self,
        state_positions: np.ndarray,
        map_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Connect agent states to nearby map nodes.

        The graph direction is:

            agent-state -> map-node
        """

        if (
            len(state_positions) == 0
            or len(map_positions) == 0
        ):

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        map_tree = cKDTree(
            map_positions
        )

        edge_list: list[
            tuple[int, int]
        ] = []

        for state_index, position in enumerate(
            state_positions
        ):

            nearby_maps = map_tree.query_ball_point(
                position,
                r=self.map_radius,
                p=2.0,
            )

            for map_index in nearby_maps:

                edge_list.append(
                    (
                        state_index,
                        int(map_index),
                    )
                )

        if not edge_list:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T

    ###########################################################################
    # Map → Map Edges
    ###########################################################################

    def _build_map_map_edges(
        self,
        map_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Construct map-node spatial interaction edges.

        Self-edges are excluded.

        Both directions are retained.
        """

        if len(map_positions) == 0:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        tree = cKDTree(
            map_positions
        )

        edge_list: list[
            tuple[int, int]
        ] = []

        for source_index, position in enumerate(
            map_positions
        ):

            neighbors = tree.query_ball_point(
                position,
                r=self.map_radius,
                p=2.0,
            )

            for target_index in neighbors:

                if source_index == target_index:
                    continue

                edge_list.append(
                    (
                        source_index,
                        int(target_index),
                    )
                )

        if not edge_list:

            return np.empty(
                (2, 0),
                dtype=np.int64,
            )

        return np.asarray(
            edge_list,
            dtype=np.int64,
        ).T


###############################################################################
# Public API
###############################################################################

__all__ = [
    "SceneGraph",
    "SceneGraphBuilder",
]
