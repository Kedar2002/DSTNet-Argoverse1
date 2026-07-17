"""
datasets.collate

Batch collation for DSTNet.

Converts SceneData objects into batched tensors suitable for the
DSTNet model.

Output Batch
------------
{
    "agent_trajectories": (B,N,Tobs,2)
    "future_trajectories": (B,N,Tpred,2)
    "lane_centerlines": (B,L,P,2)
    "positions": (B,N,2)
    "headings": (B,N)
    "graph": list[GraphData]
    "agent_mask": (B,N)
    "lane_mask": (B,L)
    "metadata": dict
}
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from datasets.graph_builder import GraphData
from datasets.scene_data import SceneData

###############################################################################
# Constants
###############################################################################

OBSERVATION_STEPS = 20

PREDICTION_STEPS = 30

LANE_POINTS = 20

###############################################################################
# Tensor Helpers
###############################################################################


def _to_tensor(
    array: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    """

    return torch.as_tensor(
        array,
        dtype=dtype,
    )


###############################################################################
# Padding Helpers
###############################################################################


def _pad_sequence(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Pad one temporal sequence.

    Parameters
    ----------
    sequence
        Shape (T,...)

    target_length
        Desired temporal length.

    Returns
    -------
    ndarray
        Shape (target_length,...)
    """

    feature_shape = sequence.shape[1:]

    padded = np.zeros(
        (
            target_length,
            *feature_shape,
        ),
        dtype=np.float32,
    )

    length = min(
        len(sequence),
        target_length,
    )

    if length > 0:

        padded[:length] = sequence[:length]

    return padded


def _allocate_agent_arrays(
    batch_size: int,
    max_agents: int,
) -> dict[str, np.ndarray]:
    """
    Allocate all batched agent arrays.
    """

    return {

        "agent_trajectories": np.zeros(
            (
                batch_size,
                max_agents,
                OBSERVATION_STEPS,
                2,
            ),
            dtype=np.float32,
        ),

        "future_trajectories": np.zeros(
            (
                batch_size,
                max_agents,
                PREDICTION_STEPS,
                2,
            ),
            dtype=np.float32,
        ),

        "positions": np.zeros(
            (
                batch_size,
                max_agents,
                2,
            ),
            dtype=np.float32,
        ),

        "headings": np.zeros(
            (
                batch_size,
                max_agents,
            ),
            dtype=np.float32,
        ),

        "agent_mask": np.zeros(
            (
                batch_size,
                max_agents,
            ),
            dtype=bool,
        ),
    }


def _allocate_lane_arrays(
    batch_size: int,
    max_lanes: int,
) -> dict[str, np.ndarray]:
    """
    Allocate lane tensors.
    """

    return {

        "lane_centerlines": np.zeros(
            (
                batch_size,
                max_lanes,
                LANE_POINTS,
                2,
            ),
            dtype=np.float32,
        ),

        "lane_mask": np.zeros(
            (
                batch_size,
                max_lanes,
            ),
            dtype=bool,
        ),
    }


###############################################################################
# Metadata
###############################################################################


def _collate_metadata(
    batch: list[SceneData],
) -> dict[str, Any]:
    """
    Batch scene metadata.
    """

    return {

        "sequence_ids": [

            scene.sequence_id

            for scene in batch

        ],

        "cities": [

            scene.city

            for scene in batch

        ],

        "origins": torch.tensor(

            np.stack(

                [

                    scene.origin

                    for scene in batch

                ]

            ),

            dtype=torch.float32,

        ),

        "scene_headings": torch.tensor(

            [

                scene.heading

                for scene in batch

            ],

            dtype=torch.float32,

        ),

    }

###############################################################################
# Agent Collation
###############################################################################


def _collate_agents(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:
    """
    Collate agent features.

    Returns
    -------
    {
        "agent_trajectories": (B,N,Tobs,2)
        "future_trajectories": (B,N,Tpred,2)
        "positions": (B,N,2)
        "headings": (B,N)
        "agent_mask": (B,N)
    }
    """

    batch_size = len(batch)

    max_agents = max(
        scene.num_agents
        for scene in batch
    )

    arrays = _allocate_agent_arrays(
        batch_size=batch_size,
        max_agents=max_agents,
    )

    ###########################################################################
    # Populate arrays
    ###########################################################################

    for batch_index, scene in enumerate(batch):

        for agent_index, agent in enumerate(scene.agents):

            ###################################################################
            # Observed trajectory
            ###################################################################

            observed = _pad_sequence(
                agent["observed"],
                OBSERVATION_STEPS,
            )

            arrays["agent_trajectories"][
                batch_index,
                agent_index,
            ] = observed

            ###################################################################
            # Future trajectory
            ###################################################################

            future = _pad_sequence(
                agent["future"],
                PREDICTION_STEPS,
            )

            arrays["future_trajectories"][
                batch_index,
                agent_index,
            ] = future

            ###################################################################
            # Current position
            #
            # Last observed position in local coordinates.
            ###################################################################

            arrays["positions"][
                batch_index,
                agent_index,
            ] = observed[
                OBSERVATION_STEPS - 1
            ]

            ###################################################################
            # Current heading
            #
            # Last observed heading.
            ###################################################################

            headings = agent["heading"]

            if len(headings):

                arrays["headings"][
                    batch_index,
                    agent_index,
                ] = float(
                    headings[-1]
                )

            ###################################################################
            # Valid agent mask
            ###################################################################

            arrays["agent_mask"][
                batch_index,
                agent_index,
            ] = True

    ###########################################################################
    # Convert to tensors
    ###########################################################################

    return {

        "agent_trajectories": _to_tensor(
            arrays["agent_trajectories"],
        ),

        "future_trajectories": _to_tensor(
            arrays["future_trajectories"],
        ),

        "positions": _to_tensor(
            arrays["positions"],
        ),

        "headings": _to_tensor(
            arrays["headings"],
        ),

        "agent_mask": torch.as_tensor(
            arrays["agent_mask"],
            dtype=torch.bool,
        ),

    }

###############################################################################
# Lane Collation
###############################################################################


def _collate_lanes(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:
    """
    Collate lane centerlines.

    Returns
    -------
    {
        "lane_centerlines": (B,L,P,2)
        "lane_mask": (B,L)
    }
    """

    batch_size = len(batch)

    max_lanes = max(
        scene.num_lanes
        for scene in batch
    )

    arrays = _allocate_lane_arrays(
        batch_size=batch_size,
        max_lanes=max_lanes,
    )

    ###########################################################################
    # Populate lane tensors
    ###########################################################################

    for batch_index, scene in enumerate(batch):

        for lane_index, lane in enumerate(scene.lanes):

            centerline = _pad_sequence(
                lane["centerline"],
                LANE_POINTS,
            )

            arrays["lane_centerlines"][
                batch_index,
                lane_index,
            ] = centerline

            arrays["lane_mask"][
                batch_index,
                lane_index,
            ] = True

    ###########################################################################
    # Convert to tensors
    ###########################################################################

    return {

        "lane_centerlines": _to_tensor(
            arrays["lane_centerlines"],
        ),

        "lane_mask": torch.as_tensor(
            arrays["lane_mask"],
            dtype=torch.bool,
        ),

    }


###############################################################################
# Graph Collation
###############################################################################


def _collate_graphs(
    batch: list[SceneData],
) -> list[GraphData]:
    """
    Collect graph objects.

    GraphData instances contain variable-sized graph
    connectivity information, so they are kept as a list.
    """

    return [

        scene.graph

        for scene in batch

    ]


###############################################################################
# Validation
###############################################################################


def _validate_batch(
    batch: list[SceneData],
) -> None:
    """
    Validate input batch.
    """

    if len(batch) == 0:

        raise ValueError(
            "Received an empty batch."
        )

    for scene in batch:

        if not isinstance(
            scene,
            SceneData,
        ):

            raise TypeError(

                "collate_fn expects "

                "SceneData objects."

            )

###############################################################################
# Public API
###############################################################################


def collate_fn(
    batch: list[SceneData],
) -> dict[str, Any]:
    """
    Collate a batch of SceneData objects into the tensor format expected
    by DSTNet.

    Returned dictionary
    -------------------

    {
        "agent_trajectories": (B,N,Tobs,2)

        "future_trajectories": (B,N,Tpred,2)

        "lane_centerlines": (B,L,P,2)

        "positions": (B,N,2)

        "headings": (B,N)

        "graph": list[GraphData]

        "agent_mask": (B,N)

        "lane_mask": (B,L)

        "metadata": dict
    }
    """

    ###########################################################################
    # Validation
    ###########################################################################

    _validate_batch(
        batch,
    )

    ###########################################################################
    # Individual components
    ###########################################################################

    metadata = _collate_metadata(
        batch,
    )

    agents = _collate_agents(
        batch,
    )

    lanes = _collate_lanes(
        batch,
    )

    graphs = _collate_graphs(
        batch,
    )

    ###########################################################################
    # Assemble final batch
    ###########################################################################

    output = {

        #######################################################################
        # Model Inputs
        #######################################################################

        "agent_trajectories":
            agents["agent_trajectories"],

        "future_trajectories":
            agents["future_trajectories"],

        "lane_centerlines":
            lanes["lane_centerlines"],

        "positions":
            agents["positions"],

        "headings":
            agents["headings"],

        "graph":
            graphs,

        "agent_mask":
            agents["agent_mask"],

        "lane_mask":
            lanes["lane_mask"],

        #######################################################################
        # Metadata
        #######################################################################

        "metadata":
            metadata,
    }

    return output


###############################################################################
# Convenience Alias
###############################################################################

__all__ = [
    "collate_fn",
]

