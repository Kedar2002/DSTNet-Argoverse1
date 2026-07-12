"""
datasets.collate

Batch collation for DSTNet.

Converts SceneData objects into batched tensors.

Unlike the initial implementation, this version batches
ALL agents and ALL lanes, creating padding masks that
the attention modules can consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from datasets.scene_data import SceneData


###############################################################################
# Helpers
###############################################################################


def _to_tensor(
    array: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    return torch.as_tensor(
        array,
        dtype=dtype,
    )


def _pad_array(
    arrays: list[np.ndarray],
    pad_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of arrays.

    Returns
    -------
    tensor
        (B, max_items, ...)

    mask
        (B, max_items)
        True = valid
    """

    batch_size = len(arrays)

    max_items = max(
        array.shape[0]
        for array in arrays
    )

    padded = torch.zeros(
        (
            batch_size,
            max_items,
            *pad_shape,
        ),
        dtype=torch.float32,
    )

    mask = torch.zeros(
        (
            batch_size,
            max_items,
        ),
        dtype=torch.bool,
    )

    for i, array in enumerate(arrays):

        length = array.shape[0]

        padded[i, :length] = _to_tensor(array)

        mask[i, :length] = True

    return padded, mask


###############################################################################
# Metadata
###############################################################################


def _collate_metadata(
    batch: list[SceneData],
) -> dict[str, Any]:

    return {
        "sequence_id": [
            scene.sequence_id
            for scene in batch
        ],
        "city": [
            scene.city
            for scene in batch
        ],
        "origin": torch.as_tensor(
            np.stack(
                [
                    scene.origin
                    for scene in batch
                ]
            ),
            dtype=torch.float32,
        ),
        "heading": torch.tensor(
            [
                scene.heading
                for scene in batch
            ],
            dtype=torch.float32,
        ),
    }


###############################################################################
# Agents
###############################################################################


def _collate_agents(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:

    observed = [
        np.stack(
            [
                agent["observed"]
                for agent in scene.agents
            ]
        )
        for scene in batch
    ]

    future = [
        np.stack(
            [
                agent["future"]
                for agent in scene.agents
            ]
        )
        for scene in batch
    ]

    velocity = [
        np.stack(
            [
                agent["velocity"]
                for agent in scene.agents
            ]
        )
        for scene in batch
    ]

    heading = [
        np.stack(
            [
                agent["heading"]
                for agent in scene.agents
            ]
        )
        for scene in batch
    ]

    observed, mask = _pad_array(
        observed,
        observed[0].shape[1:],
    )

    future, _ = _pad_array(
        future,
        future[0].shape[1:],
    )

    velocity, _ = _pad_array(
        velocity,
        velocity[0].shape[1:],
    )

    heading, _ = _pad_array(
        heading,
        heading[0].shape[1:],
    )

    return {
        "observed": observed,
        "future": future,
        "velocity": velocity,
        "heading": heading,
        "mask": mask,
    }


###############################################################################
# Lanes
###############################################################################


def _collate_lanes(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:

    centerlines = [
        np.stack(
            [
                lane["centerline"]
                for lane in scene.lanes
            ]
        )
        for scene in batch
    ]

    directions = [
        np.stack(
            [
                lane["direction"]
                for lane in scene.lanes
            ]
        )
        for scene in batch
    ]

    centerlines, mask = _pad_array(
        centerlines,
        centerlines[0].shape[1:],
    )

    directions, _ = _pad_array(
        directions,
        directions[0].shape[1:],
    )

    return {
        "centerline": centerlines,
        "direction": directions,
        "mask": mask,
    }


###############################################################################
# Public API
###############################################################################


def collate_fn(
    batch: list[SceneData],
) -> dict[str, Any]:

    return {
        "metadata": _collate_metadata(batch),
        "agents": _collate_agents(batch),
        "lanes": _collate_lanes(batch),
        "graphs": [
            scene.graph
            for scene in batch
        ],
    }
