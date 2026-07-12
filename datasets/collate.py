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
# Agent Features
###############################################################################


def _pad_sequence(
    sequence: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad one variable-length sequence.

    Parameters
    ----------
    sequence
        Shape (T, ...)

    target_length
        Desired temporal length.

    Returns
    -------
    padded
        Shape (target_length, ...)

    mask
        Shape (target_length,)
    """

    feature_shape = sequence.shape[1:]

    padded = np.zeros(
        (target_length, *feature_shape),
        dtype=sequence.dtype,
    )

    mask = np.zeros(
        target_length,
        dtype=bool,
    )

    length = min(
        len(sequence),
        target_length,
    )

    if length > 0:

        padded[:length] = sequence[:length]

        mask[:length] = True

    return padded, mask


def _collate_agents(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:

    max_agents = max(
        scene.num_agents
        for scene in batch
    )

    batch_size = len(batch)

    observed = np.zeros(
        (
            batch_size,
            max_agents,
            20,
            2,
        ),
        dtype=np.float32,
    )

    observed_mask = np.zeros(
        (
            batch_size,
            max_agents,
            20,
        ),
        dtype=bool,
    )

    future = np.zeros(
        (
            batch_size,
            max_agents,
            30,
            2,
        ),
        dtype=np.float32,
    )

    future_mask = np.zeros(
        (
            batch_size,
            max_agents,
            30,
        ),
        dtype=bool,
    )

    velocity = np.zeros_like(
        observed,
    )

    speed = np.zeros(
        (
            batch_size,
            max_agents,
            20,
        ),
        dtype=np.float32,
    )

    acceleration = np.zeros_like(
        observed,
    )

    heading = np.zeros(
        (
            batch_size,
            max_agents,
            20,
        ),
        dtype=np.float32,
    )

    agent_mask = np.zeros(
        (
            batch_size,
            max_agents,
        ),
        dtype=bool,
    )

    ###########################################################################

    for batch_idx, scene in enumerate(batch):

        for agent_idx, agent in enumerate(scene.agents):

            obs, obs_mask = _pad_sequence(
                agent["observed"],
                20,
            )

            fut, fut_mask = _pad_sequence(
                agent["future"],
                30,
            )

            vel, _ = _pad_sequence(
                agent["velocity"],
                20,
            )

            spd, _ = _pad_sequence(
                agent["speed"][:, None],
                20,
            )

            acc, _ = _pad_sequence(
                agent["acceleration"],
                20,
            )

            hdg, _ = _pad_sequence(
                agent["heading"][:, None],
                20,
            )

            observed[
                batch_idx,
                agent_idx,
            ] = obs

            observed_mask[
                batch_idx,
                agent_idx,
            ] = obs_mask

            future[
                batch_idx,
                agent_idx,
            ] = fut

            future_mask[
                batch_idx,
                agent_idx,
            ] = fut_mask

            velocity[
                batch_idx,
                agent_idx,
            ] = vel

            speed[
                batch_idx,
                agent_idx,
            ] = spd.squeeze(-1)

            acceleration[
                batch_idx,
                agent_idx,
            ] = acc

            heading[
                batch_idx,
                agent_idx,
            ] = hdg.squeeze(-1)

            agent_mask[
                batch_idx,
                agent_idx,
            ] = True

    ###########################################################################

    return {

        "observed": torch.from_numpy(
            observed,
        ),

        "observed_mask": torch.from_numpy(
            observed_mask,
        ),

        "future": torch.from_numpy(
            future,
        ),

        "future_mask": torch.from_numpy(
            future_mask,
        ),

        "velocity": torch.from_numpy(
            velocity,
        ),

        "speed": torch.from_numpy(
            speed,
        ),

        "acceleration": torch.from_numpy(
            acceleration,
        ),

        "heading": torch.from_numpy(
            heading,
        ),

        "agent_mask": torch.from_numpy(
            agent_mask,
        ),
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
