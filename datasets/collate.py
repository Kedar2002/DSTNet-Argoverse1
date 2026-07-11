"""
datasets.collate

Batch collation for DSTNet.

Converts a list of SceneData objects into batched
PyTorch tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from datasets.scene_data import SceneData


###############################################################################
# Helper
###############################################################################


def _stack(
    arrays: list[np.ndarray],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Stack NumPy arrays into a PyTorch tensor.
    """

    return torch.as_tensor(
        np.stack(arrays, axis=0),
        dtype=dtype,
    )

###############################################################################
# Agent Features
###############################################################################


def _collate_agents(
    batch: list[SceneData],
) -> dict[str, torch.Tensor]:

    observed = []
    future = []
    velocity = []
    speed = []
    acceleration = []
    heading = []

    for scene in batch:

        target = scene.target_agent

        observed.append(
            target["observed"]
        )

        future.append(
            target["future"]
        )

        velocity.append(
            target["velocity"]
        )

        speed.append(
            target["speed"]
        )

        acceleration.append(
            target["acceleration"]
        )

        heading.append(
            target["heading"]
        )

    return {
        "observed": _stack(observed),
        "future": _stack(future),
        "velocity": _stack(velocity),
        "speed": _stack(speed),
        "acceleration": _stack(acceleration),
        "heading": _stack(heading),
    }

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
        "origin": _stack(
            [
                scene.origin
                for scene in batch
            ]
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
# Public API
###############################################################################


def collate_fn(
    batch: list[SceneData],
) -> dict[str, Any]:
    """
    Collate SceneData objects into a batch.

    Currently batches only the target-agent tensors.
    Graphs and lane tensors will be added once the
    custom map loader is implemented.
    """

    return {
        "metadata": _collate_metadata(batch),
        "agents": _collate_agents(batch),
        "scenes": batch,
    }
