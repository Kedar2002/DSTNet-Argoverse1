"""
engine.utils

Common utilities shared across the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_to_device(
    data: Any,
    device: torch.device | str,
) -> Any:
    """
    Recursively move nested data structures to a device.

    Supports

    - Tensor
    - dict
    - list
    - tuple

    Parameters
    ----------
    data
        Arbitrarily nested structure.

    device
        Target device.

    Returns
    -------
    Same structure on target device.
    """

    device = torch.device(device)

    if isinstance(data, torch.Tensor):

        return data.to(
            device,
            non_blocking=True,
        )

    if isinstance(data, Mapping):

        return {
            key: move_to_device(
                value,
                device,
            )
            for key, value in data.items()
        }

    if isinstance(data, list):

        return [
            move_to_device(
                value,
                device,
            )
            for value in data
        ]

    if isinstance(data, tuple):

        return tuple(
            move_to_device(
                value,
                device,
            )
            for value in data
        )

    return data


def validate_batch(
    batch: Mapping[str, Any],
    required_keys: set[str],
) -> None:
    """
    Validate required batch keys.
    """

    missing = required_keys.difference(
        batch.keys()
    )

    if missing:

        raise KeyError(

            "Batch is missing required keys: "

            f"{sorted(missing)}"

        )
