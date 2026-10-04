"""
engine.utils

Common utilities shared across the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
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


def select_supervised_agents(
    prediction: Any,
    refined_prediction: Any,
    ground_truth: torch.Tensor,
    supervision_mask: torch.Tensor | None,
    *,
    validate_non_empty: bool = True,
) -> tuple[Any, Any, torch.Tensor]:
    """Select labelled agents and the forecast at the final history step.

    SceneData stores one 30-frame future beginning after the final observed
    frame. Earlier model outputs have no matching target in that tensor, so
    comparing them with the same future would misalign the training loss.
    """

    if ground_truth.ndim != 4:
        raise ValueError("ground_truth must have shape (B,N,T,2).")

    batch_agents = ground_truth.shape[:2]
    if supervision_mask is None:
        supervision_mask = torch.ones(
            batch_agents,
            dtype=torch.bool,
            device=ground_truth.device,
        )
    else:
        if supervision_mask.shape != batch_agents:
            raise ValueError("supervision_mask must have shape (B,N).")
        supervision_mask = supervision_mask.to(
            device=ground_truth.device,
            dtype=torch.bool,
        )

    if validate_non_empty and not bool(supervision_mask.any()):
        raise ValueError("No agents with complete future labels were selected.")

    def select_agents(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if value.shape[:2] != batch_agents:
            raise ValueError(
                "Prediction batch/agent dimensions must match ground_truth."
            )
        selected = value[supervision_mask].unsqueeze(0)
        if selected.ndim >= 4 and selected.shape[2] > 1:
            selected = selected[:, :, -1:]
        return selected

    prediction = replace(
        prediction,
        trajectories=select_agents(prediction.trajectories),
        probabilities=select_agents(prediction.probabilities),
    )

    if refined_prediction is not None:
        refined_prediction = replace(
            refined_prediction,
            trajectories=select_agents(refined_prediction.trajectories),
            probabilities=select_agents(refined_prediction.probabilities),
            refinement_scores=select_agents(
                refined_prediction.refinement_scores
            ),
            offsets=select_agents(refined_prediction.offsets),
            trajectory_history=select_agents(
                refined_prediction.trajectory_history
            ),
            refinement_score_history=select_agents(
                refined_prediction.refinement_score_history
            ),
        )

    return (
        prediction,
        refined_prediction,
        ground_truth[supervision_mask].unsqueeze(0),
    )
