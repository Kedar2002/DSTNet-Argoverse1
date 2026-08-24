"""
engine.evaluator

Evaluation engine for DSTNet.

Responsibilities
----------------
- Forward inference
- Metric computation
- Benchmark integration
- Visualization hooks

Current DSTNet contract
-----------------------

Model input:

    agent_trajectories : (B,N,H,2)
    map_centerlines   : (B,M,P,2)
    positions          : (B,N,2)
    graph              : list[SceneGraph]
    agent_mask         : (B,N)
    map_mask          : (B,M)

Model output:

    coarse_prediction
    refined_prediction

Evaluation is performed on the refined prediction.

Ground truth:

    (B,N,T,2)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.utils import (
    move_to_device,
    validate_batch,
)

from evaluation.metrics import compute_metrics


###############################################################################
# Evaluator
###############################################################################


class Evaluator:
    """
    Evaluate a trained DSTNet model.

    Parameters
    ----------
    model:
        DSTNet model.

    dataloader:
        Evaluation DataLoader.

    device:
        Evaluation device.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> None:

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

        self.model.to(
            self.device
        )

        self.model.eval()

    ###########################################################################
    # Metric validation
    ###########################################################################

    @staticmethod
    def _convert_metric(
        name: str,
        value: Any,
    ) -> float:
        """
        Convert one metric to a finite Python float.

        ``compute_metrics`` may return either Python numeric values or
        scalar tensors.
        """

        if isinstance(
            value,
            torch.Tensor,
        ):

            if value.numel() != 1:

                raise ValueError(
                    f"Metric '{name}' must be scalar. "
                    f"Got tensor with shape "
                    f"{tuple(value.shape)}."
                )

            if not torch.isfinite(
                value
            ).all().item():

                raise RuntimeError(
                    f"Metric '{name}' is NaN or infinite."
                )

            value = value.detach().item()

        if not isinstance(
            value,
            (float, int),
        ):
            raise TypeError(
                f"Metric '{name}' must be numeric. "
                f"Got {type(value).__name__}."
            )

        value = float(value)

        if not torch.isfinite(
            torch.tensor(value)
        ).item():

            raise RuntimeError(
                f"Metric '{name}' is NaN or infinite."
            )

        return value

    ###########################################################################
    # Evaluation
    ###########################################################################

    @torch.no_grad()
    def evaluate(
        self,
    ) -> dict[str, float]:
        """
        Evaluate the model over the complete dataloader.

        Returns
        -------
        dict[str, float]
            Mean evaluation metrics over all batches.

        Raises
        ------
        RuntimeError
            If the dataloader contains no batches.
        """

        self.model.eval()

        running: dict[str, float] = {}

        num_batches = 0

        #######################################################################
        # Required batch fields
        #######################################################################

        required_keys = {
            "agent_trajectories",
            "map_centerlines",
            "positions",
            "graph",
            "future_trajectories",
        }

        #######################################################################
        # Evaluation loop
        #######################################################################

        for batch in self.dataloader:

            ###################################################################
            # Move tensors to device.
            #
            # SceneGraph objects remain structurally intact because
            # move_to_device() only recursively moves supported containers
            # and tensors.
            ###################################################################

            batch = move_to_device(
                batch,
                self.device,
            )

            ###################################################################
            # Validate batch
            ###################################################################

            if not isinstance(
                batch,
                Mapping,
            ):
                raise TypeError(
                    "Evaluation batch must be a mapping."
                )

            validate_batch(
                batch,
                required_keys,
            )

            ###################################################################
            # Model forward
            ###################################################################

            coarse_prediction, refined_prediction = (
                self.model(
                    agent_trajectories=batch[
                        "agent_trajectories"
                    ],
                    map_centerlines=batch[
                        "map_centerlines"
                    ],
                    positions=batch[
                        "positions"
                    ],
                    graph=batch[
                        "graph"
                    ],
                    agent_mask=batch.get(
                        "agent_mask"
                    ),
                    map_mask=batch.get(
                        "map_mask"
                    ),
                )
            )

            ###################################################################
            # Evaluate refined prediction
            #
            # Coarse prediction is intentionally not used here.
            # The final model output is the refined prediction.
            ###################################################################

            batch_metrics = compute_metrics(
                refined_prediction,
                batch[
                    "future_trajectories"
                ],
            )

            if not isinstance(
                batch_metrics,
                Mapping,
            ):
                raise TypeError(
                    "compute_metrics() must return a mapping."
                )

            ###################################################################
            # Convert and validate metrics
            ###################################################################

            for key, value in (
                batch_metrics.items()
            ):

                metric_value = (
                    self._convert_metric(
                        str(key),
                        value,
                    )
                )

                running[str(key)] = (
                    running.get(
                        str(key),
                        0.0,
                    )
                    + metric_value
                )

            num_batches += 1

        #######################################################################
        # Empty dataloader
        #######################################################################

        if num_batches == 0:

            raise RuntimeError(
                "Evaluation DataLoader produced zero batches."
            )

        #######################################################################
        # Average metrics
        #######################################################################

        metrics = {
            key: value / float(
                num_batches
            )
            for key, value in running.items()
        }

        #######################################################################
        # Final numerical validation
        #######################################################################

        for key, value in metrics.items():

            if not torch.isfinite(
                torch.tensor(value)
            ).item():

                raise RuntimeError(
                    f"Final evaluation metric "
                    f"'{key}' is non-finite."
                )

        return metrics

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Evaluator("
            f"device={self.device}, "
            f"batches={len(self.dataloader)})"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "Evaluator",
]
