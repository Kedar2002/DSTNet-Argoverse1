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

Evaluation uses the refined prediction when enabled and the coarse
prediction for backbone-only training. Metrics use the final observed
state and the valid target-agent labels.

Ground truth:

    (B,N,T,2)
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
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

    autocast_enabled:
        Use mixed precision for model inference on CUDA.

    autocast_dtype:
        Autocast dtype when mixed precision is enabled.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device | str = "cpu",
        autocast_enabled: bool = False,
        autocast_dtype: torch.dtype = torch.float16,
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
        self.autocast_enabled = bool(
            autocast_enabled
            and self.device.type == "cuda"
        )
        self.autocast_dtype = autocast_dtype

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

        total_scenes = 0

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

            amp_context = (
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=True,
                )
                if self.autocast_enabled
                else nullcontext()
            )

            with amp_context:
                coarse_prediction, refined_prediction = self.model(
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

            ###################################################################
            # Use the refined output when enabled; backbone-only training
            # evaluates the coarse prediction.
            ###################################################################

            forecast = (
                refined_prediction
                if refined_prediction is not None
                else coarse_prediction
            )

            metric_mask = batch.get(
                "target_agent_mask",
                batch.get("agent_mask"),
            )

            if (
                metric_mask is not None
                and "future_mask" in batch
            ):
                metric_mask = (
                    metric_mask
                    & batch["future_mask"]
                )

            batch_metrics = compute_metrics(
                forecast,
                batch[
                    "future_trajectories"
                ],
                agent_mask=metric_mask,
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

            batch_scenes = int(
                batch["future_trajectories"].shape[0]
            )

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
                    + metric_value * batch_scenes
                )

            total_scenes += batch_scenes

        #######################################################################
        # Empty dataloader
        #######################################################################

        if total_scenes == 0:

            raise RuntimeError(
                "Evaluation DataLoader produced zero batches."
            )

        #######################################################################
        # Average metrics
        #######################################################################

        metrics = {
            key: value / float(total_scenes)
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
