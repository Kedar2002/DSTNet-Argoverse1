"""
engine.train_step

Single optimization step for DSTNet.

Responsibilities
----------------
- Move batch to device
- Forward pass
- Loss computation
- Backward pass
- Gradient clipping
- Optimizer update
- Scheduler update
- Metric extraction
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from engine.utils import (
    move_to_device,
    validate_batch,
)

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TrainStep:
    """
    Executes one optimization step.

    Parameters
    ----------
    model
        DSTNet model.

    criterion
        Total loss module.

    optimizer
        Optimizer.

    scheduler
        Learning-rate scheduler.

    gradient_clip
        Maximum gradient norm.

    device
        Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        *,
        gradient_clip: float | None = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)

    ###########################################################################
    # Training Step
    ###########################################################################

    def __call__(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """
        Execute one optimization step.

        Parameters
        ----------
        batch
            Mini-batch from DataLoader.

        Returns
        -------
        dict
            Dictionary of scalar loss values.
        """

        self.model.train()

        batch = move_to_device(
            batch,
            self.device,
        )

        #######################################################################
        # Validate batch
        #######################################################################

        validate_batch(

            batch,

            {

                "agent_trajectories",

                "lane_centerlines",

                "positions",

                "headings",

                "graph",

                "future_trajectories",

            },

        )

        #######################################################################
        # Zero gradients
        #######################################################################

        self.optimizer.zero_grad(
            set_to_none=True,
        )

        #######################################################################
        # Forward
        #######################################################################

        coarse_prediction, refined_prediction = self.model(

            agent_trajectories=batch["agent_trajectories"],

            lane_centerlines=batch["lane_centerlines"],

            positions=batch["positions"],

            headings=batch["headings"],

            graph=batch["graph"],

            agent_mask=batch.get(
                "agent_mask",
            ),

            lane_mask=batch.get(
                "lane_mask",
            ),
        )

        #######################################################################
        # Loss
        #######################################################################

        losses = self.criterion(

            coarse_prediction,

            refined_prediction,

            batch["future_trajectories"],
        )

        if "loss" not in losses:

            raise KeyError(
                "TotalLoss must return a dictionary "
                "containing a scalar tensor under the "
                "'loss' key."
            )

        total_loss = losses["loss"]

        #######################################################################
        # Backward
        #######################################################################

        total_loss.backward()

        #######################################################################
        # Gradient Clipping
        #######################################################################

        if self.gradient_clip is not None:

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip,

            )

        #######################################################################
        # Optimizer Step
        #######################################################################

        self.optimizer.step()

        #######################################################################
        # Scheduler Step
        #######################################################################

        if self.scheduler is not None:

            self.scheduler.step()

        #######################################################################
        # Convert losses to Python scalars
        #######################################################################

        metrics: dict[str, float] = {}

        for key, value in losses.items():

            if torch.is_tensor(value):

                metrics[key] = value.detach().item()

            elif isinstance(value, (int, float)):

                metrics[key] = float(value)

        return metrics

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "TrainStep("
            f"device={self.device}, "
            f"gradient_clip={self.gradient_clip})"
        )
