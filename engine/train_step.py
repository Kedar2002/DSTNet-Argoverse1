"""
engine.train_step

Single optimization step for DSTNet.

Responsibilities
----------------
- Move batch to device
- Validate batch
- Forward pass
- Complete TotalLoss computation
- Backward pass
- Gradient clipping
- Optimizer update
- Scheduler update
- Metric extraction

Current DSTNet contract
-----------------------

Model input:

    agent_trajectories : (B,N,H,2)
    map_centerlines   : (B,M,P,2)
    positions          : (B,N,2)
    graph              : list[SceneGraph]
    agent_mask         : (B,N)
    map_mask          : (B,M)

Dataset batch may additionally contain:

    headings           : (B,N)

Model output:

    Prediction
    RefinedPrediction

TotalLoss input:

    Prediction
    RefinedPrediction
    ground_truth : (B,N,T,2)

TotalLoss output:

    {
        "loss": Tensor,
        "proposal_loss": Tensor,
        "classification_loss": Tensor,
        "score_loss": Tensor,
        "refinement_loss": Tensor,
        ...
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from engine.utils import (
    move_to_device,
    validate_batch,
)


class TrainStep:
    """
    Execute one complete DSTNet optimization step.

    This class deliberately follows the current DSTNet model/loss
    interfaces without introducing another abstraction layer.

    Parameters
    ----------
    model:
        DSTNet model.

    criterion:
        TotalLoss instance.

    optimizer:
        PyTorch optimizer.

    scheduler:
        Optional learning-rate scheduler.

    gradient_clip:
        Maximum gradient norm. ``None`` disables clipping.

    device:
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

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        if not isinstance(
            criterion,
            nn.Module,
        ):
            raise TypeError(
                "criterion must be a torch.nn.Module."
            )

        if not isinstance(
            optimizer,
            Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer."
            )

        if gradient_clip is not None:
            if gradient_clip <= 0.0:
                raise ValueError(
                    "gradient_clip must be positive or None."
                )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)

        self.last_gradient_norm: float | None = None

    ###########################################################################
    # Batch validation
    ###########################################################################

    @staticmethod
    def _validate_required_outputs(
        losses: Mapping[str, Any],
    ) -> None:
        """
        Validate the dictionary returned by TotalLoss.
        """

        if not isinstance(
            losses,
            Mapping,
        ):
            raise TypeError(
                "TotalLoss must return a mapping."
            )

        if "loss" not in losses:
            raise KeyError(
                "TotalLoss output must contain 'loss'."
            )

        total_loss = losses["loss"]

        if not isinstance(
            total_loss,
            torch.Tensor,
        ):
            raise TypeError(
                "TotalLoss['loss'] must be a torch.Tensor."
            )

        if total_loss.ndim != 0:
            raise ValueError(
                "TotalLoss['loss'] must be scalar. "
                f"Got shape {tuple(total_loss.shape)}."
            )

        if not torch.isfinite(
            total_loss
        ).all():
            raise RuntimeError(
                "TotalLoss['loss'] contains NaN or infinite values."
            )

        for name, value in losses.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):
                continue

            if value.ndim == 0:

                if not torch.isfinite(
                    value
                ).all():
                    raise RuntimeError(
                        f"Loss component '{name}' "
                        "contains NaN or infinite values."
                    )

    ###########################################################################
    # Gradient validation
    ###########################################################################

    def _validate_gradients(self) -> float:
        """
        Validate gradients and return total gradient norm.

        Parameters with ``grad is None`` are allowed because some
        parameters may legitimately be unused by a particular branch.
        """

        total_squared_norm = 0.0
        found_gradient = False

        for parameter in self.model.parameters():

            if parameter.grad is None:
                continue

            found_gradient = True

            if not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError(
                    "Non-finite gradient detected."
                )

            gradient_norm = (
                parameter.grad.detach()
                .norm(2)
                .item()
            )

            total_squared_norm += (
                gradient_norm
                * gradient_norm
            )

        total_norm = (
            total_squared_norm ** 0.5
        )

        if not found_gradient:
            raise RuntimeError(
                "No model parameter received a gradient."
            )

        return float(total_norm)

    ###########################################################################
    # Forward
    ###########################################################################

    def _forward(
        self,
        batch: Mapping[str, Any],
    ):
        """
        Execute the current DSTNet forward contract.

        The collate layer uses the dataset-facing names:

            map_centerlines
            map_mask

        while DSTNet.forward() uses:

            map_centerlines
            map_mask

        headings remain available in the batch for preprocessing/
        diagnostics but are not currently part of DSTNet.forward().
        """

        return self.model(
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

    ###########################################################################
    # Training step
    ###########################################################################

    def __call__(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, float]:
        """
        Execute one complete optimization step.

        Returns
        -------
        dict[str, float]
            Scalar loss components and optimization diagnostics.
        """

        self.model.train()

        #######################################################################
        # Move batch
        #######################################################################

        batch = move_to_device(
            batch,
            self.device,
        )

        #######################################################################
        # Validate required batch fields
        #######################################################################

        validate_batch(
            batch,
            {
                "agent_trajectories",
                "map_centerlines",
                "positions",
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

        coarse_prediction, refined_prediction = (
            self._forward(batch)
        )

        #######################################################################
        # Complete objective
        #######################################################################

        losses = self.criterion(
            coarse_prediction,
            refined_prediction,
            batch["future_trajectories"],
        )

        self._validate_required_outputs(
            losses,
        )

        total_loss = losses[
            "loss"
        ]

        #######################################################################
        # Backward
        #######################################################################

        total_loss.backward()

        #######################################################################
        # Validate gradients BEFORE clipping
        #######################################################################

        pre_clip_gradient_norm = (
            self._validate_gradients()
        )

        #######################################################################
        # Gradient clipping
        #######################################################################

        if self.gradient_clip is not None:

            clipped_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            )

            self.last_gradient_norm = float(
                clipped_norm.item()
                if isinstance(
                    clipped_norm,
                    torch.Tensor,
                )
                else clipped_norm
            )

        else:

            self.last_gradient_norm = (
                pre_clip_gradient_norm
            )

        #######################################################################
        # Optimizer update
        #######################################################################

        self.optimizer.step()

        #######################################################################
        # Scheduler update
        #
        # Current scheduler factory creates step-oriented schedulers
        # such as CosineAnnealingLR(total_steps=...).
        #######################################################################

        if self.scheduler is not None:

            self.scheduler.step()

        #######################################################################
        # Metric extraction
        #######################################################################

        metrics: dict[str, float] = {}

        for name, value in losses.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):
                continue

            if value.ndim != 0:
                continue

            metrics[name] = float(
                value.detach().item()
            )

        #######################################################################
        # Optimization diagnostics
        #######################################################################

        metrics[
            "gradient_norm"
        ] = float(
            self.last_gradient_norm
            if self.last_gradient_norm is not None
            else 0.0
        )

        metrics[
            "learning_rate"
        ] = float(
            self.optimizer.param_groups[0][
                "lr"
            ]
        )

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
            f"gradient_clip={self.gradient_clip}, "
            f"scheduler="
            f"{self.scheduler.__class__.__name__ if self.scheduler is not None else 'None'})"
        )


__all__ = [
    "TrainStep",
]
