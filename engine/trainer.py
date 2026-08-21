"""
engine.trainer

High-level DSTNet training loop.

Responsibilities
----------------
- Epoch-level training
- Validation
- Scheduler coordination
- Checkpointing
- Best-model tracking
- Global-step tracking
- Metric aggregation

Current model contract
----------------------

DSTNet.forward(...) returns:

    coarse_prediction, refined_prediction

Current loss contract
---------------------

TotalLoss(
    coarse_prediction,
    refined_prediction,
    ground_truth,
)

returns a dictionary containing:

    loss
    proposal_loss
    classification_loss
    score_loss
    refinement_loss

and refinement-loss diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.checkpoint import (
    save_best_checkpoint,
    save_checkpoint,
)
from engine.train_step import TrainStep
from engine.utils import move_to_device


class Trainer:
    """
    High-level trainer for DSTNet.

    Parameters
    ----------
    model:
        DSTNet model.

    criterion:
        TotalLoss.

    optimizer:
        Optimizer.

    scheduler:
        Optional learning-rate scheduler.

    train_loader:
        Training DataLoader.

    val_loader:
        Optional validation DataLoader.

    device:
        Training device.

    checkpoint_dir:
        Directory for checkpoints.

    gradient_clip:
        Maximum gradient norm.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | str = "cpu",
        checkpoint_dir: str = "checkpoints",
        gradient_clip: float | None = 1.0,
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
            torch.optim.Optimizer,
        ):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer."
            )

        if gradient_clip is not None:
            if gradient_clip <= 0.0:
                raise ValueError(
                    "gradient_clip must be positive or None."
                )

        self.device = torch.device(
            device
        )

        self.model = model.to(
            self.device
        )

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.train_step = TrainStep(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            gradient_clip=gradient_clip,
            device=self.device,
        )

        self.best_metric = float(
            "inf"
        )

        self.epoch = 0
        self.global_step = 0

    ###########################################################################
    # Metric aggregation
    ###########################################################################

    @staticmethod
    def _accumulate_metrics(
        running: dict[str, float],
        metrics: dict[str, float],
    ) -> None:
        """
        Add one batch's metrics to running totals.
        """

        for key, value in metrics.items():

            if not isinstance(
                value,
                (float, int),
            ):
                raise TypeError(
                    f"Metric '{key}' must be numeric."
                )

            running[key] = (
                running.get(key, 0.0)
                + float(value)
            )

    @staticmethod
    def _average_metrics(
        running: dict[str, float],
        num_batches: int,
    ) -> dict[str, float]:
        """
        Average batch metrics.
        """

        if num_batches <= 0:
            return {}

        return {
            key: value / float(
                num_batches
            )
            for key, value in running.items()
        }

    ###########################################################################
    # Training epoch
    ###########################################################################

    def train_epoch(
        self,
        epoch: int,
    ) -> dict[str, float]:
        """
        Train for one epoch.
        """

        self.model.train()

        running: dict[str, float] = {}

        start = perf_counter()

        num_batches = 0

        for batch in self.train_loader:

            metrics = self.train_step(
                batch
            )

            self._accumulate_metrics(
                running,
                metrics,
            )

            num_batches += 1

            self.global_step += 1

        if num_batches == 0:

            raise RuntimeError(
                "Training DataLoader produced zero batches."
            )

        metrics = self._average_metrics(
            running,
            num_batches,
        )

        metrics[
            "epoch_time"
        ] = (
            perf_counter()
            - start
        )

        print(
            f"[Epoch {epoch}] "
            f"Loss={metrics['loss']:.6f}"
        )

        return metrics

    ###########################################################################
    # Validation
    ###########################################################################

    @torch.no_grad()
    def validate(
        self,
    ) -> dict[str, float]:
        """
        Run validation without optimizer/scheduler updates.
        """

        if self.val_loader is None:
            return {}

        self.model.eval()

        running: dict[str, float] = {}

        num_batches = 0

        for batch in self.val_loader:

            ###################################################################
            # IMPORTANT:
            #
            # Do not call TrainStep here.
            #
            # TrainStep performs backward(), optimizer.step(), and
            # scheduler.step().
            #
            # Validation only performs:
            #
            # batch -> model -> TotalLoss
            ###################################################################

            batch = move_to_device(
                batch,
                self.device,
            )

            required = {
                "agent_trajectories",
                "lane_centerlines",
                "positions",
                "headings",
                "graph",
                "future_trajectories",
            }

            missing = [
                key
                for key in required
                if key not in batch
            ]

            if missing:

                raise KeyError(
                    "Validation batch is missing "
                    f"required fields: {missing}"
                )

            coarse_prediction, refined_prediction = (
                self.model(
                    agent_trajectories=batch[
                        "agent_trajectories"
                    ],
                    lane_centerlines=batch[
                        "lane_centerlines"
                    ],
                    positions=batch[
                        "positions"
                    ],
                    headings=batch[
                        "headings"
                    ],
                    graph=batch[
                        "graph"
                    ],
                    agent_mask=batch.get(
                        "agent_mask"
                    ),
                    lane_mask=batch.get(
                        "lane_mask"
                    ),
                )
            )

            losses = self.criterion(
                coarse_prediction,
                refined_prediction,
                batch[
                    "future_trajectories"
                ],
            )

            if not isinstance(
                losses,
                dict,
            ):
                raise TypeError(
                    "TotalLoss must return a dictionary."
                )

            if "loss" not in losses:
                raise KeyError(
                    "TotalLoss output must contain 'loss'."
                )

            for key, value in losses.items():

                if not isinstance(
                    value,
                    torch.Tensor,
                ):
                    continue

                if value.ndim != 0:
                    continue

                scalar = float(
                    value.detach().item()
                )

                if not torch.isfinite(
                    value
                ).item():
                    raise RuntimeError(
                        f"Validation loss '{key}' "
                        "is non-finite."
                    )

                running[key] = (
                    running.get(key, 0.0)
                    + scalar
                )

            num_batches += 1

        if num_batches == 0:

            raise RuntimeError(
                "Validation DataLoader produced zero batches."
            )

        return self._average_metrics(
            running,
            num_batches,
        )

    ###########################################################################
    # Checkpoint
    ###########################################################################

    def _save_latest(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """
        Save latest checkpoint.
        """

        save_checkpoint(
            path=(
                self.checkpoint_dir
                / "latest.pth"
            ),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            metrics=metrics,
        )

    ###########################################################################
    # Main training loop
    ###########################################################################

    def fit(
        self,
        epochs: int,
    ) -> None:
        """
        Train the model for ``epochs`` epochs.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        for epoch in range(
            self.epoch + 1,
            epochs + 1,
        ):

            self.epoch = epoch

            ###################################################################
            # Training
            ###################################################################

            train_metrics = (
                self.train_epoch(
                    epoch
                )
            )

            ###################################################################
            # Validation
            ###################################################################

            val_metrics = self.validate()

            ###################################################################
            # Model-selection metric
            #
            # Lower loss is better.
            ###################################################################

            metric = float(
                val_metrics.get(
                    "loss",
                    train_metrics["loss"],
                )
            )

            ###################################################################
            # Latest checkpoint
            ###################################################################

            self._save_latest(
                epoch=epoch,
                metrics=train_metrics,
            )

            ###################################################################
            # Best checkpoint
            ###################################################################

            self.best_metric = (
                save_best_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    metric=metric,
                    best_metric=self.best_metric,
                    maximize=False,
                )
            )

            ###################################################################
            # Reporting
            ###################################################################

            print()
            print("=" * 70)
            print(
                f"Epoch {epoch}/{epochs}"
            )
            print("=" * 70)

            print("Train")

            for key, value in (
                train_metrics.items()
            ):

                print(
                    f"{key:24s}: "
                    f"{value:.6f}"
                )

            if val_metrics:

                print()
                print("Validation")

                for key, value in (
                    val_metrics.items()
                ):

                    print(
                        f"{key:24s}: "
                        f"{value:.6f}"
                    )

            print()
            print(
                f"Global step : "
                f"{self.global_step}"
            )

            print(
                f"Best loss   : "
                f"{self.best_metric:.6f}"
            )

            print()

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Trainer("
            f"device={self.device}, "
            f"checkpoint_dir="
            f"'{self.checkpoint_dir}', "
            f"global_step={self.global_step})"
        )


__all__ = [
    "Trainer",
]
