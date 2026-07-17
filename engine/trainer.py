"""
engine.trainer

Training loop for DSTNet.

Responsibilities
----------------
- Epoch loop
- Training loop
- Validation loop
- Checkpointing
- Logging
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.train_step import TrainStep
from engine.checkpoint import (
    save_checkpoint,
    save_best_checkpoint,
)

class Trainer:
    """
    High-level trainer for DSTNet.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device | str = "cpu",
        checkpoint_dir: str = "checkpoints",
        gradient_clip: float | None = 1.0,
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.checkpoint_dir = Path(
            checkpoint_dir,
        )

        self.train_step = TrainStep(

            model=self.model,

            criterion=self.criterion,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            gradient_clip=gradient_clip,

            device=self.device,

        )

        self.best_metric = float("inf")

    #######################################################################
    # Train
    #######################################################################

    def train_epoch(
        self,
        epoch: int,
    ) -> dict[str, float]:

        running = {}

        start = perf_counter()

        for batch in self.train_loader:

            metrics = self.train_step(
                batch,
            )

            for key, value in metrics.items():

                running[key] = (
                    running.get(key, 0.0)
                    + value
                )

        num_batches = len(
            self.train_loader,
        )

        for key in running:

            running[key] /= num_batches

        running["epoch_time"] = (
            perf_counter() - start
        )

        print(

            f"[Epoch {epoch}] "

            f"Loss={running['loss']:.4f}"

        )

        return running

    #######################################################################
    # Validation
    #######################################################################

    @torch.no_grad()
    def validate(
        self,
    ) -> dict[str, float]:

        if self.val_loader is None:

            return {}

        self.model.eval()

        running = {}

        for batch in self.val_loader:

            batch = self.train_step._to_device(
                batch,
            )

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

            losses = self.criterion(

                coarse_prediction,

                refined_prediction,

                batch["future_trajectories"],
            )

            for key, value in losses.items():

                running[key] = (

                    running.get(key, 0.0)

                    + value.detach().item()

                )

        num_batches = len(
            self.val_loader,
        )

        for key in running:

            running[key] /= num_batches

        return running

    #######################################################################
    # Main Loop
    #######################################################################

    def fit(
        self,
        epochs: int,
    ) -> None:

        global_step = 0

        for epoch in range(1, epochs + 1):

            train_metrics = self.train_epoch(
                epoch,
            )

            val_metrics = self.validate()

            metric = val_metrics.get(

                "loss",

                train_metrics["loss"],

            )

            save_checkpoint(

                path=self.checkpoint_dir
                / "latest.pth",

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch,

                global_step=global_step,

                metrics=train_metrics,

            )

            self.best_metric = save_best_checkpoint(

                checkpoint_dir=self.checkpoint_dir,

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch,

                global_step=global_step,

                metric=metric,

                best_metric=self.best_metric,

            )

            global_step += len(
                self.train_loader,
            )

            print()

            print("=" * 60)

            print(f"Epoch {epoch}")

            print("=" * 60)

            print("Train")

            for k, v in train_metrics.items():

                print(f"{k:20s}: {v:.6f}")

            if val_metrics:

                print()

                print("Validation")

                for k, v in val_metrics.items():

                    print(f"{k:20s}: {v:.6f}")

            print()

    def __repr__(
        self,
    ) -> str:

        return (

            "Trainer("

            f"device={self.device}, "

            f"checkpoint_dir='{self.checkpoint_dir}')"

        )

