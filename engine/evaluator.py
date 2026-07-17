"""
engine.evaluator

Evaluation engine for DSTNet.

Responsibilities
----------------
- Forward inference
- Metric computation
- Benchmark integration
- Visualization hooks
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_metrics


class Evaluator:
    """
    Evaluates a trained DSTNet model.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> None:

        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

        self.model.to(self.device)

    ###########################################################################
    # Device Transfer
    ###########################################################################

    def _to_device(
        self,
        batch: Any,
    ) -> Any:

        if isinstance(batch, torch.Tensor):

            return batch.to(
                self.device,
                non_blocking=True,
            )

        if isinstance(batch, Mapping):

            return {
                key: self._to_device(value)
                for key, value in batch.items()
            }

        if isinstance(batch, (list, tuple)):

            return type(batch)(
                self._to_device(value)
                for value in batch
            )

        return batch

    ###########################################################################
    # Evaluation
    ###########################################################################

    @torch.no_grad()
    def evaluate(
        self,
    ) -> dict[str, float]:

        self.model.eval()

        running: dict[str, float] = {}

        num_batches = 0

        for batch in self.dataloader:

            batch = self._to_device(batch)

            _, refined_prediction = self.model(

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

            batch_metrics = compute_metrics(

                refined_prediction,

                batch["future_trajectories"],

            )

            for key, value in batch_metrics.items():

                running[key] = (

                    running.get(key, 0.0)

                    + value

                )

            num_batches += 1

        if num_batches == 0:

            return {}

        ###############################################################
        # Average metrics
        ###############################################################

        for key in running:

            running[key] /= num_batches

        return running

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "Evaluator("
            f"device={self.device})"
        )
