"""
evaluation.inference

Inference utilities for DSTNet.

Provides reusable interfaces for

- single-scene inference
- batched inference
- evaluation mode
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from engine.utils import move_to_device
from models.model_types import (
    Prediction,
    RefinedPrediction,
)

class InferenceEngine:
    """
    Lightweight inference wrapper for DSTNet.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:

        self.device = torch.device(device)

        self.model = model.to(
            self.device,
        )

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[
        Prediction,
        RefinedPrediction,
    ]:
        """
        Run inference on one batch.
        """

        batch = move_to_device(
            batch,
            self.device,
        )

        outputs = self.model(

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

        if (
            not isinstance(outputs, tuple)
            or len(outputs) != 2
        ):

            raise RuntimeError(
                "DSTNet.forward() must return "
                "(Prediction, RefinedPrediction)."
            )

        return outputs

    @torch.no_grad()
    def predict_loader(
        self,
        dataloader,
    ):
        """
        Generator over a DataLoader.
        """

        for batch in dataloader:

            yield self.predict(
                batch,
            )

    def __repr__(
        self,
    ) -> str:

        return (
            "InferenceEngine("
            f"device={self.device})"
        )

@torch.no_grad()
def infer(
    model: nn.Module,
    batch: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[
    Prediction,
    RefinedPrediction,
]:
    """
    One-shot inference.
    """

    engine = InferenceEngine(
        model=model,
        device=device,
    )

    return engine.predict(
        batch,
    )


