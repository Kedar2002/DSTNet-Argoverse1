"""
losses.classification_loss

Classification loss for multimodal trajectory prediction.

The confidence score of the trajectory whose proposal best
matches the ground-truth trajectory is supervised using
cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import Prediction

class ClassificationLoss(nn.Module):
    """
    Classification loss.

    Inputs
    ------

    prediction
        Prediction dataclass

    best_mode
        (B,N)

    Returns
    -------
    scalar loss
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        prediction: Prediction,
        best_mode: torch.Tensor,
    ) -> torch.Tensor:

        scores = prediction.scores

        if scores.ndim != 3:
            raise ValueError(
                "Scores must have shape (B,N,K)."
            )

        if best_mode.ndim != 2:
            raise ValueError(
                "best_mode must have shape (B,N)."
            )

        B, N, K = scores.shape

        logits = scores.reshape(
            B * N,
            K,
        )

        target = best_mode.reshape(
            B * N,
        )

        return F.cross_entropy(
            logits,
            target,
        )

    def __repr__(
        self,
    ) -> str:

        return "ClassificationLoss()"

