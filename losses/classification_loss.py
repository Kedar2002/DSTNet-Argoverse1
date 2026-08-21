"""
losses.classification_loss

DSTNet multimodal classification loss.

Paper
-----
DSTNet, Section III-G, Eq. (32)

The confidence distribution over K trajectory modes is
supervised using cross-entropy against the Winner-Takes-All
mode selected using Eq. (28).

Current tensor contract
-----------------------

Scores:

    (B,N,H,K)

Best mode:

    (B,N,H)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import Prediction


class ClassificationLoss(nn.Module):
    """
    DSTNet multimodal classification loss.

    Cross entropy is applied independently for every:

        (batch, agent, historical timestep)

    classification problem.

    The K prediction modes form the class dimension.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

    def forward(
        self,
        prediction: Prediction,
        best_mode: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prediction
            Coarse Prediction.

            scores:

                (B,N,H,K)

        best_mode
            WTA target.

            Shape:

                (B,N,H)

        Returns
        -------
        torch.Tensor
            Scalar classification loss.
        """

        if not isinstance(
            prediction,
            Prediction,
        ):
            raise TypeError(
                "prediction must be a Prediction."
            )

        scores = prediction.scores

        #######################################################################
        # Validate scores
        #######################################################################

        if not isinstance(
            scores,
            torch.Tensor,
        ):
            raise TypeError(
                "prediction.scores must be a torch.Tensor."
            )

        if scores.ndim != 4:
            raise ValueError(
                "prediction.scores must have shape "
                "(B,N,H,K). "
                f"Got {tuple(scores.shape)}."
            )

        #######################################################################
        # Validate target
        #######################################################################

        if not isinstance(
            best_mode,
            torch.Tensor,
        ):
            raise TypeError(
                "best_mode must be a torch.Tensor."
            )

        if best_mode.ndim != 3:
            raise ValueError(
                "best_mode must have shape "
                "(B,N,H). "
                f"Got {tuple(best_mode.shape)}."
            )

        B, N, H, K = scores.shape

        if best_mode.shape != (
            B,
            N,
            H,
        ):
            raise ValueError(
                "best_mode shape must match "
                "(B,N,H) of prediction.scores."
            )

        #######################################################################
        # Cross entropy expects:
        #
        # logits  -> (samples, classes)
        # target  -> (samples,)
        #######################################################################

        logits = scores.reshape(
            B * N * H,
            K,
        )

        target = best_mode.reshape(
            B * N * H,
        ).long()

        #######################################################################
        # Eq. (32)
        #######################################################################

        return F.cross_entropy(
            logits,
            target,
        )

    def __repr__(
        self,
    ) -> str:

        return "ClassificationLoss()"


__all__ = [
    "ClassificationLoss",
]
