"""
losses.proposal_loss

DSTNet coarse proposal regression loss.

Paper
-----
DSTNet, Section III-G, Eq. (28) and Eq. (30)

Winner-Takes-All selection is performed using the minimum
endpoint distance.

The selected proposal is then optimized using Huber loss.

Current tensor contract
-----------------------

Prediction trajectories:

    (B,N,H,K,T,2)

Ground truth:

    (B,N,T,2)

Best mode:

    (B,N,H)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.model_types import Prediction

from losses.targets import (
    best_mode_from_endpoint,
    validate_trajectory_shapes,
)


class ProposalLoss(nn.Module):
    """
    DSTNet proposal regression loss.

    Implements:

        k_best = argmin_k endpoint_distance

    followed by:

        L_proposal =
            L_Huber(
                Y^(0)_{k_best},
                G
            )
    """

    def __init__(
        self,
        delta: float = 1.0,
    ) -> None:

        super().__init__()

        if delta <= 0.0:
            raise ValueError(
                "delta must be positive."
            )

        self.delta = float(delta)

    def forward(
        self,
        prediction: Prediction,
        ground_truth: torch.Tensor,
        *,
        return_best_mode: bool = False,
    ):
        """
        Parameters
        ----------
        prediction
            DSTNet coarse prediction.

        ground_truth
            Shape:

                (B,N,T,2)

        return_best_mode
            If True, also return the WTA mode indices.

        Returns
        -------
        loss

            Scalar Huber proposal loss.

        or

        loss, best_mode

            where best_mode has shape (B,N,H).
        """

        if not isinstance(
            prediction,
            Prediction,
        ):
            raise TypeError(
                "prediction must be a Prediction."
            )

        trajectories = prediction.trajectories

        validate_trajectory_shapes(
            trajectories,
            ground_truth,
        )

        #######################################################################
        # Winner-Takes-All selection
        #######################################################################

        _, best_mode = best_mode_from_endpoint(
            trajectories,
            ground_truth,
        )

        #######################################################################
        # Dimensions
        #######################################################################

        B, N, H, K, T, C = trajectories.shape

        #######################################################################
        # Gather the winning mode
        #######################################################################

        gather_index = (
            best_mode
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                B,
                N,
                H,
                1,
                T,
                C,
            )
        )

        best_prediction = torch.gather(
            trajectories,
            dim=3,
            index=gather_index,
        ).squeeze(3)

        #######################################################################
        # Expand ground truth over H
        #######################################################################

        gt = (
            ground_truth
            .unsqueeze(2)
            .expand(
                B,
                N,
                H,
                T,
                C,
            )
        )

        #######################################################################
        # Huber regression
        #######################################################################

        loss = F.huber_loss(
            best_prediction,
            gt,
            reduction="mean",
            delta=self.delta,
        )

        #######################################################################
        # Return
        #######################################################################

        if return_best_mode:
            return (
                loss,
                best_mode,
            )

        return loss

    def __repr__(
        self,
    ) -> str:

        return (
            "ProposalLoss("
            f"delta={self.delta})"
        )


__all__ = [
    "ProposalLoss",
]
