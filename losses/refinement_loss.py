"""
losses.refinement_loss

Loss for Adaptive Anchor-based Refinement (AAR).

DSTNet training contract
------------------------

Coarse trajectories:

    Y^(0) : (B,N,H,K,T,2)

Refined trajectories:

    Y^R   : (B,N,H,K,T,2)

Ground truth:

    G     : (B,N,T,2)

The historical dimension H is intentionally preserved.

The loss supervises the refined trajectory associated with the
Winner-Takes-All mode selected from the endpoint criterion used
for DSTNet target construction.

The objective contains:

    1. Refined trajectory regression
    2. Endpoint accuracy
    3. Bezier smoothness regularization
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.model_types import RefinedPrediction

from models.refinement.bezier import (
    bezier_regularization,
)

from losses.targets import (
    validate_trajectory_shapes,
    best_mode_from_endpoint,
)


###############################################################################
# Refinement Loss
###############################################################################


class RefinementLoss(nn.Module):
    """
    Loss for refined DSTNet trajectories.

    Parameters
    ----------
    trajectory_weight
        Weight of the full refined-trajectory regression term.

    endpoint_weight
        Weight of the final-point regression term.

    smoothness_weight
        Weight of the Bezier regularization term.

    Notes
    -----
    The loss operates on:

        (B,N,H,K,T,2)

    rather than the older:

        (B,N,K,T,2)

    representation.
    """

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        endpoint_weight: float = 0.5,
        smoothness_weight: float = 0.05,
    ) -> None:

        super().__init__()

        #######################################################################
        # Validate configuration
        #######################################################################

        if trajectory_weight < 0.0:
            raise ValueError(
                "trajectory_weight must be non-negative."
            )

        if endpoint_weight < 0.0:
            raise ValueError(
                "endpoint_weight must be non-negative."
            )

        if smoothness_weight < 0.0:
            raise ValueError(
                "smoothness_weight must be non-negative."
            )

        self.trajectory_weight = (
            float(trajectory_weight)
        )

        self.endpoint_weight = (
            float(endpoint_weight)
        )

        self.smoothness_weight = (
            float(smoothness_weight)
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        prediction: RefinedPrediction,
        ground_truth: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute the refined trajectory loss.

        Parameters
        ----------
        prediction
            Refined DSTNet prediction.

            prediction.trajectories:

                (B,N,H,K,T,2)

        ground_truth
            Future ground truth.

            Shape:

                (B,N,T,2)

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing:

                loss
                regression
                endpoint
                smoothness
        """

        #######################################################################
        # Prediction validation
        #######################################################################

        if not isinstance(
            prediction,
            RefinedPrediction,
        ):
            raise TypeError(
                "prediction must be a RefinedPrediction."
            )

        refined = prediction.trajectories

        if not isinstance(
            refined,
            torch.Tensor,
        ):
            raise TypeError(
                "prediction.trajectories must be a torch.Tensor."
            )

        #######################################################################
        # Validate trajectory contract
        #######################################################################

        validate_trajectory_shapes(
            refined,
            ground_truth,
        )

        #######################################################################
        # Device / numerical checks
        #######################################################################

        if not torch.isfinite(
            refined,
        ).all():
            raise ValueError(
                "Refined trajectories contain NaN or infinite values."
            )

        #######################################################################
        # WTA target
        #######################################################################
        #
        # Endpoint criterion:
        #
        #     endpoint_error : (B,N,H,K)
        #
        #     best_mode      : (B,N,H)
        #
        #######################################################################

        endpoint_error, best_mode = (
            best_mode_from_endpoint(
                refined,
                ground_truth,
            )
        )

        #######################################################################
        # Gather best refined trajectory
        #
        # refined:
        #
        #     (B,N,H,K,T,2)
        #
        # best_mode:
        #
        #     (B,N,H)
        #
        # gather index:
        #
        #     (B,N,H,1,T,2)
        #
        # selected:
        #
        #     (B,N,H,T,2)
        #######################################################################

        B = refined.shape[0]
        N = refined.shape[1]
        H = refined.shape[2]
        T = refined.shape[4]

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
                2,
            )
        )

        best_prediction = torch.gather(
            refined,
            dim=3,
            index=gather_index,
        ).squeeze(3)

        #######################################################################
        # Validate selected trajectory
        #######################################################################

        expected_best_shape = (
            B,
            N,
            H,
            T,
            2,
        )

        if tuple(
            best_prediction.shape
        ) != expected_best_shape:
            raise RuntimeError(
                "Unexpected best refined trajectory shape: "
                f"expected {expected_best_shape}, "
                f"got {tuple(best_prediction.shape)}."
            )

        #######################################################################
        # Expand ground truth over historical dimension
        #
        # ground_truth:
        #
        #     (B,N,T,2)
        #
        # becomes:
        #
        #     (B,N,H,T,2)
        #######################################################################

        gt = (
            ground_truth
            .unsqueeze(2)
            .expand(
                B,
                N,
                H,
                T,
                2,
            )
        )

        #######################################################################
        # Trajectory regression
        #######################################################################

        regression_loss = F.smooth_l1_loss(
            best_prediction,
            gt,
        )

        #######################################################################
        # Endpoint accuracy
        #######################################################################

        predicted_endpoint = (
            best_prediction[
                ...,
                -1,
                :,
            ]
        )

        ground_truth_endpoint = (
            ground_truth[
                ...,
                -1,
                :,
            ]
        )

        ground_truth_endpoint = (
            ground_truth_endpoint
            .unsqueeze(2)
            .expand(
                B,
                N,
                H,
                2,
            )
        )

        endpoint_loss = F.smooth_l1_loss(
            predicted_endpoint,
            ground_truth_endpoint,
        )

        #######################################################################
        # Bezier regularization
        #######################################################################
        #
        # bezier_regularization() may operate on the trajectory dimensions
        # expected by the refinement implementation.
        #
        # Apply it independently over the historical dimension H.
        #
        # Flatten B,N,H into a single leading dimension:
        #
        #     (B,N,H,T,2)
        #
        #         ↓
        #
        #     (B*N*H,T,2)
        #
        #######################################################################

        flattened_best = (
            best_prediction.reshape(
                B * N * H,
                T,
                2,
            )
        )

        flattened_gt = (
            gt.reshape(
                B * N * H,
                T,
                2,
            )
        )

        smoothness_loss = (
            bezier_regularization(
                flattened_gt,
                flattened_best,
            )
        )

        #######################################################################
        # Total loss
        #######################################################################

        total = (
            self.trajectory_weight
            * regression_loss
            +
            self.endpoint_weight
            * endpoint_loss
            +
            self.smoothness_weight
            * smoothness_loss
        )

        #######################################################################
        # Numerical validation
        #######################################################################

        losses = {
            "loss": total,
            "regression": regression_loss,
            "endpoint": endpoint_loss,
            "smoothness": smoothness_loss,
        }

        for name, value in losses.items():

            if not torch.isfinite(
                value,
            ).all():
                raise RuntimeError(
                    f"Refinement loss component '{name}' "
                    "contains NaN or infinite values."
                )

        return losses

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "RefinementLoss("
            f"trajectory={self.trajectory_weight}, "
            f"endpoint={self.endpoint_weight}, "
            f"smoothness={self.smoothness_weight})"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "RefinementLoss",
]
