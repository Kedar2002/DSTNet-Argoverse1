"""
DSTNet refinement confidence-score loss.

Paper
-----
DSTNet, Section III-G, Eq. (26) and Eq. (33)

The refinement module produces a confidence score for every
refinement iteration. The target score is constructed from the
endpoint error of the corresponding trajectory:

    RScore_i =
        1 - (e_i - e_min) / (e_max - e_min)

where

    e_i   = endpoint error of Y^(i)
    e_min = minimum endpoint error across refinement iterations
    e_max = maximum endpoint error across refinement iterations.

Eq. (33) supervises the predicted refinement scores with an L1 loss.

Current tensor contract
-----------------------

Refined trajectories:

    (B,N,H,K,T,2)

Trajectory history:

    (B,N,H,K,C+1,T,2)

Predicted refinement-score history:

    (B,N,H,K,C+1)

Ground truth:

    (B,N,T,2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.model_types import RefinedPrediction


class ScoreLoss(nn.Module):
    """
    Confidence calibration loss for Adaptive Anchor-based Refinement.

    The loss is evaluated over all stored refinement iterations,
    including the initial coarse prediction Y^(0).

    Parameters
    ----------
    reduction
        Reduction applied to the per-element L1 loss.

        Supported values:

            "mean"
            "sum"
            "none"

    eps
        Numerical stabilizer used when all refinement iterations
        have identical endpoint error.
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:

        super().__init__()

        if reduction not in {
            "mean",
            "sum",
            "none",
        }:
            raise ValueError(
                "reduction must be one of "
                "{'mean', 'sum', 'none'}."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.reduction = reduction
        self.eps = float(eps)

    def forward(
        self,
        prediction: RefinedPrediction,
        ground_truth: Tensor,
    ) -> Tensor:
        """
        Compute the refinement confidence-score loss.

        Parameters
        ----------
        prediction
            RefinedPrediction containing:

                trajectory_history:
                    (B,N,H,K,C+1,T,2)

                refinement_score_history:
                    (B,N,H,K,C+1)

        ground_truth
            Future ground truth:

                (B,N,T,2)

        Returns
        -------
        torch.Tensor
            Scalar loss for "mean"/"sum" reduction.

            For "none", returns the unreduced loss tensor.
        """

        #######################################################################
        # Prediction type
        #######################################################################

        if not isinstance(
            prediction,
            RefinedPrediction,
        ):
            raise TypeError(
                "prediction must be a RefinedPrediction."
            )

        #######################################################################
        # Required refinement histories
        #######################################################################

        trajectory_history = (
            prediction.trajectory_history
        )

        score_history = (
            prediction.refinement_score_history
        )

        if trajectory_history is None:
            raise ValueError(
                "RefinedPrediction.trajectory_history is required "
                "for ScoreLoss."
            )

        if score_history is None:
            raise ValueError(
                "RefinedPrediction.refinement_score_history is required "
                "for ScoreLoss."
            )

        #######################################################################
        # Ground truth type
        #######################################################################

        if not isinstance(
            ground_truth,
            torch.Tensor,
        ):
            raise TypeError(
                "ground_truth must be a torch.Tensor."
            )

        #######################################################################
        # Shape validation
        #######################################################################

        if trajectory_history.ndim != 7:
            raise ValueError(
                "trajectory_history must have shape "
                "(B,N,H,K,C+1,T,2). "
                f"Got {tuple(trajectory_history.shape)}."
            )

        if score_history.ndim != 5:
            raise ValueError(
                "refinement_score_history must have shape "
                "(B,N,H,K,C+1). "
                f"Got {tuple(score_history.shape)}."
            )

        if ground_truth.ndim != 4:
            raise ValueError(
                "ground_truth must have shape "
                "(B,N,T,2). "
                f"Got {tuple(ground_truth.shape)}."
            )

        #######################################################################
        # Dimensions
        #######################################################################

        (
            B,
            N,
            H,
            K,
            C_plus_one,
            T,
            coordinate_dim,
        ) = trajectory_history.shape

        #######################################################################
        # Coordinate dimension
        #######################################################################

        if coordinate_dim != 2:
            raise ValueError(
                "Trajectory coordinate dimension must be 2. "
                f"Got {coordinate_dim}."
            )

        #######################################################################
        # Score history shape
        #######################################################################

        expected_score_shape = (
            B,
            N,
            H,
            K,
            C_plus_one,
        )

        if tuple(
            score_history.shape
        ) != expected_score_shape:
            raise ValueError(
                "refinement_score_history must have shape "
                f"{expected_score_shape}. "
                f"Got {tuple(score_history.shape)}."
            )

        #######################################################################
        # Ground-truth shape
        #######################################################################

        expected_gt_shape = (
            B,
            N,
            T,
            2,
        )

        if tuple(
            ground_truth.shape
        ) != expected_gt_shape:
            raise ValueError(
                "ground_truth must have shape "
                f"{expected_gt_shape}. "
                f"Got {tuple(ground_truth.shape)}."
            )

        #######################################################################
        # Device compatibility
        #######################################################################

        if (
            trajectory_history.device
            != ground_truth.device
        ):
            raise ValueError(
                "trajectory_history and ground_truth "
                "must be on the same device."
            )

        if (
            score_history.device
            != ground_truth.device
        ):
            raise ValueError(
                "refinement_score_history and ground_truth "
                "must be on the same device."
            )

        #######################################################################
        # Floating-point validation
        #######################################################################

        if not torch.is_floating_point(
            trajectory_history,
        ):
            raise TypeError(
                "trajectory_history must contain floating-point values."
            )

        if not torch.is_floating_point(
            score_history,
        ):
            raise TypeError(
                "refinement_score_history must contain "
                "floating-point values."
            )

        if not torch.is_floating_point(
            ground_truth,
        ):
            raise TypeError(
                "ground_truth must contain floating-point values."
            )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.isfinite(
            trajectory_history,
        ).all():
            raise ValueError(
                "trajectory_history contains NaN or infinite values."
            )

        if not torch.isfinite(
            score_history,
        ).all():
            raise ValueError(
                "refinement_score_history contains NaN or infinite values."
            )

        if not torch.isfinite(
            ground_truth,
        ).all():
            raise ValueError(
                "ground_truth contains NaN or infinite values."
            )

        #######################################################################
        # Endpoint errors for every refinement iteration
        #
        # trajectory_history:
        #
        #     (B,N,H,K,C+1,T,2)
        #
        # final point:
        #
        #     (B,N,H,K,C+1,2)
        #######################################################################

        predicted_endpoint = trajectory_history[
            ...,
            -1,
            :,
        ]

        #######################################################################
        # Ground-truth final point:
        #
        #     (B,N,2)
        #
        # Broadcast to:
        #
        #     (B,N,H,K,C+1,2)
        #######################################################################

        ground_truth_endpoint = ground_truth[
            ...,
            -1,
            :,
        ]

        ground_truth_endpoint = (
            ground_truth_endpoint
            .unsqueeze(2)
            .unsqueeze(2)
            .unsqueeze(2)
        )

        #######################################################################
        # Euclidean endpoint error
        #
        # Result:
        #
        #     (B,N,H,K,C+1)
        #######################################################################

        endpoint_error = torch.linalg.vector_norm(
            predicted_endpoint
            - ground_truth_endpoint,
            dim=-1,
        )

        if not torch.isfinite(
            endpoint_error,
        ).all():
            raise RuntimeError(
                "Endpoint error contains NaN or infinite values."
            )

        #######################################################################
        # Eq. (26)
        #
        # e_min and e_max are computed across refinement iterations.
        #
        #     e_min : (B,N,H,K,1)
        #     e_max : (B,N,H,K,1)
        #######################################################################

        error_min = endpoint_error.min(
            dim=-1,
            keepdim=True,
        ).values

        error_max = endpoint_error.max(
            dim=-1,
            keepdim=True,
        ).values

        error_range = (
            error_max
            - error_min
        )

        #######################################################################
        # Normalized endpoint improvement
        #
        #     (e_i - e_min)
        #     ----------------
        #       e_max-e_min
        #
        # Best trajectory:
        #
        #     e_i = e_min
        #
        #     target = 1
        #
        # Worst trajectory:
        #
        #     e_i = e_max
        #
        #     target = 0
        #######################################################################

        normalized_improvement = torch.where(
            error_range > self.eps,
            (
                endpoint_error
                - error_min
            )
            / error_range,
            torch.zeros_like(
                endpoint_error,
            ),
        )

        target_scores = (
            1.0
            - normalized_improvement
        )

        #######################################################################
        # Numerical protection
        #######################################################################

        target_scores = torch.clamp(
            target_scores,
            min=0.0,
            max=1.0,
        )

        #######################################################################
        # Eq. (33)
        #
        # L_score =
        #
        #     1/(C+1)
        #     Σ_i
        #     |RScore_i - target_i|
        #
        # Since F.l1_loss averages over all dimensions for "mean",
        # this is equivalent to uniform averaging over all stored
        # refinement scores.
        #######################################################################

        loss = F.l1_loss(
            score_history,
            target_scores,
            reduction=self.reduction,
        )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.isfinite(
            loss,
        ).all():
            raise RuntimeError(
                "ScoreLoss produced NaN or infinite values."
            )

        return loss

    def __repr__(
        self,
    ) -> str:

        return (
            "ScoreLoss("
            f"reduction={self.reduction}, "
            f"eps={self.eps})"
        )


__all__ = [
    "ScoreLoss",
]
