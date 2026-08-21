"""
losses.targets

Ground-truth target construction for DSTNet training.

Paper
-----
DSTNet, Section III-G, Eq. (28)

The paper uses a Winner-Takes-All (WTA) strategy.

For each agent n, historical prediction time t, and
prediction mode k:

    k_best = argmin_k
        ||Y^(0)_{n,t,k,T} - G_{n,t,T}||_2

where the comparison is performed at the final prediction
endpoint.

Current implementation contract
--------------------------------

Coarse trajectories:

    (B, N, H, K, T, 2)

Ground truth:

    (B, N, T, 2)

Therefore:

    endpoint_error : (B, N, H, K)

and:

    best_mode : (B, N, H)

The historical dimension H is intentionally preserved.

Notes
-----

1. The WTA decision is made independently for every
   (batch, agent, historical timestep) tuple.

2. Only the final prediction point is used for selecting
   the winning mode, as specified by Eq. (28).

3. The returned best-mode index is differentiability-free
   because argmin is used only for target construction.

4. The endpoint error remains differentiable with respect
   to the trajectory tensor and may therefore be used by
   downstream regression-loss calculations.
"""

from __future__ import annotations

import torch


###############################################################################
# Shape validation
###############################################################################


def validate_trajectory_shapes(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:
    """
    Validate the DSTNet trajectory tensor contract.

    Parameters
    ----------
    trajectories
        Coarse multimodal trajectories.

        Shape:

            (B, N, H, K, T, 2)

    ground_truth
        Future ground-truth trajectory.

        Shape:

            (B, N, T, 2)

    Raises
    ------
    TypeError
        If either input is not a torch.Tensor or is not floating point.

    ValueError
        If tensor dimensions, coordinates, horizons, or numerical
        values violate the DSTNet contract.
    """

    ###########################################################################
    # Tensor type
    ###########################################################################

    if not isinstance(
        trajectories,
        torch.Tensor,
    ):
        raise TypeError(
            "trajectories must be a torch.Tensor."
        )

    if not isinstance(
        ground_truth,
        torch.Tensor,
    ):
        raise TypeError(
            "ground_truth must be a torch.Tensor."
        )

    ###########################################################################
    # Tensor dimensionality
    ###########################################################################

    if trajectories.ndim != 6:
        raise ValueError(
            "trajectories must have shape "
            "(B,N,H,K,T,2). "
            f"Got {tuple(trajectories.shape)}."
        )

    if ground_truth.ndim != 4:
        raise ValueError(
            "ground_truth must have shape "
            "(B,N,T,2). "
            f"Got {tuple(ground_truth.shape)}."
        )

    ###########################################################################
    # Floating-point requirement
    ###########################################################################

    if not torch.is_floating_point(
        trajectories,
    ):
        raise TypeError(
            "trajectories must contain floating-point values."
        )

    if not torch.is_floating_point(
        ground_truth,
    ):
        raise TypeError(
            "ground_truth must contain floating-point values."
        )

    ###########################################################################
    # Device compatibility
    ###########################################################################

    if trajectories.device != ground_truth.device:
        raise ValueError(
            "trajectories and ground_truth must be on the same device. "
            f"Got {trajectories.device} and "
            f"{ground_truth.device}."
        )

    ###########################################################################
    # Dimension extraction
    ###########################################################################

    batch_size = trajectories.shape[0]
    num_agents = trajectories.shape[1]
    history_steps = trajectories.shape[2]
    num_modes = trajectories.shape[3]
    prediction_steps = trajectories.shape[4]
    coordinate_dim = trajectories.shape[5]

    ###########################################################################
    # Non-empty dimensions
    ###########################################################################

    if batch_size <= 0:
        raise ValueError(
            "Batch dimension B must be positive."
        )

    if num_agents <= 0:
        raise ValueError(
            "Agent dimension N must be positive."
        )

    if history_steps <= 0:
        raise ValueError(
            "Historical dimension H must be positive."
        )

    if num_modes <= 0:
        raise ValueError(
            "Prediction-mode dimension K must be positive."
        )

    if prediction_steps <= 0:
        raise ValueError(
            "Prediction horizon T must be positive."
        )

    ###########################################################################
    # Coordinate dimension
    ###########################################################################

    if coordinate_dim != 2:
        raise ValueError(
            "Trajectory coordinate dimension must be 2. "
            f"Got {coordinate_dim}."
        )

    ###########################################################################
    # Ground-truth dimensions
    ###########################################################################

    if ground_truth.shape[0] != batch_size:
        raise ValueError(
            "Batch dimension mismatch between "
            "trajectories and ground_truth: "
            f"{batch_size} != {ground_truth.shape[0]}."
        )

    if ground_truth.shape[1] != num_agents:
        raise ValueError(
            "Agent dimension mismatch between "
            "trajectories and ground_truth: "
            f"{num_agents} != {ground_truth.shape[1]}."
        )

    if ground_truth.shape[2] != prediction_steps:
        raise ValueError(
            "Prediction horizon mismatch between "
            "trajectories and ground_truth: "
            f"{prediction_steps} != {ground_truth.shape[2]}."
        )

    if ground_truth.shape[3] != 2:
        raise ValueError(
            "Ground-truth coordinate dimension must be 2. "
            f"Got {ground_truth.shape[3]}."
        )

    ###########################################################################
    # Numerical validation
    ###########################################################################

    if not torch.isfinite(
        trajectories,
    ).all():
        raise ValueError(
            "trajectories contains NaN or infinite values."
        )

    if not torch.isfinite(
        ground_truth,
    ).all():
        raise ValueError(
            "ground_truth contains NaN or infinite values."
        )


###############################################################################
# Winner-Takes-All target
###############################################################################


def best_mode_from_endpoint(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute the DSTNet Winner-Takes-All target.

    Implements Eq. (28):

        k_best =
            argmin_k
            ||Y^(0)_{n,t,k,T} - G_{n,t,T}||_2

    Parameters
    ----------
    trajectories
        Coarse multimodal predictions.

        Shape:

            (B, N, H, K, T, 2)

    ground_truth
        Future ground truth.

        Shape:

            (B, N, T, 2)

    Returns
    -------
    endpoint_error
        Euclidean endpoint distance for every prediction mode.

        Shape:

            (B, N, H, K)

    best_mode
        Winning mode index for every agent and historical
        prediction timestep.

        Shape:

            (B, N, H)

    Notes
    -----
    The returned ``best_mode`` is an integer tensor and is
    intended for target construction rather than gradient
    propagation.
    """

    ###########################################################################
    # Validate inputs
    ###########################################################################

    validate_trajectory_shapes(
        trajectories,
        ground_truth,
    )

    ###########################################################################
    # Final predicted point
    #
    # trajectories:
    #
    #     (B,N,H,K,T,2)
    #
    # selecting T-1 gives:
    #
    #     (B,N,H,K,2)
    ###########################################################################

    predicted_endpoint = trajectories[
        ...,
        -1,
        :,
    ]

    ###########################################################################
    # Final ground-truth point
    #
    # ground_truth:
    #
    #     (B,N,T,2)
    #
    # selecting T-1 gives:
    #
    #     (B,N,2)
    ###########################################################################

    ground_truth_endpoint = ground_truth[
        ...,
        -1,
        :,
    ]

    ###########################################################################
    # Broadcast ground truth over H and K
    #
    # Before:
    #
    #     (B,N,2)
    #
    # After:
    #
    #     (B,N,1,1,2)
    #
    # which broadcasts against:
    #
    #     (B,N,H,K,2)
    ###########################################################################

    ground_truth_endpoint = (
        ground_truth_endpoint
        .unsqueeze(2)
        .unsqueeze(2)
    )

    ###########################################################################
    # Euclidean endpoint displacement
    #
    #     ||Y_endpoint - G_endpoint||_2
    #
    # Result:
    #
    #     (B,N,H,K)
    ###########################################################################

    endpoint_error = torch.linalg.vector_norm(
        predicted_endpoint
        - ground_truth_endpoint,
        dim=-1,
    )

    ###########################################################################
    # Numerical validation
    ###########################################################################

    if not torch.isfinite(
        endpoint_error,
    ).all():
        raise RuntimeError(
            "Endpoint error contains NaN or infinite values."
        )

    ###########################################################################
    # Winner-Takes-All
    #
    # Select the mode with minimum endpoint error:
    #
    #     (B,N,H,K)
    #             ↓ min over K
    #     (B,N,H)
    ###########################################################################

    best_error, best_mode = endpoint_error.min(
        dim=-1,
    )

    ###########################################################################
    # Validate returned tensors
    ###########################################################################

    if not torch.isfinite(
        best_error,
    ).all():
        raise RuntimeError(
            "Best endpoint error contains NaN or infinite values."
        )

    expected_best_mode_shape = (
        trajectories.shape[0],
        trajectories.shape[1],
        trajectories.shape[2],
    )

    if tuple(best_mode.shape) != expected_best_mode_shape:
        raise RuntimeError(
            "Unexpected best_mode shape: "
            f"expected {expected_best_mode_shape}, "
            f"got {tuple(best_mode.shape)}."
        )

    if best_mode.dtype != torch.int64:
        raise RuntimeError(
            "best_mode must have dtype torch.int64. "
            f"Got {best_mode.dtype}."
        )

    return (
        best_error,
        best_mode,
    )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "validate_trajectory_shapes",
    "best_mode_from_endpoint",
]
