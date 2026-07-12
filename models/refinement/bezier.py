"""
Bezier trajectory utilities for DSTNet.

This implementation follows the refinement strategy described
in the DSTNet paper while reconstructing the omitted Bezier
module using standard cubic Bezier mathematics.

Author: Thesis Reconstruction
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Bernstein Basis
# ---------------------------------------------------------

def bernstein_basis(
    degree: int,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Bernstein basis.

    Args
    ----
    degree : polynomial degree
    t : (...,) values in [0,1]

    Returns
    -------
    (..., degree+1)
    """

    basis = []

    for i in range(degree + 1):

        coeff = math.comb(degree, i)

        b = (
            coeff
            * (t ** i)
            * ((1.0 - t) ** (degree - i))
        )

        basis.append(b)

    return torch.stack(basis, dim=-1)

# ---------------------------------------------------------
# Cubic Basis Matrix
# ---------------------------------------------------------

def cubic_basis(
    steps: int,
    device=None,
):
    """
    Returns

    [T,4]

    containing the cubic Bernstein basis.
    """

    t = torch.linspace(
        0.0,
        1.0,
        steps,
        device=device,
    )

    return bernstein_basis(
        degree=3,
        t=t,
    )

# ---------------------------------------------------------
# Evaluate Cubic Bezier
# ---------------------------------------------------------

def evaluate_cubic_bezier(
    control_points: torch.Tensor,
    num_points: int,
):
    """
    Parameters
    ----------
    control_points

        (...,4,2)

    Returns
    -------
    (...,T,2)
    """

    B = cubic_basis(
        num_points,
        control_points.device,
    )

    return torch.einsum(
        "tc,...cd->...td",
        B,
        control_points,
    )

# ---------------------------------------------------------
# Evaluate Arbitrary Degree
# ---------------------------------------------------------

def evaluate_bezier(
    control_points: torch.Tensor,
    num_points: int,
):
    """
    Generic Bezier evaluator.

    control_points

    (...,K,2)
    """

    degree = control_points.size(-2) - 1

    t = torch.linspace(
        0,
        1,
        num_points,
        device=control_points.device,
    )

    basis = bernstein_basis(
        degree,
        t,
    )

    return torch.einsum(
        "tk,...kd->...td",
        basis,
        control_points,
    )

# ---------------------------------------------------------
# Trajectory -> Control Points
# ---------------------------------------------------------

def trajectory_to_control_points(
    traj: torch.Tensor,
):
    """
    Build cubic Bezier control points from a trajectory.

    Args

        traj (...,T,2)

    Returns

        (...,4,2)
    """

    T = traj.size(-2)

    p0 = traj[..., 0, :]

    p1 = traj[..., T // 3, :]

    p2 = traj[..., 2 * T // 3, :]

    p3 = traj[..., -1, :]

    return torch.stack(
        [
            p0,
            p1,
            p2,
            p3,
        ],
        dim=-2,
    )

# ---------------------------------------------------------
# Smooth Existing Trajectory
# ---------------------------------------------------------

def bezier_smooth(
    traj: torch.Tensor,
):
    """
    Convert trajectory

    -> cubic Bezier

    -> evaluate back.

    Shape

    (...,T,2)
    """

    T = traj.size(-2)

    control = trajectory_to_control_points(
        traj
    )

    smooth = evaluate_cubic_bezier(
        control,
        T,
    )

    return smooth

# ---------------------------------------------------------
# Arc Length
# ---------------------------------------------------------

def arc_length(
    traj: torch.Tensor,
):
    """
    Approximate trajectory length.
    """

    delta = traj[..., 1:, :] - traj[..., :-1, :]

    dist = delta.norm(dim=-1)

    return dist.sum(dim=-1)

# ---------------------------------------------------------
# Curvature
# ---------------------------------------------------------

def curvature(
    traj: torch.Tensor,
):
    """
    Approximate curvature.

    Useful during refinement scoring.
    """

    d1 = traj[..., 1:, :] - traj[..., :-1, :]

    d2 = d1[..., 1:, :] - d1[..., :-1, :]

    return d2.norm(dim=-1)

# ---------------------------------------------------------
# Learnable Bezier Smoothing Module
# ---------------------------------------------------------

class BezierSmoothingModule(nn.Module):
    """
    Differentiable Bezier smoothing module.

    Input
    -----
    trajectory:
        (..., T, 2)

    feature (optional):
        (..., D)

    Output
    ------
    smoothed trajectory
    """

    def __init__(
        self,
        hidden_dim: int,
        prediction_steps: int,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps

        #
        # Learn small residual corrections
        # to the four control points.
        #
        self.control_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8),
        )

        #
        # confidence gate
        #
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        trajectory: torch.Tensor,
        feature: torch.Tensor | None = None,
    ) -> torch.Tensor:

        """
        trajectory

        [B,K,T,2]

        feature

        [B,K,D]
        """

        control = trajectory_to_control_points(
            trajectory
        )

        #
        # Optional learned correction
        #
        if feature is not None:

            offset = self.control_mlp(feature)

            offset = offset.view(
                *feature.shape[:-1],
                4,
                2,
            )

            gate = self.gate(feature).unsqueeze(-1)

            control = control + gate * offset

        smooth = evaluate_cubic_bezier(
            control,
            self.prediction_steps,
        )

        return smooth

# ---------------------------------------------------------
# Residual Bezier Refinement
# ---------------------------------------------------------

class ResidualBezierRefinement(nn.Module):
    """
    Blend original prediction
    with Bezier reconstruction.

    final =
        alpha * bezier +
        (1-alpha) * original
    """

    def __init__(
        self,
        hidden_dim: int,
        prediction_steps: int,
    ):
        super().__init__()

        self.bezier = BezierSmoothingModule(
            hidden_dim,
            prediction_steps,
        )

        self.alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        trajectory,
        feature,
    ):

        smooth = self.bezier(
            trajectory,
            feature,
        )

        alpha = self.alpha(feature)

        alpha = alpha.unsqueeze(-1)

        return (
            alpha * smooth
            + (1.0 - alpha) * trajectory
        )

# ---------------------------------------------------------
# Batch Bezier
# ---------------------------------------------------------

class BatchBezier(nn.Module):
    """
    Simple wrapper used by the
    refinement module.
    """

    def __init__(
        self,
        prediction_steps,
    ):
        super().__init__()

        self.prediction_steps = prediction_steps

    def forward(
        self,
        control_points,
    ):

        return evaluate_cubic_bezier(
            control_points,
            self.prediction_steps,
        )

# ---------------------------------------------------------
# Smoothness Loss
# ---------------------------------------------------------

def bezier_smoothness_loss(
    trajectory: torch.Tensor,
):
    """
    Penalizes excessive curvature.

    Returns scalar.
    """

    first = trajectory[..., 1:, :] - trajectory[..., :-1, :]

    second = first[..., 1:, :] - first[..., :-1, :]

    return second.pow(2).mean()

def curvature_loss(
    trajectory,
):

    curv = curvature(
        trajectory
    )

    return curv.mean()

def endpoint_loss(
    original,
    smoothed,
):

    start = F.mse_loss(
        smoothed[..., 0, :],
        original[..., 0, :],
    )

    end = F.mse_loss(
        smoothed[..., -1, :],
        original[..., -1, :],
    )

    return start + end

def bezier_regularization(
    original,
    smoothed,
    lambda_curve=0.05,
):

    return (
        endpoint_loss(
            original,
            smoothed,
        )
        + lambda_curve
        * curvature_loss(smoothed)
    )


