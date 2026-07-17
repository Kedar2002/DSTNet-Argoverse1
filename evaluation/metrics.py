"""
evaluation.metrics

Trajectory forecasting metrics for DSTNet.

Implemented metrics

    ADE
    FDE
    minADE@K
    minFDE@K
    Miss Rate
"""

from __future__ import annotations

import torch

from models.model_types import RefinedPrediction

def average_displacement_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Average Displacement Error.

    prediction
        (...,T,2)

    target
        (...,T,2)
    """

    displacement = torch.norm(

        prediction - target,

        dim=-1,

    )

    return displacement.mean(
        dim=-1,
    )

def final_displacement_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Final Displacement Error.
    """

    return torch.norm(

        prediction[..., -1, :]

        -

        target[..., -1, :],

        dim=-1,

    )

def min_ade(
    prediction: RefinedPrediction,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    prediction

        (B,N,K,T,2)

    target

        (B,N,T,2)
    """

    gt = target.unsqueeze(2)

    gt = gt.expand_as(
        prediction.trajectories,
    )

    ade = average_displacement_error(

        prediction.trajectories,

        gt,

    )

    return ade.min(
        dim=-1,
    ).values.mean()

def min_fde(
    prediction: RefinedPrediction,
    target: torch.Tensor,
) -> torch.Tensor:

    gt = target.unsqueeze(2)

    gt = gt.expand_as(
        prediction.trajectories,
    )

    fde = final_displacement_error(

        prediction.trajectories,

        gt,

    )

    return fde.min(
        dim=-1,
    ).values.mean()

def miss_rate(
    prediction: RefinedPrediction,
    target: torch.Tensor,
    threshold: float = 2.0,
) -> torch.Tensor:
    """
    Argoverse miss rate.

    A prediction is a miss if

        minFDE > threshold
    """

    gt = target.unsqueeze(2)

    gt = gt.expand_as(
        prediction.trajectories,
    )

    fde = final_displacement_error(

        prediction.trajectories,

        gt,

    )

    best = fde.min(
        dim=-1,
    ).values

    miss = best > threshold

    return miss.float().mean()

def compute_metrics(
    prediction: RefinedPrediction,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Compute evaluation metrics for one batch.
    """

    ade = min_ade(
        prediction,
        target,
    )

    fde = min_fde(
        prediction,
        target,
    )

    mr = miss_rate(
        prediction,
        target,
    )

    return {

        "minADE": float(
            ade.cpu(),
        ),

        "minFDE": float(
            fde.cpu(),
        ),

        "MissRate": float(
            mr.cpu(),
        ),

    }

