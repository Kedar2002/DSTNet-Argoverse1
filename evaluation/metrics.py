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
from torch import Tensor

from models.model_types import Prediction, RefinedPrediction

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
    prediction: Prediction | RefinedPrediction,
    target: Tensor,
    agent_mask: Tensor | None = None,
) -> torch.Tensor:
    """
    Compute minADE over K for each selected agent's latest history state.
    """

    trajectories, target = _select_evaluation_agents(
        prediction,
        target,
        agent_mask,
    )

    ade = average_displacement_error(
        trajectories,
        target.unsqueeze(1),
    )

    return ade.min(dim=-1).values.mean()

def min_fde(
    prediction: Prediction | RefinedPrediction,
    target: Tensor,
    agent_mask: Tensor | None = None,
) -> torch.Tensor:

    trajectories, target = _select_evaluation_agents(
        prediction,
        target,
        agent_mask,
    )

    fde = final_displacement_error(
        trajectories,
        target.unsqueeze(1),
    )

    return fde.min(
        dim=-1,
    ).values.mean()

def miss_rate(
    prediction: Prediction | RefinedPrediction,
    target: Tensor,
    threshold: float = 2.0,
    agent_mask: Tensor | None = None,
) -> torch.Tensor:
    """
    Argoverse miss rate.

    A prediction is a miss if

        minFDE > threshold
    """

    trajectories, target = _select_evaluation_agents(
        prediction,
        target,
        agent_mask,
    )

    fde = final_displacement_error(
        trajectories,
        target.unsqueeze(1),
    )

    best = fde.min(
        dim=-1,
    ).values

    miss = best > threshold

    return miss.float().mean()

def _select_evaluation_agents(
    prediction: Prediction | RefinedPrediction,
    target: Tensor,
    agent_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Select the latest forecast and only agents with valid labels."""

    trajectories = prediction.trajectories

    if trajectories.ndim == 6:
        # The model retains predictions for every observed state. Forecast
        # metrics use the final observed state, which is the actual forecast
        # origin for this scene.
        trajectories = trajectories[:, :, -1]
    elif trajectories.ndim != 5:
        raise ValueError(
            "Predictions must have shape (B,N,H,K,T,2) or (B,N,K,T,2)."
        )

    if target.ndim != 4 or target.shape[-1] != 2:
        raise ValueError(
            "target must have shape (B,N,T,2)."
        )

    if trajectories.shape[:2] != target.shape[:2]:
        raise ValueError(
            "Prediction and target batch/agent dimensions do not match."
        )

    if trajectories.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            "Prediction and target horizon/coordinate dimensions do not match."
        )

    if agent_mask is None:
        agent_mask = torch.ones(
            target.shape[:2],
            dtype=torch.bool,
            device=target.device,
        )
    else:
        if agent_mask.shape != target.shape[:2]:
            raise ValueError(
                "agent_mask must have shape (B,N)."
            )
        agent_mask = agent_mask.to(
            device=target.device,
            dtype=torch.bool,
        )

    if not bool(agent_mask.any()):
        raise ValueError(
            "No valid agents were selected for evaluation."
        )

    return (
        trajectories[agent_mask].float(),
        target[agent_mask].float(),
    )


def compute_metrics(
    prediction: Prediction | RefinedPrediction,
    target: Tensor,
    agent_mask: Tensor | None = None,
) -> dict[str, float]:
    """
    Compute evaluation metrics for one batch.
    """

    trajectories, target = _select_evaluation_agents(
        prediction,
        target,
        agent_mask,
    )

    target_modes = target.unsqueeze(1)
    ade_by_mode = average_displacement_error(
        trajectories,
        target_modes,
    )
    fde_by_mode = final_displacement_error(
        trajectories,
        target_modes,
    )

    ade = ade_by_mode.min(dim=-1).values.mean()
    best_fde = fde_by_mode.min(dim=-1).values
    fde = best_fde.mean()
    mr = (best_fde > 2.0).float().mean()

    ade_value, fde_value, mr_value = torch.stack(
        (ade, fde, mr)
    ).detach().cpu().tolist()

    return {

        "minADE": float(ade_value),

        "minFDE": float(fde_value),

        "MissRate": float(mr_value),

    }

