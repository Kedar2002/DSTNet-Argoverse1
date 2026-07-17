"""
evaluation.visualization

Trajectory visualization utilities for DSTNet.

Supports visualization of

- Lane centerlines
- Observed trajectories
- Ground-truth futures
- Predicted trajectories
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.model_types import (
    Prediction,
    RefinedPrediction,
)

def _to_numpy(
    x,
) -> np.ndarray:

    if isinstance(
        x,
        torch.Tensor,
    ):

        return x.detach().cpu().numpy()

    return np.asarray(x)

def visualize_prediction(
    *,
    lane_centerlines,
    observed,
    ground_truth,
    prediction: Prediction | RefinedPrediction,
    save_path: str | Path | None = None,
    show: bool = True,
    title: str = "DSTNet Prediction",
) -> None:
    """
    Visualize one scene.

    Parameters
    ----------
    lane_centerlines
        (L,P,2)

    observed
        (N,Tobs,2)

    ground_truth
        (N,Tpred,2)

    prediction
        Prediction or RefinedPrediction
    """

    plt.figure(
        figsize=(8,8),
    )

    lane_centerlines = _to_numpy(
        lane_centerlines,
    )

    for lane in lane_centerlines:

        plt.plot(

            lane[:,0],

            lane[:,1],

            color="lightgray",

            linewidth=1,

            zorder=1,

        )

    observed = _to_numpy(
        observed,
    )

    for traj in observed:

        plt.plot(

            traj[:,0],

            traj[:,1],

            color="tab:blue",

            linewidth=2,

            zorder=3,

        )

    ground_truth = _to_numpy(
        ground_truth,
    )

    for traj in ground_truth:

        plt.plot(

            traj[:,0],

            traj[:,1],

            "--",

            color="tab:green",

            linewidth=2,

            zorder=4,

        )

    predictions = _to_numpy(
        prediction.trajectories,
    )

    if predictions.ndim == 5:

        predictions = predictions[0]

    for agent in predictions:

        for mode in agent:

            plt.plot(

                mode[:,0],

                mode[:,1],

                color="tab:red",

                alpha=0.35,

                linewidth=1.5,

                zorder=2,

            )

    plt.title(
        title,
    )

    plt.xlabel(
        "x (m)",
    )

    plt.ylabel(
        "y (m)",
    )

    plt.axis(
        "equal",
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    if save_path is not None:

        save_path = Path(
            save_path,
        )

        save_path.parent.mkdir(

            parents=True,

            exist_ok=True,

        )

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches="tight",

        )

    if show:

        plt.show()

    else:

        plt.close()

def save_visualization(
    **kwargs,
) -> None:
    """
    Save without displaying.
    """

    visualize_prediction(

        show=False,

        **kwargs,

    )


