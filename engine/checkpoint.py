"""
engine.checkpoint

Checkpoint utilities for DSTNet.

Provides saving and loading of

    • Model
    • Optimizer
    • Scheduler
    • Epoch
    • Global step
    • Metrics
    • Configuration

This module enables seamless training resumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

###############################################################################
# Save
###############################################################################


def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save a complete training checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "epoch": epoch,

        "global_step": global_step,

        "model": model.state_dict(),

        "optimizer": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),

        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "metrics": metrics or {},

        "config": config or {},

    }

    torch.save(
        checkpoint,
        path,
    )

###############################################################################
# Load
###############################################################################


def load_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load checkpoint.

    Returns
    -------
    Dictionary containing

        epoch
        global_step
        metrics
        config
    """

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model"],
    )

    if (
        optimizer is not None
        and checkpoint["optimizer"] is not None
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer"],
        )

    if (
        scheduler is not None
        and checkpoint["scheduler"] is not None
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler"],
        )

    return {

        "epoch": checkpoint.get(
            "epoch",
            0,
        ),

        "global_step": checkpoint.get(
            "global_step",
            0,
        ),

        "metrics": checkpoint.get(
            "metrics",
            {},
        ),

        "config": checkpoint.get(
            "config",
            {},
        ),
    }

###############################################################################
# Latest Checkpoint
###############################################################################


def latest_checkpoint(
    checkpoint_dir: str | Path,
) -> Path | None:
    """
    Return newest checkpoint.
    """

    checkpoint_dir = Path(
        checkpoint_dir,
    )

    checkpoints = sorted(
        checkpoint_dir.glob("*.pth"),
    )

    if not checkpoints:

        return None

    return checkpoints[-1]

###############################################################################
# Best Checkpoint
###############################################################################


def save_best_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    global_step: int,
    metric: float,
    best_metric: float,
    maximize: bool = False,
) -> float:
    """
    Save best-performing model.
    """

    improved = (
        metric > best_metric
        if maximize
        else metric < best_metric
    )

    if improved:

        save_checkpoint(

            path=Path(checkpoint_dir)
            / "best_model.pth",

            model=model,

            optimizer=optimizer,

            scheduler=scheduler,

            epoch=epoch,

            global_step=global_step,

            metrics={
                "best_metric": metric,
            },
        )

        return metric

    return best_metric

###############################################################################
# Resume
###############################################################################


def resume_training(
    *,
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
):
    """
    Resume from latest checkpoint.

    Returns

        state dict

    or

        None
    """

    checkpoint = latest_checkpoint(
        checkpoint_dir,
    )

    if checkpoint is None:

        return None

    return load_checkpoint(

        path=checkpoint,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

    )

###############################################################################
# Utility
###############################################################################


def checkpoint_summary(
    state: dict[str, Any],
) -> str:
    """
    Pretty checkpoint summary.
    """

    return (
        f"Epoch      : {state['epoch']}\n"
        f"GlobalStep : {state['global_step']}\n"
        f"Metrics    : {state['metrics']}"
    )

