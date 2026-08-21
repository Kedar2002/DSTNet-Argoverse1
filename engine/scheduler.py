"""
engine.scheduler

Learning-rate scheduler factory for DSTNet.

Supported schedulers
--------------------
- Constant
- StepLR
- MultiStepLR
- CosineAnnealingLR
- Warmup + Cosine

The scheduler is stepped once per optimizer update by TrainStep.
Therefore ``total_steps`` should normally represent the total number
of optimizer updates, not the number of epochs.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LRScheduler,
    MultiStepLR,
    StepLR,
)


###############################################################################
# Warmup + cosine
###############################################################################


def warmup_cosine_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """

    if warmup_steps < 0:
        raise ValueError(
            "warmup_steps must be non-negative."
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive."
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot exceed total_steps."
        )

    ###########################################################################
    # No warmup
    ###########################################################################

    if warmup_steps == 0:

        progress = (
            float(current_step)
            / float(max(1, total_steps))
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

    ###########################################################################
    # Warmup
    ###########################################################################

    if current_step < warmup_steps:

        return float(current_step) / float(
            max(1, warmup_steps)
        )

    ###########################################################################
    # Cosine decay
    ###########################################################################

    progress = (
        current_step - warmup_steps
    ) / float(
        max(
            1,
            total_steps - warmup_steps,
        )
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )


###############################################################################
# Scheduler factory
###############################################################################


def build_scheduler(
    optimizer: Optimizer,
    scheduler: str = "warmup_cosine",
    *,
    total_steps: int,
    warmup_steps: int = 1000,
    step_size: int = 10,
    gamma: float = 0.1,
    milestones: list[int] | None = None,
) -> LRScheduler:
    """
    Build a learning-rate scheduler.

    Parameters
    ----------
    optimizer:
        Optimizer whose learning rate is scheduled.

    scheduler:
        One of:

            constant
            step
            multistep
            cosine
            warmup_cosine

    total_steps:
        Total optimizer-update count used by cosine schedules.

    warmup_steps:
        Number of optimizer updates used for linear warmup.

    step_size:
        StepLR decay interval.

    gamma:
        Multiplicative decay factor.

    milestones:
        MultiStepLR milestones.
    """

    if not isinstance(
        scheduler,
        str,
    ):
        raise TypeError(
            "scheduler must be a string."
        )

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be positive."
        )

    scheduler_name = (
        scheduler.strip().lower()
    )

    ###########################################################################
    # Constant
    ###########################################################################

    if scheduler_name == "constant":

        return LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )

    ###########################################################################
    # StepLR
    ###########################################################################

    if scheduler_name == "step":

        if step_size <= 0:
            raise ValueError(
                "step_size must be positive."
            )

        if gamma <= 0.0:
            raise ValueError(
                "gamma must be positive."
            )

        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    ###########################################################################
    # MultiStepLR
    ###########################################################################

    if scheduler_name == "multistep":

        if gamma <= 0.0:
            raise ValueError(
                "gamma must be positive."
            )

        if milestones is None:
            milestones = []

        if any(
            milestone < 0
            for milestone in milestones
        ):
            raise ValueError(
                "All milestones must be non-negative."
            )

        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    ###########################################################################
    # Cosine
    ###########################################################################

    if scheduler_name == "cosine":

        return CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                total_steps,
            ),
        )

    ###########################################################################
    # Warmup + cosine
    ###########################################################################

    if scheduler_name == "warmup_cosine":

        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be non-negative."
            )

        if warmup_steps > total_steps:
            raise ValueError(
                "warmup_steps cannot exceed total_steps."
            )

        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                warmup_cosine_lambda(
                    step,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                )
            ),
        )

    raise ValueError(
        f"Unknown scheduler '{scheduler}'. "
        "Supported values are: "
        "constant, step, multistep, cosine, warmup_cosine."
    )


###############################################################################
# Utility
###############################################################################


def scheduler_summary(
    scheduler: LRScheduler,
) -> str:
    """
    Return a concise scheduler description.
    """

    return (
        f"{scheduler.__class__.__name__}"
    )


__all__ = [
    "warmup_cosine_lambda",
    "build_scheduler",
    "scheduler_summary",
]
