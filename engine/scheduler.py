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
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
)

###############################################################################
# Warmup + Cosine
###############################################################################


def warmup_cosine_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
):
    """
    Linear warmup followed by cosine decay.
    """

    ###############################################################
    # Warmup
    ###############################################################

    if current_step < warmup_steps:

        return float(current_step) / float(
            max(1, warmup_steps)
        )

    ###############################################################
    # Cosine
    ###############################################################

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
# Scheduler Factory
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
):
    """
    Build scheduler.
    """

    scheduler = scheduler.lower()

    ###################################################################
    # Constant LR
    ###################################################################

    if scheduler == "constant":

        return LambdaLR(
            optimizer,
            lambda _: 1.0,
        )

    ###################################################################
    # StepLR
    ###################################################################

    if scheduler == "step":

        return StepLR(

            optimizer,

            step_size=step_size,

            gamma=gamma,

        )

    ###################################################################
    # MultiStepLR
    ###################################################################

    if scheduler == "multistep":

        return MultiStepLR(

            optimizer,

            milestones=milestones or [],

            gamma=gamma,

        )

    ###################################################################
    # Cosine Annealing
    ###################################################################

    if scheduler == "cosine":

        return CosineAnnealingLR(

            optimizer,

            T_max=total_steps,

        )

    ###################################################################
    # Warmup + Cosine
    ###################################################################

    if scheduler == "warmup_cosine":

        return LambdaLR(

            optimizer,

            lambda step: warmup_cosine_lambda(

                step,

                warmup_steps=warmup_steps,

                total_steps=total_steps,

            ),

        )

    raise ValueError(
        f"Unknown scheduler '{scheduler}'."
    )

###############################################################################
# Utility
###############################################################################


def scheduler_summary(
    scheduler,
) -> str:

    return (
        f"{scheduler.__class__.__name__}"
    )


