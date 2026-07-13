"""
engine.optimizer

Optimizer factory for DSTNet.

Creates optimizers with proper parameter grouping for
transformer-style models.

Features
--------
- Adam
- AdamW
- SGD
- Weight decay filtering
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


###############################################################################
# Parameter Grouping
###############################################################################


def build_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Separate parameters into

        decay
        no_decay

    following transformer best practices.
    """

    decay = []

    no_decay = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        ###############################################################
        # No Weight Decay
        ###############################################################

        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "layernorm" in name.lower()
            or parameter.ndim == 1
        ):

            no_decay.append(parameter)

        else:

            decay.append(parameter)

    return [

        {
            "params": decay,
            "weight_decay": weight_decay,
        },

        {
            "params": no_decay,
            "weight_decay": 0.0,
        },

    ]


###############################################################################
# Optimizer Factory
###############################################################################


def build_optimizer(
    model: nn.Module,
    optimizer: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Build optimizer.
    """

    optimizer = optimizer.lower()

    parameter_groups = build_parameter_groups(
        model,
        weight_decay,
    )

    ###################################################################
    # Adam
    ###################################################################

    if optimizer == "adam":

        return torch.optim.Adam(

            parameter_groups,

            lr=learning_rate,

            betas=betas,

        )

    ###################################################################
    # AdamW
    ###################################################################

    if optimizer == "adamw":

        return torch.optim.AdamW(

            parameter_groups,

            lr=learning_rate,

            betas=betas,

        )

    ###################################################################
    # SGD
    ###################################################################

    if optimizer == "sgd":

        return torch.optim.SGD(

            parameter_groups,

            lr=learning_rate,

            momentum=momentum,

        )

    raise ValueError(
        f"Unknown optimizer '{optimizer}'."
    )


###############################################################################
# Utility
###############################################################################


def optimizer_summary(
    optimizer: torch.optim.Optimizer,
) -> str:
    """
    Pretty optimizer summary.
    """

    lines = []

    lines.append(
        optimizer.__class__.__name__
    )

    for i, group in enumerate(
        optimizer.param_groups
    ):

        lines.append(

            f"Group {i}"

            f" | lr={group['lr']}"

            f" | wd={group['weight_decay']}"

            f" | params={len(group['params'])}"

        )

    return "\n".join(lines)


