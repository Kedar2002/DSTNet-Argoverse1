"""
models.refinement.anchor_selector

Adaptive Anchor Selector for DSTNet.

This module selects the anchor trajectory for each prediction
mode from the coarse trajectory proposals.

The selected anchors are passed to the refinement stage.

Paper
-----
Adaptive Anchor-based Refinement (AAR)
"""

from __future__ import annotations

import torch
from torch import nn


class AnchorSelector(nn.Module):
    """
    Adaptive Anchor Selector.

    Input
    -----
        trajectories
            (B,N,K,T,2)

        scores
            (B,N,K)

    Output
    ------
        anchors
            (B,N,K,T,2)
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

    ###########################################################################

    def forward(
        self,
        trajectories: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:

        if trajectories.ndim != 5:
            raise ValueError(
                "Expected trajectory shape (B,N,K,T,2)."
            )

        if scores.ndim != 3:
            raise ValueError(
                "Expected score shape (B,N,K)."
            )

        if trajectories.shape[:3] != scores.shape:
            raise ValueError(
                "Trajectory and score dimensions mismatch."
            )

        #######################################################################
        # Current implementation
        #
        # Use coarse trajectories as adaptive anchors.
        #
        # The DSTNet paper does not specify a separate anchor generation
        # mechanism. Therefore, the decoder outputs are treated as the
        # initial anchors for the refinement module.
        #######################################################################

        return trajectories

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return "AnchorSelector()"
