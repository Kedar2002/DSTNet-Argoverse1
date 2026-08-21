"""
models.refinement.anchor_selector

Anchor Selection for DSTNet
===========================

DSTNet
------

Section III-F — Anchor-Based Trajectory Refinement.

The refinement stage selects two feature anchor points from each
coarse trajectory:

    1. midpoint
    2. endpoint

These anchors are subsequently used for anchor-centric contextual
encoding and two-segment trajectory refinement.

Tensor contract
---------------

Input:

    trajectories
        (B,N,H,K,T,2)

Output:

    AnchorSelection

        anchors
            (B,N,H,K,2,2)

        radii
            (2,)

where:

    anchors[..., 0, :] = midpoint
    anchors[..., 1, :] = endpoint

The historical dimension H and multimodal dimension K are both
preserved.

The retrieval radius is represented as a two-element linearly
decaying schedule. The numerical values are configurable because
the supplied paper text does not establish a unique numerical
pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


###############################################################################
# Anchor Selection Result
###############################################################################


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    """
    Result of anchor selection.

    Parameters
    ----------
    anchors
        Midpoint and endpoint coordinates.

        Shape:

            (B,N,H,K,2,2)

    radii
        Retrieval radius for each anchor.

        Shape:

            (2,)

    midpoint_index
        Zero-based midpoint trajectory index.

    endpoint_index
        Zero-based endpoint trajectory index.
    """

    anchors: Tensor
    radii: Tensor
    midpoint_index: int
    endpoint_index: int


###############################################################################
# Anchor Selector
###############################################################################


class AnchorSelector(nn.Module):
    """
    Select midpoint and endpoint anchors from coarse trajectories.

    Input
    -----

    trajectories:

        (B,N,H,K,T,2)

    Output
    ------

    AnchorSelection:

        anchors:
            (B,N,H,K,2,2)

        radii:
            (2,)

    Anchor ordering
    ---------------

    Anchor index 0:
        midpoint

    Anchor index 1:
        endpoint
    """

    NUM_ANCHORS = 2

    MIDPOINT = 0
    ENDPOINT = 1

    def __init__(
        self,
        radius_start: float = 30.0,
        radius_end: float = 10.0,
    ) -> None:

        super().__init__()

        if radius_start <= 0.0:
            raise ValueError(
                "radius_start must be positive."
            )

        if radius_end <= 0.0:
            raise ValueError(
                "radius_end must be positive."
            )

        if radius_end > radius_start:
            raise ValueError(
                "radius_end must not exceed radius_start."
            )

        self.radius_start = float(
            radius_start
        )

        self.radius_end = float(
            radius_end
        )

    ###########################################################################
    # Midpoint index
    ###########################################################################

    @staticmethod
    def _midpoint_index(
        prediction_steps: int,
    ) -> int:
        """
        Return the zero-based midpoint index.

        For T=30:

            midpoint = 14

        Endpoint:

            index = 29
        """

        if prediction_steps < 2:
            raise ValueError(
                "prediction_steps must be at least 2."
            )

        return (
            prediction_steps - 1
        ) // 2

    ###########################################################################
    # Radius schedule
    ###########################################################################

    def _build_radii(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Construct the two-anchor linearly decaying radius schedule.

        Returns
        -------
        Tensor
            Shape (2,)

        Example with default configuration:

            tensor([30., 10.])
        """

        return torch.linspace(
            self.radius_start,
            self.radius_end,
            steps=self.NUM_ANCHORS,
            device=device,
            dtype=dtype,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        trajectories: Tensor,
    ) -> AnchorSelection:
        """
        Select midpoint and endpoint anchors.

        Parameters
        ----------
        trajectories
            Coarse multimodal trajectories.

            Shape:

                (B,N,H,K,T,2)

        Returns
        -------
        AnchorSelection
        """

        #######################################################################
        # Type validation
        #######################################################################

        if not isinstance(
            trajectories,
            torch.Tensor,
        ):
            raise TypeError(
                "trajectories must be a torch.Tensor."
            )

        #######################################################################
        # Shape validation
        #######################################################################

        if trajectories.ndim != 6:
            raise ValueError(
                "Expected trajectories with shape "
                "(B,N,H,K,T,2), "
                f"got {tuple(trajectories.shape)}."
            )

        if trajectories.shape[-1] != 2:
            raise ValueError(
                "Trajectory coordinate dimension must equal 2."
            )

        B, N, H, K, T, _ = (
            trajectories.shape
        )

        if T < 2:
            raise ValueError(
                "Trajectory must contain at least two "
                "prediction steps."
            )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.is_floating_point(
            trajectories,
        ):
            raise TypeError(
                "trajectories must contain floating-point values."
            )

        if not torch.isfinite(
            trajectories,
        ).all():
            raise ValueError(
                "trajectories contains NaN or infinite values."
            )

        #######################################################################
        # Anchor indices
        #######################################################################

        midpoint_index = (
            self._midpoint_index(T)
        )

        endpoint_index = T - 1

        #######################################################################
        # Select midpoint
        #######################################################################

        midpoint = trajectories[
            ...,
            midpoint_index,
            :,
        ]

        #######################################################################
        # Select endpoint
        #######################################################################

        endpoint = trajectories[
            ...,
            endpoint_index,
            :,
        ]

        #######################################################################
        # Stack anchors
        #
        # Result:
        #
        #     (B,N,H,K,2,2)
        #
        # where:
        #
        #     [...,0,:] -> midpoint
        #     [...,1,:] -> endpoint
        #######################################################################

        anchors = torch.stack(
            (
                midpoint,
                endpoint,
            ),
            dim=-2,
        )

        #######################################################################
        # Radius schedule
        #######################################################################

        radii = self._build_radii(
            device=trajectories.device,
            dtype=trajectories.dtype,
        )

        #######################################################################
        # Defensive output validation
        #######################################################################

        expected_shape = (
            B,
            N,
            H,
            K,
            self.NUM_ANCHORS,
            2,
        )

        if tuple(
            anchors.shape
        ) != expected_shape:
            raise RuntimeError(
                "Anchor construction produced an unexpected "
                f"shape: expected {expected_shape}, "
                f"got {tuple(anchors.shape)}."
            )

        return AnchorSelection(
            anchors=anchors,
            radii=radii,
            midpoint_index=midpoint_index,
            endpoint_index=endpoint_index,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"radius_start={self.radius_start}, "
            f"radius_end={self.radius_end}, "
            f"num_anchors={self.NUM_ANCHORS}"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "AnchorSelection",
    "AnchorSelector",
]
