"""
models.refinement.context_encoder

Anchor-Centric Context Encoder for DSTNet
==========================================

DSTNet
------

Section III-F — Anchor-Based Trajectory Refinement.

The refinement module selects feature anchor points from the
coarse trajectories and retrieves contextual information around
those anchors.

This implementation preserves the explicit:

    (B,N,H,K)

representation used throughout the current DSTNet pipeline.

Tensor contract
---------------

Scene features:

    Z_STM
        (B,N,H,K,D)

Anchors:

    (B,N,H,K,A,2)

where:

    A = 2

        anchor 0 -> midpoint
        anchor 1 -> endpoint

Radii:

    (A,)

Output:

    Z_A
        (B,N,H,K,A,D)

The two-anchor dimension is preserved because the two trajectory
segments are refined independently.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.layers.mlp import MLP


###############################################################################
# Context Encoder
###############################################################################


class ContextEncoder(nn.Module):
    """
    Anchor-centric contextual feature encoder.

    The encoder performs anchor-conditioned attention over the
    historical scene representation.

    Input
    -----

    scene_features:

        (B,N,H,K,D)

    anchors:

        (B,N,H,K,A,2)

    radii:

        (A,)

    Output
    ------

    context:

        (B,N,H,K,A,D)

    Notes
    -----

    The supplied DSTNet paper describes anchor-centric contextual
    encoding but does not provide a complete standalone equation
    specifying every internal projection.

    Therefore this module implements the contextual attention
    mechanism as an explicit, modular component without claiming
    that the exact projection architecture is reproduced verbatim
    from the paper.
    """

    NUM_ANCHORS = 2

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy "
                "0 <= dropout < 1."
            )

        self.hidden_dim = hidden_dim

        #######################################################################
        # Scene normalization
        #######################################################################

        self.scene_norm = nn.LayerNorm(
            hidden_dim,
        )

        #######################################################################
        # Anchor geometry encoder
        #
        # Geometry:
        #
        #     [x, y, radius]
        #
        # -> D
        #######################################################################

        self.anchor_geometry_encoder = MLP(
            input_dim=3,
            hidden_dims=[hidden_dim],
            output_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Q / K / V projections
        #######################################################################

        self.query_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.key_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.value_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        #######################################################################
        # Context projection
        #######################################################################

        self.context_projection = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        #######################################################################
        # Output normalization
        #######################################################################

        self.output_norm = nn.LayerNorm(
            hidden_dim,
        )

        #######################################################################
        # Dropout
        #######################################################################

        self.attention_dropout = nn.Dropout(
            dropout,
        )

    ###########################################################################
    # Input validation
    ###########################################################################

    def _validate_inputs(
        self,
        scene_features: Tensor,
        anchors: Tensor,
        radii: Tensor,
    ) -> None:
        """
        Validate all ContextEncoder inputs.
        """

        if not isinstance(
            scene_features,
            torch.Tensor,
        ):
            raise TypeError(
                "scene_features must be a torch.Tensor."
            )

        if not isinstance(
            anchors,
            torch.Tensor,
        ):
            raise TypeError(
                "anchors must be a torch.Tensor."
            )

        if not isinstance(
            radii,
            torch.Tensor,
        ):
            raise TypeError(
                "radii must be a torch.Tensor."
            )

        #######################################################################
        # Scene features
        #######################################################################

        if scene_features.ndim != 5:
            raise ValueError(
                "scene_features must have shape "
                "(B,N,H,K,D)."
            )

        #######################################################################
        # Anchors
        #######################################################################

        if anchors.ndim != 6:
            raise ValueError(
                "anchors must have shape "
                "(B,N,H,K,A,2)."
            )

        if anchors.shape[-2] != (
            self.NUM_ANCHORS
        ):
            raise ValueError(
                "anchors must contain exactly "
                f"{self.NUM_ANCHORS} anchors."
            )

        if anchors.shape[-1] != 2:
            raise ValueError(
                "Anchor coordinates must be 2-D."
            )

        #######################################################################
        # Radii
        #######################################################################

        if radii.ndim != 1:
            raise ValueError(
                "radii must have shape (A,)."
            )

        if radii.shape[0] != (
            self.NUM_ANCHORS
        ):
            raise ValueError(
                "radii must contain exactly "
                f"{self.NUM_ANCHORS} values."
            )

        #######################################################################
        # Shared dimensions
        #######################################################################

        B, N, H, K, D = (
            scene_features.shape
        )

        if D != self.hidden_dim:
            raise ValueError(
                "Scene feature dimension mismatch: "
                f"expected {self.hidden_dim}, "
                f"got {D}."
            )

        if tuple(
            anchors.shape[:4]
        ) != (
            B,
            N,
            H,
            K,
        ):
            raise ValueError(
                "scene_features and anchors must agree on "
                "(B,N,H,K)."
            )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.is_floating_point(
            scene_features,
        ):
            raise TypeError(
                "scene_features must contain floating-point values."
            )

        if not torch.is_floating_point(
            anchors,
        ):
            raise TypeError(
                "anchors must contain floating-point values."
            )

        if not torch.isfinite(
            scene_features,
        ).all():
            raise ValueError(
                "scene_features contains NaN or infinite values."
            )

        if not torch.isfinite(
            anchors,
        ).all():
            raise ValueError(
                "anchors contains NaN or infinite values."
            )

        if not torch.isfinite(
            radii,
        ).all():
            raise ValueError(
                "radii contains NaN or infinite values."
            )

        if torch.any(
            radii <= 0.0,
        ):
            raise ValueError(
                "All retrieval radii must be positive."
            )

    ###########################################################################
    # Anchor Geometry
    ###########################################################################

    def _encode_anchor_geometry(
        self,
        anchors: Tensor,
        radii: Tensor,
    ) -> Tensor:
        """
        Encode anchor coordinates and retrieval radius.

        Parameters
        ----------
        anchors
            (B,N,H,K,A,2)

        radii
            (A,)

        Returns
        -------
        Tensor

            (B,N,H,K,A,D)
        """

        #######################################################################
        # Convert radius to anchor device/dtype
        #######################################################################

        radius = radii.to(
            device=anchors.device,
            dtype=anchors.dtype,
        )

        #######################################################################
        # Expand:
        #
        # (A,)
        #
        # ->
        #
        # (B,N,H,K,A,1)
        #######################################################################

        radius = radius.view(
            1,
            1,
            1,
            1,
            self.NUM_ANCHORS,
            1,
        )

        radius = radius.expand(
            anchors.shape[0],
            anchors.shape[1],
            anchors.shape[2],
            anchors.shape[3],
            self.NUM_ANCHORS,
            1,
        )

        #######################################################################
        # Geometry:
        #
        # [x, y, radius]
        #######################################################################

        geometry = torch.cat(
            (
                anchors,
                radius,
            ),
            dim=-1,
        )

        #######################################################################
        # Encode
        #######################################################################

        encoded = (
            self.anchor_geometry_encoder(
                geometry,
            )
        )

        return encoded

    ###########################################################################
    # Anchor-Conditioned Temporal Attention
    ###########################################################################

    def _encode_context(
        self,
        scene_features: Tensor,
        anchor_features: Tensor,
    ) -> Tensor:
        """
        Perform anchor-conditioned attention over H historical
        scene states.

        Parameters
        ----------
        scene_features
            (B,N,H,K,D)

        anchor_features
            (B,N,H,K,A,D)

        Returns
        -------
        Tensor

            (B,N,H,K,A,D)
        """

        B, N, H, K, D = (
            scene_features.shape
        )

        #######################################################################
        # Normalize scene representation
        #######################################################################

        scene = self.scene_norm(
            scene_features,
        )

        #######################################################################
        # Keys and values
        #######################################################################

        keys = self.key_projection(
            scene,
        )

        values = self.value_projection(
            scene,
        )

        #######################################################################
        # Anchor queries
        #######################################################################

        queries = self.query_projection(
            anchor_features,
        )

        #######################################################################
        # Attention score
        #
        # Query:
        #
        #     (B,N,Hq,K,A,D)
        #
        # Key:
        #
        #     (B,N,Hk,K,D)
        #
        # We calculate:
        #
        #     score(hq,hk)
        #
        # for every query historical state and every historical
        # context state.
        #
        # Result:
        #
        #     (B,N,Hq,K,A,Hk)
        #######################################################################

        scores = torch.einsum(
            "bnhkad,bnjkd->bnhkaj",
            queries,
            keys,
        )

        scores = scores / math.sqrt(
            float(D),
        )

        #######################################################################
        # Temporal attention
        #######################################################################

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.attention_dropout(
            attention,
        )

        #######################################################################
        # Aggregate historical context
        #
        # attention:
        #
        #     (B,N,Hq,K,A,Hk)
        #
        # values:
        #
        #     (B,N,Hk,K,D)
        #
        # result:
        #
        #     (B,N,Hq,K,A,D)
        #######################################################################

        context = torch.einsum(
            "bnhkaj,bnjkd->bnhkad",
            attention,
            values,
        )

        #######################################################################
        # Anchor-conditioned residual
        #######################################################################

        context = (
            context
            + anchor_features
        )

        #######################################################################
        # Output projection
        #######################################################################

        context = self.context_projection(
            context,
        )

        #######################################################################
        # Final normalization
        #######################################################################

        context = self.output_norm(
            context,
        )

        return context

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        scene_features: Tensor,
        anchors: Tensor,
        radii: Tensor,
    ) -> Tensor:
        """
        Encode contextual features around the selected anchors.

        Parameters
        ----------
        scene_features
            Z_STM.

            Shape:

                (B,N,H,K,D)

        anchors
            Midpoint and endpoint coordinates.

            Shape:

                (B,N,H,K,2,2)

        radii
            Retrieval radii.

            Shape:

                (2,)

        Returns
        -------
        Tensor

            Anchor-centric contextual representation:

                (B,N,H,K,2,D)
        """

        #######################################################################
        # Validate
        #######################################################################

        self._validate_inputs(
            scene_features,
            anchors,
            radii,
        )

        #######################################################################
        # Encode anchor geometry
        #######################################################################

        anchor_features = (
            self._encode_anchor_geometry(
                anchors,
                radii,
            )
        )

        #######################################################################
        # Anchor-conditioned temporal context
        #######################################################################

        context = self._encode_context(
            scene_features,
            anchor_features,
        )

        #######################################################################
        # Defensive output validation
        #######################################################################

        expected_shape = (
            scene_features.shape[0],
            scene_features.shape[1],
            scene_features.shape[2],
            scene_features.shape[3],
            self.NUM_ANCHORS,
            self.hidden_dim,
        )

        if tuple(
            context.shape
        ) != expected_shape:
            raise RuntimeError(
                "ContextEncoder produced an unexpected "
                f"shape: expected {expected_shape}, "
                f"got {tuple(context.shape)}."
            )

        if not torch.isfinite(
            context,
        ).all():
            raise RuntimeError(
                "ContextEncoder produced NaN or "
                "infinite values."
            )

        return context

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_anchors={self.NUM_ANCHORS}"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "ContextEncoder",
]
