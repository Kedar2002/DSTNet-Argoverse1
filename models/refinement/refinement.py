"""
models.refinement.refinement

Adaptive Anchor-Based Trajectory Refinement
===========================================

DSTNet refinement pipeline
--------------------------

Coarse trajectory:

    Y^(0)
       |
       v
Anchor selection
       |
       +---- midpoint anchor
       |
       +---- endpoint anchor
       |
       v
Anchor-centric context encoding
       |
       v
Z^A
       |
       v
Two trajectory segments
       |
       +-------------------------------+
       |                               |
       v                               v
Segment 1 refinement            Segment 2 refinement
       |                               |
       +---------------+---------------+
                       |
                       v
                 ΔY^(i)
                       |
                       v
        Y^(i+1) = Y^(i) + ΔY^(i)

The paper describes a two-iteration refinement process.
Each refinement cycle consists of two trajectory-segment
refinements.

Tensor contract
---------------

z_stm
    (B,N,H,K,D)

Prediction.trajectories
    (B,N,H,K,T,2)

anchors
    (B,N,H,K,2,2)

anchor_context
    (B,N,H,K,2,D)

refined trajectories
    (B,N,H,K,T,2)

probabilities
    (B,N,H,K)

refinement_scores
    (B,N,H,K)

offsets
    (B,N,H,K,T,2)

Training-history tensors
------------------------

trajectory_history
    (B,N,H,K,C+1,T,2)

    Contains:

        Y^(0), Y^(1), ..., Y^(C)

refinement_score_history
    (B,N,H,K,C+1)

    Contains:

        RScore_0, RScore_1, ..., RScore_C

The history is retained so the score-loss implementation can
directly implement the paper's Eq. (33).

Important
---------

The paper defines the normalized refinement score as

    RScore_c =
        1 - (e_c - e_min) / (e_max - e_min)

but the supplied paper text does not define an inference-time
ground-truth error construction in enough detail to silently
invent one here.

Therefore:

    compute_refinement_score()

implements Eq. (26) exactly.

The neural refinement-score head predicts the score used by
the model. The training loss will construct the ground-truth
score labels from the refinement trajectory errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.layers.attention import MultiHeadAttention

from models.model_types import (
    Prediction,
    RefinedPrediction,
)

from models.refinement.anchor_selector import (
    AnchorSelection,
    AnchorSelector,
)

from models.refinement.context_encoder import (
    ContextEncoder,
)


###############################################################################
# Refinement Cycle Output
###############################################################################


@dataclass(frozen=True, slots=True)
class RefinementCycleOutput:
    """
    Output of one two-segment refinement cycle.

    Parameters
    ----------
    trajectories
        Refined trajectory after this cycle.

        Shape:

            (B,N,H,K,T,2)

    offsets
        Offset produced by this cycle.

        Shape:

            (B,N,H,K,T,2)

    refinement_scores
        Predicted refinement-quality score associated with
        the trajectory entering this cycle.

        Shape:

            (B,N,H,K)

    anchor_context
        Anchor-conditioned contextual representation.

        Shape:

            (B,N,H,K,2,D)
    """

    trajectories: Tensor
    offsets: Tensor
    refinement_scores: Tensor
    anchor_context: Tensor


###############################################################################
# Segment Refinement
###############################################################################


class SegmentRefinement(nn.Module):
    """
    Refine one trajectory segment.

    Paper mechanism
    ---------------

        Q = Z_STM
        K = Z_A
        V = Z_A

    The integrated representation is used to predict
    trajectory offsets.

    Input
    -----

    z_stm
        (B,N,H,K,D)

    anchor_context
        (B,N,H,K,D)

    trajectory
        (B,N,H,K,T,2)

    segment_mask
        (T,)

    Output
    ------

    offsets
        (B,N,H,K,T,2)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        #######################################################################
        # Cross attention
        #######################################################################

        self.cross_attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        #######################################################################
        # Normalization
        #######################################################################

        self.query_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.context_norm = nn.LayerNorm(
            hidden_dim,
        )

        self.output_norm = nn.LayerNorm(
            hidden_dim,
        )

        #######################################################################
        # Offset prediction
        #######################################################################

        self.offset_head = nn.Sequential(
            nn.Linear(
                hidden_dim + 2,
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
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    ###########################################################################
    # Cross attention
    ###########################################################################

    def _cross_attention(
        self,
        *,
        z_stm: Tensor,
        anchor_context: Tensor,
    ) -> Tensor:
        """
        Perform

            Q = Z_STM
            K = Z_A
            V = Z_A

        Parameters
        ----------
        z_stm
            (B,N,H,K,D)

        anchor_context
            (B,N,H,K,D)

        Returns
        -------

        integrated
            (B,N,H,K,D)
        """

        if z_stm.ndim != 5:
            raise ValueError(
                "z_stm must have shape "
                "(B,N,H,K,D)."
            )

        if anchor_context.ndim != 5:
            raise ValueError(
                "anchor_context must have shape "
                "(B,N,H,K,D)."
            )

        if z_stm.shape != anchor_context.shape:
            raise ValueError(
                "z_stm and anchor_context must have "
                "identical shapes."
            )

        B, N, H, K, D = z_stm.shape

        #######################################################################
        # Flatten B,N,H,K for attention.
        #######################################################################

        query = z_stm.reshape(
            B * N * H,
            K,
            D,
        )

        key = anchor_context.reshape(
            B * N * H,
            K,
            D,
        )

        value = key

        #######################################################################
        # Normalize Q/K/V inputs.
        #######################################################################

        query = self.query_norm(
            query,
        )

        key = self.context_norm(
            key,
        )

        value = key

        #######################################################################
        # Cross attention
        #######################################################################

        integrated = self.cross_attention(
            query=query,
            key=key,
            value=value,
        )

        #######################################################################
        # Restore B,N,H,K,D
        #######################################################################

        integrated = integrated.reshape(
            B,
            N,
            H,
            K,
            D,
        )

        integrated = self.output_norm(
            integrated,
        )

        return integrated

    ###########################################################################
    # Offset prediction
    ###########################################################################

    def predict_offsets(
        self,
        *,
        integrated_features: Tensor,
        trajectory: Tensor,
        segment_mask: Tensor,
    ) -> Tensor:
        """
        Predict offsets for one trajectory segment.

        Parameters
        ----------
        integrated_features
            (B,N,H,K,D)

        trajectory
            (B,N,H,K,T,2)

        segment_mask
            (T,)

        Returns
        -------

        offsets
            (B,N,H,K,T,2)
        """

        if integrated_features.ndim != 5:
            raise ValueError(
                "integrated_features must have shape "
                "(B,N,H,K,D)."
            )

        if trajectory.ndim != 6:
            raise ValueError(
                "trajectory must have shape "
                "(B,N,H,K,T,2)."
            )

        B, N, H, K, T, coordinate_dim = (
            trajectory.shape
        )

        if coordinate_dim != 2:
            raise ValueError(
                "Trajectory coordinate dimension must be 2."
            )

        if integrated_features.shape[:4] != (
            B,
            N,
            H,
            K,
        ):
            raise ValueError(
                "integrated_features and trajectory must "
                "agree on (B,N,H,K)."
            )

        #######################################################################
        # Expand context over prediction horizon.
        #######################################################################

        context = (
            integrated_features
            .unsqueeze(-2)
            .expand(
                B,
                N,
                H,
                K,
                T,
                self.hidden_dim,
            )
        )

        #######################################################################
        # Combine context with current trajectory coordinates.
        #######################################################################

        features = torch.cat(
            (
                context,
                trajectory,
            ),
            dim=-1,
        )

        #######################################################################
        # Predict coordinate offsets.
        #######################################################################

        offsets = self.offset_head(
            features,
        )

        #######################################################################
        # Restrict updates to this trajectory segment.
        #######################################################################

        mask = segment_mask.to(
            device=trajectory.device,
            dtype=trajectory.dtype,
        )

        mask = mask.view(
            1,
            1,
            1,
            1,
            T,
            1,
        )

        offsets = offsets * mask

        return offsets

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        z_stm: Tensor,
        anchor_context: Tensor,
        trajectory: Tensor,
        segment_mask: Tensor,
    ) -> Tensor:
        """
        Refine one trajectory segment.
        """

        integrated = self._cross_attention(
            z_stm=z_stm,
            anchor_context=anchor_context,
        )

        offsets = self.predict_offsets(
            integrated_features=integrated,
            trajectory=trajectory,
            segment_mask=segment_mask,
        )

        return offsets

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}"
        )


###############################################################################
# Refinement
###############################################################################


class Refinement(nn.Module):
    """
    Adaptive Anchor-Based Trajectory Refinement.

    A refinement iteration contains two segment refinements:

        Segment 1:
            start -> midpoint

        Segment 2:
            midpoint -> endpoint

    The paper uses a two-iteration refinement strategy.

    Input
    -----

    z_stm
        (B,N,H,K,D)

    prediction.trajectories
        (B,N,H,K,T,2)

    prediction.probabilities
        (B,N,H,K)

    Output
    ------

    RefinedPrediction
    """

    NUM_SEGMENTS = 2

    ANCHOR_MIDPOINT = 0
    ANCHOR_ENDPOINT = 1

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        prediction_steps: int = 30,
        refinement_iterations: int = 2,
        radius_start: float = 30.0,
        radius_end: float = 10.0,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if prediction_steps < 2:
            raise ValueError(
                "prediction_steps must be at least 2."
            )

        if refinement_iterations <= 0:
            raise ValueError(
                "refinement_iterations must be positive."
            )

        if radius_start <= 0.0:
            raise ValueError(
                "radius_start must be positive."
            )

        if radius_end <= 0.0:
            raise ValueError(
                "radius_end must be positive."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.prediction_steps = prediction_steps
        self.refinement_iterations = (
            refinement_iterations
        )

        self.radius_start = radius_start
        self.radius_end = radius_end

        #######################################################################
        # Anchor selector
        #######################################################################

        self.anchor_selector = AnchorSelector(
            radius_start=radius_start,
            radius_end=radius_end,
        )

        #######################################################################
        # Anchor-context encoder
        #######################################################################

        self.context_encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Two segment refinement modules
        #######################################################################

        self.segment_refiners = nn.ModuleList(
            [
                SegmentRefinement(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(
                    self.NUM_SEGMENTS
                )
            ]
        )

        #######################################################################
        # Learned refinement-quality head
        #
        # Output:
        #
        #     (B,N,H,K)
        #
        #######################################################################

        self.refinement_score_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

    ###########################################################################
    # Segment masks
    ###########################################################################

    @staticmethod
    def _build_segment_masks(
        prediction_steps: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Construct the two trajectory-segment masks.

        For T=30:

            midpoint = 14

            segment 1:
                0 ... 14

            segment 2:
                15 ... 29

        This prevents the midpoint from being updated twice.
        """

        if prediction_steps < 2:
            raise ValueError(
                "prediction_steps must be at least 2."
            )

        midpoint = (
            prediction_steps - 1
        ) // 2

        indices = torch.arange(
            prediction_steps,
            device=device,
        )

        first_mask = (
            indices <= midpoint
        )

        second_mask = (
            indices > midpoint
        )

        return (
            first_mask,
            second_mask,
        )

    ###########################################################################
    # Input validation
    ###########################################################################

    def _validate_inputs(
        self,
        z_stm: Tensor,
        prediction: Prediction,
    ) -> None:
        """
        Validate refinement inputs.
        """

        if not isinstance(
            z_stm,
            torch.Tensor,
        ):
            raise TypeError(
                "z_stm must be a torch.Tensor."
            )

        if z_stm.ndim != 5:
            raise ValueError(
                "z_stm must have shape "
                "(B,N,H,K,D)."
            )

        if not torch.is_floating_point(
            z_stm,
        ):
            raise TypeError(
                "z_stm must contain floating-point values."
            )

        if not torch.isfinite(
            z_stm,
        ).all():
            raise ValueError(
                "z_stm contains NaN or infinite values."
            )

        trajectories = (
            prediction.trajectories
        )

        probabilities = (
            prediction.probabilities
        )

        if trajectories.ndim != 6:
            raise ValueError(
                "prediction.trajectories must have shape "
                "(B,N,H,K,T,2)."
            )

        if probabilities.ndim != 4:
            raise ValueError(
                "prediction.probabilities must have shape "
                "(B,N,H,K)."
            )

        if trajectories.shape[-1] != 2:
            raise ValueError(
                "Trajectory coordinate dimension must be 2."
            )

        if trajectories.shape[:4] != (
            z_stm.shape[0],
            z_stm.shape[1],
            z_stm.shape[2],
            z_stm.shape[3],
        ):
            raise ValueError(
                "z_stm and prediction trajectories must "
                "agree on (B,N,H,K)."
            )

        if z_stm.shape[-1] != (
            self.hidden_dim
        ):
            raise ValueError(
                f"Expected z_stm hidden dimension "
                f"{self.hidden_dim}, got "
                f"{z_stm.shape[-1]}."
            )

        expected_probability_shape = (
            trajectories.shape[0],
            trajectories.shape[1],
            trajectories.shape[2],
            trajectories.shape[3],
        )

        if tuple(
            probabilities.shape
        ) != expected_probability_shape:
            raise ValueError(
                "Prediction probabilities must have shape "
                "(B,N,H,K) matching trajectories."
            )

        if not torch.isfinite(
            trajectories,
        ).all():
            raise ValueError(
                "Prediction trajectories contain "
                "NaN or infinite values."
            )

        if not torch.isfinite(
            probabilities,
        ).all():
            raise ValueError(
                "Prediction probabilities contain "
                "NaN or infinite values."
            )

    ###########################################################################
    # Anchor selection
    ###########################################################################

    def _select_anchors(
        self,
        *,
        trajectories: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Select midpoint and endpoint anchors.

        Returns
        -------

        anchors
            (B,N,H,K,2,2)

        radii
            (2,)
        """

        selection: AnchorSelection = (
            self.anchor_selector(
                trajectories,
            )
        )

        anchors = selection.anchors
        radii = selection.radii

        expected_anchor_shape = (
            trajectories.shape[0],
            trajectories.shape[1],
            trajectories.shape[2],
            trajectories.shape[3],
            self.NUM_SEGMENTS,
            2,
        )

        if tuple(
            anchors.shape
        ) != expected_anchor_shape:
            raise RuntimeError(
                "AnchorSelector returned invalid anchor shape. "
                f"Expected {expected_anchor_shape}, got "
                f"{tuple(anchors.shape)}."
            )

        if tuple(
            radii.shape
        ) != (
            self.NUM_SEGMENTS,
        ):
            raise RuntimeError(
                "AnchorSelector returned invalid radii shape. "
                f"Expected {(self.NUM_SEGMENTS,)}, got "
                f"{tuple(radii.shape)}."
            )

        return (
            anchors,
            radii,
        )

    ###########################################################################
    # Anchor context encoding
    ###########################################################################

    def _encode_anchor_context(
        self,
        *,
        z_stm: Tensor,
        anchors: Tensor,
        radii: Tensor,
    ) -> Tensor:
        """
        Encode anchor-conditioned context.

        Returns

            (B,N,H,K,2,D)
        """

        anchor_context = (
            self.context_encoder(
                scene_features=z_stm,
                anchors=anchors,
                radii=radii,
            )
        )

        expected_shape = (
            z_stm.shape[0],
            z_stm.shape[1],
            z_stm.shape[2],
            z_stm.shape[3],
            self.NUM_SEGMENTS,
            self.hidden_dim,
        )

        if tuple(
            anchor_context.shape
        ) != expected_shape:
            raise RuntimeError(
                "ContextEncoder returned invalid shape. "
                f"Expected {expected_shape}, got "
                f"{tuple(anchor_context.shape)}."
            )

        if not torch.isfinite(
            anchor_context,
        ).all():
            raise RuntimeError(
                "ContextEncoder produced non-finite values."
            )

        return anchor_context

    ###########################################################################
    # Refinement score prediction
    ###########################################################################

    def _predict_refinement_score(
        self,
        anchor_context: Tensor,
    ) -> Tensor:
        """
        Predict refinement quality.

        Input:

            (B,N,H,K,2,D)

        Output:

            (B,N,H,K)
        """

        if anchor_context.ndim != 6:
            raise ValueError(
                "anchor_context must have shape "
                "(B,N,H,K,2,D)."
            )

        #######################################################################
        # Fuse midpoint and endpoint contextual features.
        #######################################################################

        context = anchor_context.mean(
            dim=-2,
        )

        score = self.refinement_score_head(
            context,
        )

        return score.squeeze(
            dim=-1,
        )

    ###########################################################################
    # Score for a particular trajectory state
    ###########################################################################

    def _score_for_trajectory(
        self,
        *,
        z_stm: Tensor,
        trajectory: Tensor,
    ) -> Tensor:
        """
        Produce a predicted refinement score for a particular
        trajectory state.

        This is used for:

            Y^(0) -> RScore_0
            Y^(1) -> RScore_1
            ...
            Y^(C) -> RScore_C

        Returns

            (B,N,H,K)
        """

        anchors, radii = (
            self._select_anchors(
                trajectories=trajectory,
            )
        )

        anchor_context = (
            self._encode_anchor_context(
                z_stm=z_stm,
                anchors=anchors,
                radii=radii,
            )
        )

        return self._predict_refinement_score(
            anchor_context,
        )

    ###########################################################################
    # One refinement cycle
    ###########################################################################

    def _refinement_cycle(
        self,
        *,
        z_stm: Tensor,
        trajectory: Tensor,
    ) -> RefinementCycleOutput:
        """
        Execute one complete two-segment refinement cycle.

        The cycle computes:

            anchors
                ->
            anchor context
                ->
            segment 1 offset
                ->
            segment 2 offset
                ->
            refined trajectory
        """

        #######################################################################
        # Anchor selection
        #######################################################################

        anchors, radii = (
            self._select_anchors(
                trajectories=trajectory,
            )
        )

        #######################################################################
        # Anchor-centric context
        #######################################################################

        anchor_context = (
            self._encode_anchor_context(
                z_stm=z_stm,
                anchors=anchors,
                radii=radii,
            )
        )

        #######################################################################
        # Segment masks
        #######################################################################

        first_mask, second_mask = (
            self._build_segment_masks(
                prediction_steps=trajectory.shape[-2],
                device=trajectory.device,
            )
        )

        #######################################################################
        # Segment 1: start -> midpoint
        #######################################################################

        midpoint_context = anchor_context[
            ...,
            self.ANCHOR_MIDPOINT,
            :,
        ]

        first_offset = (
            self.segment_refiners[0](
                z_stm=z_stm,
                anchor_context=midpoint_context,
                trajectory=trajectory,
                segment_mask=first_mask,
            )
        )

        trajectory_after_first = (
            trajectory
            + first_offset
        )

        #######################################################################
        # Segment 2: midpoint -> endpoint
        #######################################################################

        endpoint_context = anchor_context[
            ...,
            self.ANCHOR_ENDPOINT,
            :,
        ]

        second_offset = (
            self.segment_refiners[1](
                z_stm=z_stm,
                anchor_context=endpoint_context,
                trajectory=trajectory_after_first,
                segment_mask=second_mask,
            )
        )

        trajectory_after_second = (
            trajectory_after_first
            + second_offset
        )

        #######################################################################
        # Total cycle offset
        #######################################################################

        total_offset = (
            trajectory_after_second
            - trajectory
        )

        #######################################################################
        # Score associated with the incoming trajectory.
        #
        # This is the score for the current refinement state.
        #######################################################################

        refinement_scores = (
            self._predict_refinement_score(
                anchor_context,
            )
        )

        return RefinementCycleOutput(
            trajectories=trajectory_after_second,
            offsets=total_offset,
            refinement_scores=refinement_scores,
            anchor_context=anchor_context,
        )

    ###########################################################################
    # Eq. (26)
    ###########################################################################

    @staticmethod
    def compute_refinement_score(
        error: Tensor,
        minimum_error: Tensor | float,
        maximum_error: Tensor | float,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Compute the normalized refinement score from Eq. (26).

            RScore_c =
                1 -
                (e_c - e_min)
                ----------------
                (e_max - e_min)

        Parameters
        ----------
        error
            Current refinement error e_c.

        minimum_error
            e_min.

        maximum_error
            e_max.

        eps
            Numerical stability constant.

        Returns
        -------

        score
            Normalized score in [0,1].
        """

        if not isinstance(
            error,
            torch.Tensor,
        ):
            raise TypeError(
                "error must be a torch.Tensor."
            )

        minimum = torch.as_tensor(
            minimum_error,
            device=error.device,
            dtype=error.dtype,
        )

        maximum = torch.as_tensor(
            maximum_error,
            device=error.device,
            dtype=error.dtype,
        )

        denominator = (
            maximum
            - minimum
        ).clamp_min(
            eps,
        )

        score = (
            1.0
            - (
                error
                - minimum
            )
            / denominator
        )

        return score.clamp(
            0.0,
            1.0,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        z_stm: Tensor | None = None,
        prediction: Prediction,
        encoder_features: Tensor | None = None,
    ) -> RefinedPrediction:
        """
        Refine coarse decoder trajectories.

        Parameters
        ----------
        z_stm
            Tri-ATM output.

            (B,N,H,K,D)

        prediction
            Coarse decoder prediction.

        encoder_features
            Backward-compatible alias for z_stm.

        Notes
        -----

        ``encoder_features`` is retained only for compatibility with
        earlier DSTNet call sites. If both are supplied, z_stm is used.

        The refinement history contains:

            Y^(0), Y^(1), ..., Y^(C)

        and

            RScore_0, RScore_1, ..., RScore_C
        """

        #######################################################################
        # Backward-compatible argument handling
        #######################################################################

        if z_stm is None:
            z_stm = encoder_features

        if z_stm is None:
            raise TypeError(
                "Refinement.forward() requires either "
                "'z_stm' or 'encoder_features'."
            )

        #######################################################################
        # Validate
        #######################################################################

        self._validate_inputs(
            z_stm,
            prediction,
        )

        #######################################################################
        # Initial coarse trajectory
        #
        # Y^(0)
        #######################################################################

        initial_trajectory = (
            prediction.trajectories
        )

        probabilities = (
            prediction.probabilities
        )

        refined = initial_trajectory

        #######################################################################
        # History containers
        #
        # The first element is the coarse prediction Y^(0).
        #######################################################################

        trajectory_history: list[Tensor] = [
            refined,
        ]

        initial_score = (
            self._score_for_trajectory(
                z_stm=z_stm,
                trajectory=refined,
            )
        )

        refinement_score_history: list[Tensor] = [
            initial_score,
        ]

        #######################################################################
        # Refinement cycles
        #######################################################################

        for _ in range(
            self.refinement_iterations
        ):

            cycle = (
                self._refinement_cycle(
                    z_stm=z_stm,
                    trajectory=refined,
                )
            )

            ###################################################################
            # Y^(i+1)
            ###################################################################

            refined = cycle.trajectories

            ###################################################################
            # Store Y^(i+1)
            ###################################################################

            trajectory_history.append(
                refined,
            )

            ###################################################################
            # Predict score for the UPDATED trajectory.
            #
            # Therefore history aligns as:
            #
            #     trajectory_history[i]
            #         <-->
            #     refinement_score_history[i]
            ###################################################################

            updated_score = (
                self._score_for_trajectory(
                    z_stm=z_stm,
                    trajectory=refined,
                )
            )

            refinement_score_history.append(
                updated_score,
            )

        #######################################################################
        # Total offset from coarse prediction
        #######################################################################

        total_offsets = (
            refined
            - initial_trajectory
        )

        #######################################################################
        # Stack history
        #######################################################################

        trajectory_history_tensor = (
            torch.stack(
                trajectory_history,
                dim=4,
            )
        )

        refinement_score_history_tensor = (
            torch.stack(
                refinement_score_history,
                dim=-1,
            )
        )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.isfinite(
            refined,
        ).all():
            raise FloatingPointError(
                "Refinement produced non-finite "
                "trajectory values."
            )

        if not torch.isfinite(
            total_offsets,
        ).all():
            raise FloatingPointError(
                "Refinement produced non-finite "
                "offset values."
            )

        if not torch.isfinite(
            probabilities,
        ).all():
            raise FloatingPointError(
                "Prediction probabilities contain "
                "non-finite values."
            )

        if not torch.isfinite(
            trajectory_history_tensor,
        ).all():
            raise FloatingPointError(
                "Trajectory refinement history contains "
                "non-finite values."
            )

        if not torch.isfinite(
            refinement_score_history_tensor,
        ).all():
            raise FloatingPointError(
                "Refinement score history contains "
                "non-finite values."
            )

        #######################################################################
        # Final typed output
        #######################################################################

        return RefinedPrediction(
            trajectories=refined,
            probabilities=probabilities,
            refinement_scores=(
                refinement_score_history_tensor[
                    ...,
                    -1,
                ]
            ),
            offsets=total_offsets,
            trajectory_history=(
                trajectory_history_tensor
            ),
            refinement_score_history=(
                refinement_score_history_tensor
            ),
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"prediction_steps={self.prediction_steps}, "
            f"refinement_iterations="
            f"{self.refinement_iterations}"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "Refinement",
    "SegmentRefinement",
    "RefinementCycleOutput",
]
