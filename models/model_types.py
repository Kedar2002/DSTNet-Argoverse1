"""
models.model_types

Strongly-typed data structures used throughout the DSTNet model.

These classes define the interfaces exchanged between model
components. They are intentionally lightweight and contain only
tensors or immutable metadata.

Current DSTNet tensor contracts
--------------------------------

Agent embeddings:

    Ea
        (B,N,H,D)

Map embeddings:

    Em
        (B,M,D)

Relative spatio-temporal embeddings:

    Er
        edge_index : (2,U)
        embeddings  : (U,D)
        edge_type   : (U,)

GSTA output:

    Z_scene
        (B,N,H,K,D)

Tri-ATM output:

    Z_STM
        (B,N,H,K,D)

Coarse decoder output:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

Refinement output:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

    refinement_scores
        (B,N,H,K)

    offsets
        (B,N,H,K,T,2)

The historical dimension H is intentionally preserved through
GSTA, Tri-ATM, decoding, and refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


###############################################################################
# Relative Spatio-Temporal Embedding
###############################################################################


@dataclass(slots=True)
class RelativeSpatioTemporalEmbedding:
    """
    Relative spatio-temporal embeddings.

    Er ∈ R^(U × D)

    Each row corresponds to one edge in the SceneGraph.

    Parameters
    ----------
    edge_index
        Unified graph edge indices.

        Shape:

            (2,U)

    embeddings
        Learned relative edge embeddings.

        Shape:

            (U,D)

    edge_type
        Integer edge-type identifier.

        Shape:

            (U,)

        Convention:

            0 = temporal
            1 = spatial
            2 = agent-map
            3 = map-map
    """

    edge_index: torch.Tensor

    embeddings: torch.Tensor

    edge_type: torch.Tensor | None = None

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the basic representation.
        """

        if not isinstance(
            self.edge_index,
            torch.Tensor,
        ):
            raise TypeError(
                "edge_index must be a torch.Tensor."
            )

        if not isinstance(
            self.embeddings,
            torch.Tensor,
        ):
            raise TypeError(
                "embeddings must be a torch.Tensor."
            )

        if self.edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape (2,U). "
                f"Got {tuple(self.edge_index.shape)}."
            )

        if self.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index first dimension must be 2. "
                f"Got {self.edge_index.shape[0]}."
            )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (U,D). "
                f"Got {tuple(self.embeddings.shape)}."
            )

        if (
            self.embeddings.shape[0]
            != self.edge_index.shape[1]
        ):
            raise ValueError(
                "Number of edge embeddings must match "
                "number of edges."
            )

        if self.edge_type is not None:

            if not isinstance(
                self.edge_type,
                torch.Tensor,
            ):
                raise TypeError(
                    "edge_type must be a torch.Tensor or None."
                )

            if self.edge_type.ndim != 1:
                raise ValueError(
                    "edge_type must have shape (U,). "
                    f"Got {tuple(self.edge_type.shape)}."
                )

            if (
                self.edge_type.shape[0]
                != self.edge_index.shape[1]
            ):
                raise ValueError(
                    "edge_type length must match "
                    "number of edges."
                )


###############################################################################
# Encoded Scene
###############################################################################


@dataclass(slots=True)
class EncodedScene:
    """
    Encoded scene representation.

    This container represents the output of the scene encoder
    before multimodal prediction decoding.

    Parameters
    ----------
    Ea
        Agent embeddings.

        Shape:

            (B,N,H,D)

    Em
        Map embeddings.

        Shape:

            (B,M,D)

    Er
        Relative spatio-temporal edge embeddings.
    """

    Ea: torch.Tensor

    Em: torch.Tensor

    Er: RelativeSpatioTemporalEmbedding

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.Ea,
            torch.Tensor,
        ):
            raise TypeError(
                "EncodedScene.Ea must be a torch.Tensor."
            )

        if not isinstance(
            self.Em,
            torch.Tensor,
        ):
            raise TypeError(
                "EncodedScene.Em must be a torch.Tensor."
            )

        if self.Ea.ndim != 4:
            raise ValueError(
                "EncodedScene.Ea must have shape "
                "(B,N,H,D). "
                f"Got {tuple(self.Ea.shape)}."
            )

        if self.Em.ndim != 3:
            raise ValueError(
                "EncodedScene.Em must have shape "
                "(B,M,D). "
                f"Got {tuple(self.Em.shape)}."
            )


###############################################################################
# Mode Features
###############################################################################


@dataclass(slots=True)
class ModeFeatures:
    """
    Multimodal prediction embeddings.

    Current DSTNet representation:

        (B,N,H,K,D)

    where

        B : batch size
        N : number of agents
        H : historical time steps
        K : prediction modes
        D : hidden feature dimension

    The historical dimension H is preserved because GSTA,
    Tri-ATM, and the current decoder operate on agent-state
    prediction embeddings.
    """

    features: torch.Tensor

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.features,
            torch.Tensor,
        ):
            raise TypeError(
                "ModeFeatures.features must be a torch.Tensor."
            )

        if self.features.ndim != 5:
            raise ValueError(
                "ModeFeatures.features must have shape "
                "(B,N,H,K,D). "
                f"Got {tuple(self.features.shape)}."
            )


###############################################################################
# Prediction
###############################################################################


@dataclass(slots=True)
class Prediction:
    """
    Coarse multimodal trajectory prediction.

    This is the output of the trajectory decoder before
    anchor-based refinement.

    Parameters
    ----------
    trajectories
        Coarse trajectory hypotheses.

        Shape:

            (B,N,H,K,T,2)

    probabilities
        Probability/logit-normalized weight for each
        trajectory hypothesis.

        Shape:

            (B,N,H,K)

        The K dimension indexes the multimodal trajectory
        hypotheses.

    Notes
    -----
    The field is explicitly named ``probabilities`` rather
    than ``scores`` to distinguish trajectory likelihoods from
    the separate refinement-quality score used by AAR.
    """

    trajectories: torch.Tensor

    probabilities: torch.Tensor

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the coarse prediction representation.
        """

        if not isinstance(
            self.trajectories,
            torch.Tensor,
        ):
            raise TypeError(
                "Prediction.trajectories must be a torch.Tensor."
            )

        if not isinstance(
            self.probabilities,
            torch.Tensor,
        ):
            raise TypeError(
                "Prediction.probabilities must be a "
                "torch.Tensor."
            )

        if self.trajectories.ndim != 6:
            raise ValueError(
                "Prediction.trajectories must have shape "
                "(B,N,H,K,T,2). "
                f"Got {tuple(self.trajectories.shape)}."
            )

        if self.trajectories.shape[-1] != 2:
            raise ValueError(
                "Prediction trajectory coordinate dimension "
                "must equal 2."
            )

        if self.probabilities.ndim != 4:
            raise ValueError(
                "Prediction.probabilities must have shape "
                "(B,N,H,K). "
                f"Got {tuple(self.probabilities.shape)}."
            )

        expected_prefix = (
            self.trajectories.shape[0],
            self.trajectories.shape[1],
            self.trajectories.shape[2],
            self.trajectories.shape[3],
        )

        if tuple(
            self.probabilities.shape
        ) != expected_prefix:
            raise ValueError(
                "Prediction.probabilities must match "
                "(B,N,H,K) of trajectories. "
                f"Expected {expected_prefix}, got "
                f"{tuple(self.probabilities.shape)}."
            )

    ###########################################################################
    # Backward-compatible alias
    ###########################################################################

    @property
    def scores(
        self,
    ) -> torch.Tensor:
        """
        Backward-compatible alias for ``probabilities``.

        New code should use:

            prediction.probabilities

        Existing loss/test code using:

            prediction.scores

        remains functional during the migration.
        """

        return self.probabilities


###############################################################################
# Refined Prediction
###############################################################################


@dataclass(slots=True)
class RefinedPrediction:
    """
    Final refined trajectory prediction.

    Main outputs
    ------------

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

    refinement_scores
        (B,N,H,K)

    offsets
        (B,N,H,K,T,2)

    Training histories
    ------------------

    trajectory_history
        (B,N,H,K,C+1,T,2)

        Contains:

            Y^(0), Y^(1), ..., Y^(C)

    refinement_score_history
        (B,N,H,K,C+1)

        Contains:

            RScore_0, RScore_1, ..., RScore_C
    """

    trajectories: torch.Tensor

    probabilities: torch.Tensor

    refinement_scores: torch.Tensor

    offsets: torch.Tensor | None = None

    trajectory_history: torch.Tensor | None = None

    refinement_score_history: torch.Tensor | None = None

    def __post_init__(
        self,
    ) -> None:

        #######################################################################
        # Trajectories
        #######################################################################

        if not isinstance(
            self.trajectories,
            torch.Tensor,
        ):
            raise TypeError(
                "RefinedPrediction.trajectories must be "
                "a torch.Tensor."
            )

        if self.trajectories.ndim != 6:
            raise ValueError(
                "RefinedPrediction.trajectories must have shape "
                "(B,N,H,K,T,2). "
                f"Got {tuple(self.trajectories.shape)}."
            )

        if self.trajectories.shape[-1] != 2:
            raise ValueError(
                "RefinedPrediction trajectory coordinate "
                "dimension must equal 2."
            )

        #######################################################################
        # Probabilities
        #######################################################################

        if not isinstance(
            self.probabilities,
            torch.Tensor,
        ):
            raise TypeError(
                "RefinedPrediction.probabilities must be "
                "a torch.Tensor."
            )

        if self.probabilities.ndim != 4:
            raise ValueError(
                "RefinedPrediction.probabilities must have shape "
                "(B,N,H,K). "
                f"Got {tuple(self.probabilities.shape)}."
            )

        expected_prefix = (
            self.trajectories.shape[0],
            self.trajectories.shape[1],
            self.trajectories.shape[2],
            self.trajectories.shape[3],
        )

        if tuple(
            self.probabilities.shape
        ) != expected_prefix:
            raise ValueError(
                "RefinedPrediction.probabilities must match "
                "(B,N,H,K) of trajectories. "
                f"Expected {expected_prefix}, got "
                f"{tuple(self.probabilities.shape)}."
            )

        #######################################################################
        # Refinement scores
        #######################################################################

        if not isinstance(
            self.refinement_scores,
            torch.Tensor,
        ):
            raise TypeError(
                "RefinedPrediction.refinement_scores must be "
                "a torch.Tensor."
            )

        if self.refinement_scores.ndim != 4:
            raise ValueError(
                "RefinedPrediction.refinement_scores must have "
                "shape (B,N,H,K). "
                f"Got {tuple(self.refinement_scores.shape)}."
            )

        if tuple(
            self.refinement_scores.shape
        ) != expected_prefix:
            raise ValueError(
                "RefinedPrediction.refinement_scores must match "
                "(B,N,H,K) of trajectories."
            )

        #######################################################################
        # Offsets
        #######################################################################

        if self.offsets is not None:

            if not isinstance(
                self.offsets,
                torch.Tensor,
            ):
                raise TypeError(
                    "RefinedPrediction.offsets must be a "
                    "torch.Tensor or None."
                )

            if self.offsets.shape != (
                self.trajectories.shape
            ):
                raise ValueError(
                    "RefinedPrediction.offsets must have the "
                    "same shape as trajectories."
                )

        #######################################################################
        # Trajectory history
        #######################################################################

        if self.trajectory_history is not None:

            if not isinstance(
                self.trajectory_history,
                torch.Tensor,
            ):
                raise TypeError(
                    "trajectory_history must be a torch.Tensor "
                    "or None."
                )

            if self.trajectory_history.ndim != 7:
                raise ValueError(
                    "trajectory_history must have shape "
                    "(B,N,H,K,C+1,T,2). "
                    f"Got "
                    f"{tuple(self.trajectory_history.shape)}."
                )

            expected_history_prefix = (
                self.trajectories.shape[0],
                self.trajectories.shape[1],
                self.trajectories.shape[2],
                self.trajectories.shape[3],
            )

            if self.trajectory_history.shape[:4] != (
                expected_history_prefix
            ):
                raise ValueError(
                    "trajectory_history must match trajectories "
                    "on (B,N,H,K). "
                    f"Expected prefix "
                    f"{expected_history_prefix}, got "
                    f"{tuple(self.trajectory_history.shape[:4])}."
                )

            if self.trajectory_history.shape[-2:] != (
                self.trajectories.shape[-2],
                self.trajectories.shape[-1],
            ):
                raise ValueError(
                    "trajectory_history must match trajectories "
                    "on (T,2). "
                    f"Expected "
                    f"{tuple(self.trajectories.shape[-2:])}, got "
                    f"{tuple(self.trajectory_history.shape[-2:])}."
                )

            if self.trajectory_history.shape[4] < 1:
                raise ValueError(
                    "trajectory_history must contain at least "
                    "the initial coarse trajectory Y^(0)."
                )

            if not torch.isfinite(
                self.trajectory_history
            ).all():
                raise ValueError(
                    "trajectory_history contains NaN or "
                    "infinite values."
                )

        #######################################################################
        # Refinement score history
        #######################################################################

        if self.refinement_score_history is not None:

            if not isinstance(
                self.refinement_score_history,
                torch.Tensor,
            ):
                raise TypeError(
                    "refinement_score_history must be a "
                    "torch.Tensor or None."
                )

            if self.refinement_score_history.ndim != 5:
                raise ValueError(
                    "refinement_score_history must have shape "
                    "(B,N,H,K,C+1). "
                    f"Got "
                    f"{tuple(self.refinement_score_history.shape)}."
                )

            expected_score_prefix = (
                self.trajectories.shape[0],
                self.trajectories.shape[1],
                self.trajectories.shape[2],
                self.trajectories.shape[3],
            )

            if self.refinement_score_history.shape[:4] != (
                expected_score_prefix
            ):
                raise ValueError(
                    "refinement_score_history must match "
                    "trajectories on (B,N,H,K). "
                    f"Expected prefix "
                    f"{expected_score_prefix}, got "
                    f"{tuple(self.refinement_score_history.shape[:4])}."
                )

            if self.refinement_score_history.shape[4] < 1:
                raise ValueError(
                    "refinement_score_history must contain at "
                    "least RScore_0."
                )

            if not torch.isfinite(
                self.refinement_score_history
            ).all():
                raise ValueError(
                    "refinement_score_history contains NaN or "
                    "infinite values."
                )

    ###########################################################################
    # Backward-compatible alias
    ###########################################################################

    @property
    def scores(
        self,
    ) -> torch.Tensor:
        """
        Backward-compatible alias for trajectory probabilities.
        """

        return self.probabilities


###############################################################################
# Attention Features
###############################################################################


@dataclass(slots=True)
class AttentionFeatures:
    """
    Intermediate attention representations.

    These tensors correspond to internal GSTA representations.

    Zti
        Temporal agent-state representation.

    Zsj
        Spatial/map representation.

    The exact dimensions depend on the corresponding GSTA
    stage, therefore these containers intentionally do not
    enforce one fixed dimensionality.
    """

    Zti: torch.Tensor

    Zsj: torch.Tensor


###############################################################################
# Public API
###############################################################################


__all__ = [
    "RelativeSpatioTemporalEmbedding",
    "EncodedScene",
    "ModeFeatures",
    "Prediction",
    "RefinedPrediction",
    "AttentionFeatures",
]
