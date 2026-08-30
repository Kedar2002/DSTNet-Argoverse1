"""
losses.total_loss

Complete DSTNet training objective.

Paper
-----
DSTNet, Section III-G

The complete objective combines:

    L_proposal
    L_classification
    L_score
    L_refinement

with the score term weighted by lambda:

    L_total =
        L_proposal
        + L_classification
        + L_refinement
        + lambda * L_score

The default implementation uses:

    lambda = 0.1

Current DSTNet tensor contracts
-------------------------------

Prediction:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

RefinedPrediction, when refinement is enabled:

    trajectories
        (B,N,H,K,T,2)

    scores
        (B,N,H,K)

    offsets
        (B,N,H,K,T,2)

Important
---------
The current models.dstnet.DSTNet supports:

    refinement_enabled=True
        -> refined_prediction is a RefinedPrediction

    refinement_enabled=False
        -> refined_prediction is None

Therefore this loss explicitly supports both configurations.

When refinement is disabled:

    L_score      = 0
    L_refinement = 0

and:

    L_total =
        L_proposal
        + L_classification

The disabled refinement losses are created from the coarse prediction
tensor so that they remain device- and dtype-compatible.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.model_types import (
    Prediction,
    RefinedPrediction,
)

from losses.proposal_loss import ProposalLoss
from losses.classification_loss import ClassificationLoss
from losses.score_loss import ScoreLoss
from losses.refinement_loss import RefinementLoss


###############################################################################
# TotalLoss
###############################################################################


class TotalLoss(nn.Module):
    """
    Complete DSTNet training objective.

    Parameters
    ----------
    proposal_weight:
        Weight applied to the proposal loss.

    classification_weight:
        Weight applied to the classification loss.

    score_weight:
        Weight lambda applied to ScoreLoss.

        Default:
            0.1

    refinement_weight:
        Weight applied to the refinement loss.

    refinement_enabled:
        Whether refinement-dependent losses should be evaluated.

        When False:

            L_score = 0
            L_refinement = 0

        and the objective becomes:

            L_total =
                L_proposal
                + L_classification

    Notes
    -----
    With refinement enabled and default weights:

        L_total =
            L_proposal
            + L_classification
            + L_refinement
            + 0.1 L_score

    With refinement disabled:

        L_total =
            L_proposal
            + L_classification
    """

    def __init__(
        self,
        proposal_weight: float = 1.0,
        classification_weight: float = 1.0,
        score_weight: float = 0.1,
        refinement_weight: float = 1.0,
        refinement_enabled: bool = True,
    ) -> None:

        super().__init__()

        #######################################################################
        # Validate weights
        #######################################################################

        if proposal_weight < 0.0:
            raise ValueError(
                "proposal_weight must be non-negative."
            )

        if classification_weight < 0.0:
            raise ValueError(
                "classification_weight must be non-negative."
            )

        if score_weight < 0.0:
            raise ValueError(
                "score_weight must be non-negative."
            )

        if refinement_weight < 0.0:
            raise ValueError(
                "refinement_weight must be non-negative."
            )

        if not isinstance(
            refinement_enabled,
            bool,
        ):
            raise TypeError(
                "refinement_enabled must be a bool."
            )

        #######################################################################
        # Store configuration
        #######################################################################

        self.proposal_weight = float(
            proposal_weight,
        )

        self.classification_weight = float(
            classification_weight,
        )

        self.score_weight = float(
            score_weight,
        )

        self.refinement_weight = float(
            refinement_weight,
        )

        self.refinement_enabled = refinement_enabled

        #######################################################################
        # Individual loss modules
        #######################################################################

        self.proposal_loss = ProposalLoss()

        self.classification_loss = (
            ClassificationLoss()
        )

        self.score_loss = ScoreLoss()

        self.refinement_loss = (
            RefinementLoss()
        )

    ###########################################################################
    # Validation helpers
    ###########################################################################

    @staticmethod
    def _validate_scalar_loss(
        name: str,
        value: Tensor,
    ) -> None:
        """
        Validate that a loss is a finite scalar tensor.
        """

        if not isinstance(
            value,
            Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if value.ndim != 0:
            raise ValueError(
                f"{name} must be scalar. "
                f"Got shape {tuple(value.shape)}."
            )

        if not torch.isfinite(
            value,
        ).all():
            raise RuntimeError(
                f"{name} contains NaN or infinite values."
            )

    @staticmethod
    def _zero_loss(
        reference: Tensor,
    ) -> Tensor:
        """
        Create a zero scalar compatible with the supplied tensor.

        The returned tensor has:

            device = reference.device
            dtype   = reference.dtype

        and remains connected to the reference computation graph.

        No refinement-specific tensors are accessed here.
        """

        return reference.sum() * 0.0

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        prediction: Prediction,
        refined_prediction: RefinedPrediction | None,
        ground_truth: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute the complete DSTNet training objective.

        Parameters
        ----------
        prediction:
            Coarse decoder prediction.

            trajectories:
                (B,N,H,K,T,2)

            probabilities:
                (B,N,H,K)

        refined_prediction:
            Refined prediction produced by DSTNet.

            RefinedPrediction
                when refinement is enabled

            None
                when refinement is disabled

        ground_truth:
            Future ground-truth trajectories.

            Shape:

                (B,N,T,2)

        Returns
        -------
        dict[str, Tensor]

            Always contains:

                loss
                proposal_loss
                classification_loss
                score_loss
                refinement_loss

            When refinement is enabled, additional metrics returned
            by RefinementLoss are also included.
        """

        #######################################################################
        # Validate coarse prediction
        #######################################################################

        if not isinstance(
            prediction,
            Prediction,
        ):
            raise TypeError(
                "prediction must be a Prediction."
            )

        #######################################################################
        # Validate refined prediction according to configuration
        #######################################################################

        if self.refinement_enabled:

            if refined_prediction is None:

                raise ValueError(
                    "refinement_enabled=True but "
                    "refined_prediction is None."
                )

            if not isinstance(
                refined_prediction,
                RefinedPrediction,
            ):

                raise TypeError(
                    "refined_prediction must be a "
                    "RefinedPrediction when "
                    "refinement_enabled=True."
                )

        else:

            ###################################################################
            # When refinement is explicitly disabled, DSTNet should return
            # None. Do not access any refinement-specific attributes.
            ###################################################################

            if refined_prediction is not None:

                raise ValueError(
                    "refinement_enabled=False but "
                    "refined_prediction is not None."
                )

        #######################################################################
        # Validate ground truth
        #######################################################################

        if not isinstance(
            ground_truth,
            Tensor,
        ):
            raise TypeError(
                "ground_truth must be a torch.Tensor."
            )

        if ground_truth.ndim != 4:

            raise ValueError(
                "ground_truth must have shape "
                "(B,N,T,2), "
                f"got {tuple(ground_truth.shape)}."
            )

        if ground_truth.shape[-1] != 2:

            raise ValueError(
                "ground_truth must have final dimension 2 "
                "for x/y coordinates. "
                f"Got shape {tuple(ground_truth.shape)}."
            )

        if not torch.isfinite(
            ground_truth,
        ).all():

            raise RuntimeError(
                "ground_truth contains NaN or infinite values."
            )

        #######################################################################
        # Validate coarse prediction tensors
        #######################################################################

        if not isinstance(
            prediction.trajectories,
            Tensor,
        ):

            raise TypeError(
                "prediction.trajectories must be a "
                "torch.Tensor."
            )

        if not isinstance(
            prediction.probabilities,
            Tensor,
        ):

            raise TypeError(
                "prediction.probabilities must be a "
                "torch.Tensor."
            )

        if not torch.isfinite(
            prediction.trajectories,
        ).all():

            raise RuntimeError(
                "prediction.trajectories contains NaN "
                "or infinite values."
            )

        if not torch.isfinite(
            prediction.probabilities,
        ).all():

            raise RuntimeError(
                "prediction.probabilities contains NaN "
                "or infinite values."
            )

        #######################################################################
        # Proposal loss
        #
        # DSTNet proposal stage:
        #
        #     k_best = argmin endpoint error
        #
        # followed by the proposal trajectory loss.
        #######################################################################

        proposal_output = self.proposal_loss(
            prediction,
            ground_truth,
            return_best_mode=True,
        )

        if not isinstance(
            proposal_output,
            tuple,
        ):

            raise TypeError(
                "ProposalLoss must return "
                "(loss, best_mode) when "
                "return_best_mode=True."
            )

        if len(proposal_output) != 2:

            raise ValueError(
                "ProposalLoss must return exactly "
                "two values: (loss, best_mode)."
            )

        proposal_loss, best_mode = (
            proposal_output
        )

        #######################################################################
        # Classification loss
        #######################################################################

        classification_loss = (
            self.classification_loss(
                prediction,
                best_mode,
            )
        )

        #######################################################################
        # Refinement-dependent losses
        #######################################################################

        refinement_metrics: dict[str, Tensor] = {}

        if self.refinement_enabled:

            ###################################################################
            # At this point refined_prediction is guaranteed to be non-None
            # by the validation above.
            ###################################################################

            assert refined_prediction is not None

            ###################################################################
            # Score loss
            ###################################################################

            score_loss = self.score_loss(
                refined_prediction,
                ground_truth,
            )

            ###################################################################
            # Refined trajectory loss
            ###################################################################

            refinement_output = (
                self.refinement_loss(
                    refined_prediction,
                    ground_truth,
                )
            )

            if not isinstance(
                refinement_output,
                dict,
            ):

                raise TypeError(
                    "RefinementLoss must return a dictionary."
                )

            if "loss" not in refinement_output:

                raise KeyError(
                    "RefinementLoss output must contain "
                    "the key 'loss'."
                )

            refinement_loss = (
                refinement_output["loss"]
            )

            ###################################################################
            # Preserve additional refinement metrics.
            #
            # IMPORTANT:
            #
            # Do not assume fields such as:
            #
            #     trajectory_history
            #     refinement_score_history
            #
            # exist in RefinedPrediction.
            #
            # The current DSTNet contract contains:
            #
            #     trajectories
            #     scores
            #     offsets
            ###################################################################

            refinement_metrics = {
                key: value
                for key, value in refinement_output.items()
                if key != "loss"
            }

        else:

            ###################################################################
            # Refinement disabled.
            #
            # DSTNet returns:
            #
            #     refined_prediction = None
            #
            # Therefore neither ScoreLoss nor RefinementLoss is evaluated.
            ###################################################################

            score_loss = self._zero_loss(
                prediction.trajectories,
            )

            refinement_loss = self._zero_loss(
                prediction.trajectories,
            )

        #######################################################################
        # Validate individual losses
        #######################################################################

        self._validate_scalar_loss(
            "proposal_loss",
            proposal_loss,
        )

        self._validate_scalar_loss(
            "classification_loss",
            classification_loss,
        )

        self._validate_scalar_loss(
            "score_loss",
            score_loss,
        )

        self._validate_scalar_loss(
            "refinement_loss",
            refinement_loss,
        )

        #######################################################################
        # Validate refinement metrics
        #######################################################################

        for name, value in refinement_metrics.items():

            if not isinstance(
                value,
                Tensor,
            ):

                raise TypeError(
                    f"Refinement metric '{name}' must be "
                    "a torch.Tensor."
                )

            if not torch.isfinite(
                value,
            ).all():

                raise RuntimeError(
                    f"Refinement metric '{name}' contains "
                    "NaN or infinite values."
                )

        #######################################################################
        # Weighted total objective
        #
        # Refinement enabled:
        #
        #     L_total =
        #         L_proposal
        #         + L_classification
        #         + L_refinement
        #         + 0.1 L_score
        #
        # Refinement disabled:
        #
        #     score_loss = 0
        #     refinement_loss = 0
        #
        #     therefore:
        #
        #         L_total =
        #             L_proposal
        #             + L_classification
        #######################################################################

        total_loss = (
            self.proposal_weight
            * proposal_loss
            +
            self.classification_weight
            * classification_loss
            +
            self.refinement_weight
            * refinement_loss
            +
            self.score_weight
            * score_loss
        )

        #######################################################################
        # Validate total loss
        #######################################################################

        self._validate_scalar_loss(
            "loss",
            total_loss,
        )

        #######################################################################
        # Final result
        #######################################################################

        result = {
            "loss": total_loss,
            "proposal_loss": proposal_loss,
            "classification_loss": classification_loss,
            "score_loss": score_loss,
            "refinement_loss": refinement_loss,
        }

        result.update(
            refinement_metrics,
        )

        return result

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "TotalLoss("
            f"proposal={self.proposal_weight}, "
            f"classification={self.classification_weight}, "
            f"score={self.score_weight}, "
            f"refinement={self.refinement_weight}, "
            f"refinement_enabled="
            f"{self.refinement_enabled})"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "TotalLoss",
]
