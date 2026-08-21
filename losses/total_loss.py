"""
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

The current implementation uses:

    lambda = 0.1

Current tensor contracts
------------------------

Prediction:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

RefinedPrediction:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

    refinement_scores
        (B,N,H,K)

    trajectory_history
        (B,N,H,K,C+1,T,2)

    refinement_score_history
        (B,N,H,K,C+1)
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


class TotalLoss(nn.Module):
    """
    Complete DSTNet training objective.

    Parameters
    ----------
    proposal_weight
        Weight applied to the proposal loss.

    classification_weight
        Weight applied to the classification loss.

    score_weight
        Weight lambda applied to ScoreLoss.

        Default:

            0.1

    refinement_weight
        Weight applied to the refinement loss.

    Notes
    -----
    With the default values, the objective is:

        L_total =
            L_proposal
            + L_classification
            + L_refinement
            + 0.1 L_score
    """

    def __init__(
        self,
        proposal_weight: float = 1.0,
        classification_weight: float = 1.0,
        score_weight: float = 0.1,
        refinement_weight: float = 1.0,
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
    # Forward
    ###########################################################################

    def forward(
        self,
        prediction: Prediction,
        refined_prediction: RefinedPrediction,
        ground_truth: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute the complete DSTNet objective.

        Parameters
        ----------
        prediction
            Coarse decoder prediction.

        refined_prediction
            Output of the anchor-based refinement module.

        ground_truth
            Future ground-truth trajectories.

            Shape:

                (B,N,T,2)

        Returns
        -------
        dict[str, Tensor]

            Contains:

                loss
                proposal_loss
                classification_loss
                score_loss
                refinement_loss

            plus the individual refinement-loss metrics.
        """

        #######################################################################
        # Validate inputs
        #######################################################################

        if not isinstance(
            prediction,
            Prediction,
        ):
            raise TypeError(
                "prediction must be a Prediction."
            )

        if not isinstance(
            refined_prediction,
            RefinedPrediction,
        ):
            raise TypeError(
                "refined_prediction must be a RefinedPrediction."
            )

        if not isinstance(
            ground_truth,
            torch.Tensor,
        ):
            raise TypeError(
                "ground_truth must be a torch.Tensor."
            )

        #######################################################################
        # Proposal loss
        #
        # Eq. (28):
        #
        #     k_best = argmin endpoint error
        #
        # Eq. (30):
        #
        #     L_proposal =
        #         Huber(Y^(0)_kbest, G)
        #######################################################################

        proposal_loss, best_mode = (
            self.proposal_loss(
                prediction,
                ground_truth,
                return_best_mode=True,
            )
        )

        #######################################################################
        # Classification loss
        #
        # Eq. (32)
        #######################################################################

        classification_loss = (
            self.classification_loss(
                prediction,
                best_mode,
            )
        )

        #######################################################################
        # Refinement score loss
        #
        # Eq. (26) + Eq. (33)
        #
        # This must operate on RefinedPrediction because the required
        # refinement trajectory and score histories are stored there.
        #######################################################################

        score_loss = self.score_loss(
            refined_prediction,
            ground_truth,
        )

        #######################################################################
        # Refined trajectory loss
        #######################################################################

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

        #######################################################################
        # Weighted total objective
        #
        # Default:
        #
        #     L_total =
        #         L_proposal
        #         + L_classification
        #         + L_refinement
        #         + 0.1 L_score
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
        # Validate all scalar losses
        #######################################################################

        components = {
            "loss": total_loss,
            "proposal_loss": proposal_loss,
            "classification_loss": classification_loss,
            "score_loss": score_loss,
            "refinement_loss": refinement_loss,
        }

        for name, value in components.items():

            if not isinstance(
                value,
                torch.Tensor,
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

        #######################################################################
        # Include refinement metrics
        #######################################################################

        metrics = {
            key: value
            for key, value in refinement_output.items()
            if key != "loss"
        }

        #######################################################################
        # Final result
        #######################################################################

        return {
            "loss": total_loss,
            "proposal_loss": proposal_loss,
            "classification_loss": classification_loss,
            "score_loss": score_loss,
            "refinement_loss": refinement_loss,
            **metrics,
        }

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
            f"refinement={self.refinement_weight})"
        )


__all__ = [
    "TotalLoss",
]
