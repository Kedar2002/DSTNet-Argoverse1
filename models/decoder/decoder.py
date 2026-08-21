"""
models.decoder.decoder

Multimodal trajectory decoder for DSTNet.

Paper
-----
DSTNet, Section III-E — Multimodal Trajectory Output
Equation (25)

The paper specifies:

    Y^(0)_{n,t,k} = MLP(Z^STM_{n,t,k})

where:

    Z^STM_{n,t,k} ∈ R^D

and:

    Y^(0)_{n,t,k} ∈ R^(T_future × 2)

The decoder therefore operates independently on every:

    (batch, agent, historical_time, mode)

prediction embedding.

Current model contract
----------------------

The decoder returns a Prediction object containing:

    trajectories
        (B,N,H,K,T,2)

    probabilities
        (B,N,H,K)

The trajectory generation follows Eq. (25).

The probability head is a minimal implementation required by
the current Prediction data contract. The supplied DSTNet paper
section does not specify the exact probability-head architecture,
so it should not be interpreted as an exact reproduction of an
unstated architectural detail.

Input
-----

Z_STM

    (B,N,H,K,D)

Output
------

Prediction

    trajectories:
        (B,N,H,K,T,2)

    probabilities:
        (B,N,H,K)

Notes
-----

1. The historical dimension H is deliberately preserved.

2. The trajectory decoder is a shared two-layer MLP.

3. No trajectory-specific attention is introduced here.

4. No residual connection or LayerNorm is added around the
   trajectory MLP because these are not specified by Eq. (25).

5. The K prediction modes remain explicitly represented.

6. The decoder produces multimodal trajectory probabilities
   separately from the trajectory coordinates.

7. The decoder does not perform anchor-based refinement.
   Refinement is handled by the subsequent refinement module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.layers.mlp import MLP
from models.model_types import Prediction


###############################################################################
# Decoder
###############################################################################


class Decoder(nn.Module):
    """
    DSTNet multimodal trajectory decoder.

    Trajectory branch
    -----------------

    Implements Eq. (25):

        Y^(0)_{n,t,k} = MLP(Z^STM_{n,t,k})

    The same MLP parameters are shared across:

        • agents
        • historical timesteps
        • prediction modes

    Probability branch
    ------------------

    A separate lightweight projection produces one logit for
    each prediction mode. Softmax is applied across K.

    This probability branch is an implementation requirement of
    the current Prediction type contract. The exact probability
    head architecture is not specified by the supplied DSTNet
    paper text.

    Parameters
    ----------
    hidden_dim
        Hidden feature dimension D.

    prediction_steps
        Number of future trajectory points T.

    dropout
        Dropout probability inside the trajectory MLP.

        The paper specifies a two-layer MLP but does not explicitly
        specify decoder dropout. Therefore the default is 0.0.

    Attributes
    ----------
    hidden_dim
        Decoder input feature dimension.

    prediction_steps
        Number of future prediction steps.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        prediction_steps: int = 30,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        #######################################################################
        # Configuration validation
        #######################################################################

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if prediction_steps <= 0:
            raise ValueError(
                "prediction_steps must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.hidden_dim = hidden_dim
        self.prediction_steps = prediction_steps

        #######################################################################
        # Trajectory MLP
        #
        # Eq. (25):
        #
        #     Y^(0) = MLP(Z^STM)
        #
        # The MLP performs:
        #
        #     D
        #     ↓
        #     D
        #     ↓
        #     T × 2
        #
        # Because the MLP operates on the final dimension, the
        # same parameters are shared across every:
        #
        #     (B,N,H,K)
        #
        # prediction embedding.
        #######################################################################

        self.decoder = MLP(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim],
            output_dim=prediction_steps * 2,
            dropout=dropout,
        )

        #######################################################################
        # Multimodal probability head
        #
        # Input:
        #
        #     (B,N,H,K,D)
        #
        # Output:
        #
        #     (B,N,H,K,1)
        #
        # which is squeezed to:
        #
        #     (B,N,H,K)
        #
        # Softmax is then applied over K.
        #
        # This is a minimal implementation of the probability
        # branch required by the current Prediction contract.
        #######################################################################

        self.probability_head = nn.Linear(
            hidden_dim,
            1,
        )

    ###########################################################################
    # Input Validation
    ###########################################################################

    def _validate_input(
        self,
        z_stm: Tensor,
    ) -> None:
        """
        Validate Z_STM before decoding.
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
                "Expected z_stm with shape "
                "(B,N,H,K,D), "
                f"got {tuple(z_stm.shape)}."
            )

        feature_dim = z_stm.shape[-1]

        if feature_dim != self.hidden_dim:
            raise ValueError(
                "Decoder hidden dimension mismatch: "
                f"expected {self.hidden_dim}, "
                f"got {feature_dim}."
            )

        if not torch.is_floating_point(
            z_stm,
        ):
            raise TypeError(
                "z_stm must contain floating-point features."
            )

        if not torch.isfinite(
            z_stm,
        ).all():
            raise ValueError(
                "z_stm contains NaN or infinite values."
            )

    ###########################################################################
    # Trajectory decoding
    ###########################################################################

    def _decode_trajectories(
        self,
        z_stm: Tensor,
    ) -> Tensor:
        """
        Generate coarse trajectory proposals.

        Parameters
        ----------
        z_stm
            Shape:

                (B,N,H,K,D)

        Returns
        -------
        Tensor

            Shape:

                (B,N,H,K,T,2)
        """

        B = z_stm.shape[0]
        N = z_stm.shape[1]
        H = z_stm.shape[2]
        K = z_stm.shape[3]

        #######################################################################
        # Apply shared trajectory MLP
        #
        # (B,N,H,K,D)
        #
        #     ↓
        #
        # (B,N,H,K,T×2)
        #######################################################################

        coarse_flat = self.decoder(
            z_stm,
        )

        #######################################################################
        # Restore trajectory structure
        #
        # (B,N,H,K,T×2)
        #
        #     ↓
        #
        # (B,N,H,K,T,2)
        #######################################################################

        trajectories = coarse_flat.reshape(
            B,
            N,
            H,
            K,
            self.prediction_steps,
            2,
        )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.isfinite(
            trajectories,
        ).all():
            raise RuntimeError(
                "Decoder produced NaN or infinite "
                "trajectory values."
            )

        return trajectories

    ###########################################################################
    # Probability decoding
    ###########################################################################

    def _decode_probabilities(
        self,
        z_stm: Tensor,
    ) -> Tensor:
        """
        Generate multimodal trajectory probabilities.

        Parameters
        ----------
        z_stm
            Shape:

                (B,N,H,K,D)

        Returns
        -------
        Tensor

            Shape:

                (B,N,H,K)

        Notes
        -----
        One scalar logit is generated for each prediction mode.

        Softmax is applied across K so that, for every:

            (B,N,H)

        combination,

            sum_k probability_k = 1.
        """

        #######################################################################
        # Mode logits
        #
        # (B,N,H,K,D)
        #
        #     ↓
        #
        # (B,N,H,K,1)
        #######################################################################

        logits = self.probability_head(
            z_stm,
        )

        #######################################################################
        # Remove final singleton dimension
        #
        # (B,N,H,K,1)
        #
        #     ↓
        #
        # (B,N,H,K)
        #######################################################################

        logits = logits.squeeze(
            dim=-1,
        )

        #######################################################################
        # Normalize across prediction modes
        #######################################################################

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        #######################################################################
        # Numerical validation
        #######################################################################

        if not torch.isfinite(
            probabilities,
        ).all():
            raise RuntimeError(
                "Decoder produced NaN or infinite "
                "trajectory probabilities."
            )

        #######################################################################
        # Probability validation
        #######################################################################

        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(
                probabilities.sum(dim=-1),
            ),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Decoder trajectory probabilities do not "
                "sum to one across prediction modes."
            )

        return probabilities

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        z_stm: Tensor,
    ) -> Prediction:
        """
        Decode coarse multimodal trajectories.

        Parameters
        ----------
        z_stm
            DSTNet Tri-ATM output.

            Shape:

                (B,N,H,K,D)

        Returns
        -------
        Prediction

            trajectories:

                (B,N,H,K,T,2)

            probabilities:

                (B,N,H,K)
        """

        #######################################################################
        # Input validation
        #######################################################################

        self._validate_input(
            z_stm,
        )

        #######################################################################
        # Coarse trajectories
        #######################################################################

        trajectories = (
            self._decode_trajectories(
                z_stm,
            )
        )

        #######################################################################
        # Multimodal probabilities
        #######################################################################

        probabilities = (
            self._decode_probabilities(
                z_stm,
            )
        )

        #######################################################################
        # Construct typed prediction
        #######################################################################

        prediction = Prediction(
            trajectories=trajectories,
            probabilities=probabilities,
        )

        return prediction

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"prediction_steps={self.prediction_steps}"
        )


###############################################################################
# Public API
###############################################################################


__all__ = [
    "Decoder",
]
