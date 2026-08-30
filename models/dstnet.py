"""
models.dstnet

Complete DSTNet model.

Pipeline
--------
Observed Agent Trajectories
Observed Lane Centerlines
        |
        +------------------+
        |                  |
        v                  v
   AgentEncoder       MapEncoder
        |                  |
        +--------+---------+
                 |
                 v
              Encoder
        (GSTA + Tri-ATM)
                 |
                 v
              Z_STM
        (B,N,H,K,D)
                 |
                 v
              Decoder
                 |
                 v
              Y^(0)
        (B,N,H,K,T,2)
                 |
                 v
        Adaptive Anchor
           Refinement
                 |
                 v
        RefinedPrediction
    trajectories + scores + offsets

Notes
-----
The current implementation follows the interfaces verified in the
Phase-1 through Phase-5 integration tests.

In particular:

1. AgentEncoder produces agent embeddings.
2. MapEncoder produces lane/map embeddings.
3. Encoder performs the complete GSTA + Tri-ATM stage and directly
   returns Z_STM with shape:

       (B, N, H, K, D)

4. No separate ModeEmbedding is applied here because the current
   Encoder already produces the multimodal K dimension.

5. Decoder directly consumes Z_STM and produces Prediction:

       trajectories:
           (B, N, H, K, T, 2)

       probabilities:
           (B, N, H, K)

6. Refinement consumes Z_STM and Prediction and produces:

       trajectories:
           (B, N, H, K, T, 2)

       scores:
           (B, N, H, K)

       offsets:
           (B, N, H, K, T, 2)

7. Refinement can be disabled for memory/stability testing. When
   refinement is disabled, the model returns ``None`` for the refined
   prediction and does not attempt to validate or access refinement
   outputs.

The top-level model therefore acts only as the composition layer.
It does not duplicate architectural operations already implemented
inside Encoder, Decoder, or Refinement.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from datasets.scene_data import SceneGraph

from models.decoder.decoder import Decoder
from models.encoders.agent_encoder import AgentEncoder
from models.encoders.encoder import Encoder
from models.encoders.map_encoder import MapEncoder
from models.model_types import Prediction, RefinedPrediction
from models.refinement.refinement import Refinement


###############################################################################
# DSTNet
###############################################################################


class DSTNet(nn.Module):
    """
    Complete DSTNet model.

    Parameters
    ----------
    observation_steps:
        Number of observed historical trajectory steps.

    prediction_steps:
        Number of future trajectory steps.

    map_points:
        Number of sampled points used for each lane centerline.

    hidden_dim:
        Common feature dimension D.

    num_heads:
        Number of attention heads used by the Encoder and Refinement.

    num_encoder_layers:
        Number of Tri-ATM layers used by the Encoder.

    num_modes:
        Number of multimodal trajectory hypotheses K.

    interaction_radius:
        Maximum spatial interaction radius used by the Encoder.

    refinement_iterations:
        Number of refinement cycles.

    r_min:
        Minimum adaptive interaction radius.

    r_max:
        Maximum adaptive interaction radius.

    radius_hidden_dim:
        Hidden dimension used by the adaptive-radius predictor.

    refinement_enabled:
        Whether anchor-based trajectory refinement is enabled.

    dropout:
        Dropout probability used by the learnable modules.
    """

    def __init__(
        self,
        observation_steps: int = 20,
        prediction_steps: int = 30,
        map_points: int = 20,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_modes: int = 6,
        interaction_radius: float = 30.0,
        refinement_iterations: int = 2,
        r_min: float = 5.0,
        r_max: float | None = None,
        radius_hidden_dim: int | None = None,
        refinement_enabled: bool = True,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        #######################################################################
        # Configuration validation
        #######################################################################

        if observation_steps <= 0:
            raise ValueError(
                "observation_steps must be positive."
            )

        if prediction_steps <= 0:
            raise ValueError(
                "prediction_steps must be positive."
            )

        if map_points <= 0:
            raise ValueError(
                "map_points must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        if num_encoder_layers <= 0:
            raise ValueError(
                "num_encoder_layers must be positive."
            )

        if num_modes <= 0:
            raise ValueError(
                "num_modes must be positive."
            )

        if interaction_radius <= 0.0:
            raise ValueError(
                "interaction_radius must be positive."
            )

        if refinement_iterations <= 0:
            raise ValueError(
                "refinement_iterations must be positive."
            )

        if r_min <= 0.0:
            raise ValueError(
                "r_min must be positive."
            )

        if r_max is not None and r_max <= r_min:
            raise ValueError(
                "r_max must be greater than r_min."
            )

        if radius_hidden_dim is not None and radius_hidden_dim <= 0:
            raise ValueError(
                "radius_hidden_dim must be positive when provided."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
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

        self.observation_steps = observation_steps
        self.prediction_steps = prediction_steps
        self.map_points = map_points
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_modes = num_modes
        self.interaction_radius = interaction_radius
        self.refinement_iterations = refinement_iterations
        self.r_min = r_min
        self.r_max = r_max
        self.radius_hidden_dim = radius_hidden_dim
        self.refinement_enabled = refinement_enabled
        self.dropout = dropout

        #######################################################################
        # Local Agent Encoder
        #######################################################################

        self.agent_encoder = AgentEncoder(
            observation_steps=observation_steps,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Local Map Encoder
        #######################################################################

        self.map_encoder = MapEncoder(
            num_points=map_points,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Global Encoder
        #
        # Current Encoder contract:
        #
        #     Ea
        #     Em
        #      |
        #      v
        #    Encoder
        #      |
        #      v
        #    Z_STM
        #
        # Shape:
        #
        #     (B,N,H,K,D)
        #######################################################################

        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            num_modes=num_modes,
            observation_steps=observation_steps,
            interaction_radius=interaction_radius,
            r_min=r_min,
            r_max=r_max,
            radius_hidden_dim=radius_hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Coarse Multimodal Decoder
        #######################################################################

        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            prediction_steps=prediction_steps,
            dropout=dropout,
        )

        #######################################################################
        # Anchor-Based Refinement
        #######################################################################

        if self.refinement_enabled:

            self.refinement = Refinement(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                prediction_steps=prediction_steps,
                refinement_iterations=refinement_iterations,
                dropout=dropout,
            )

        else:

            self.refinement = None

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        agent_trajectories: Tensor,
        map_centerlines: Tensor,
        positions: Tensor,
        graph: SceneGraph | Sequence[SceneGraph],
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> tuple[
        Prediction,
        RefinedPrediction | None,
    ]:
        """
        Run the complete DSTNet forward pass.

        Parameters
        ----------
        agent_trajectories:
            Observed agent trajectories.

            Shape:

                (B,N,H,2)

        map_centerlines:
            Sampled lane centerlines.

            Shape:

                (B,M,P,2)

        positions:
            Current agent positions.

            Shape:

                (B,N,2)

        graph:
            SceneGraph or sequence of SceneGraph objects.

        agent_mask:
            Optional validity mask for agents.

            Shape:

                (B,N)

        map_mask:
            Optional validity mask for lanes.

            Shape:

                (B,M)

        Returns
        -------
        tuple[Prediction, RefinedPrediction | None]

            coarse_prediction:
                Decoder output containing coarse multimodal
                trajectories and their probabilities.

            refined_prediction:
                Final anchor-refined trajectories, scores,
                and offsets when refinement is enabled.

                ``None`` when refinement is disabled.
        """

        #######################################################################
        # Input validation
        #######################################################################

        if not isinstance(
            agent_trajectories,
            Tensor,
        ):
            raise TypeError(
                "agent_trajectories must be a torch.Tensor."
            )

        if not isinstance(
            map_centerlines,
            Tensor,
        ):
            raise TypeError(
                "map_centerlines must be a torch.Tensor."
            )

        if not isinstance(
            positions,
            Tensor,
        ):
            raise TypeError(
                "positions must be a torch.Tensor."
            )

        if agent_trajectories.ndim != 4:
            raise ValueError(
                "agent_trajectories must have shape "
                "(B,N,H,2), "
                f"got {tuple(agent_trajectories.shape)}."
            )

        if map_centerlines.ndim != 4:
            raise ValueError(
                "map_centerlines must have shape "
                "(B,M,P,2), "
                f"got {tuple(map_centerlines.shape)}."
            )

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape "
                "(B,N,2), "
                f"got {tuple(positions.shape)}."
            )

        if agent_trajectories.shape[0] != map_centerlines.shape[0]:
            raise ValueError(
                "Batch dimension mismatch between "
                "agent_trajectories and map_centerlines: "
                f"{agent_trajectories.shape[0]} vs "
                f"{map_centerlines.shape[0]}."
            )

        if agent_trajectories.shape[0] != positions.shape[0]:
            raise ValueError(
                "Batch dimension mismatch between "
                "agent_trajectories and positions: "
                f"{agent_trajectories.shape[0]} vs "
                f"{positions.shape[0]}."
            )

        if agent_trajectories.shape[1] != positions.shape[1]:
            raise ValueError(
                "Agent dimension mismatch between "
                "agent_trajectories and positions: "
                f"{agent_trajectories.shape[1]} vs "
                f"{positions.shape[1]}."
            )

        #######################################################################
        # Local feature extraction
        #######################################################################

        agent_features = self.agent_encoder(
            agent_trajectories,
        )

        map_features = self.map_encoder(
            map_centerlines,
        )

        #######################################################################
        # Global scene encoding
        #
        # Encoder performs:
        #
        #     relative spatiotemporal representation
        #              +
        #             GSTA
        #              +
        #           Tri-ATM
        #
        # and directly returns:
        #
        #     Z_STM = (B,N,H,K,D)
        #######################################################################

        z_stm = self.encoder(
            agent_features=agent_features,
            map_features=map_features,
            positions=positions,
            graph=graph,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

        #######################################################################
        # Defensive shape validation
        #######################################################################

        expected_shape = (
            agent_trajectories.shape[0],
            agent_trajectories.shape[1],
            agent_trajectories.shape[2],
            self.num_modes,
            self.hidden_dim,
        )

        if tuple(z_stm.shape) != expected_shape:
            raise RuntimeError(
                "Encoder produced an unexpected Z_STM shape: "
                f"expected {expected_shape}, "
                f"got {tuple(z_stm.shape)}."
            )

        if not torch.isfinite(
            z_stm
        ).all():
            raise FloatingPointError(
                "Encoder produced NaN or infinite values."
            )

        #######################################################################
        # Coarse multimodal decoding
        #
        #     Z_STM
        #       |
        #       v
        #     Decoder
        #       |
        #       v
        #     Prediction
        #######################################################################

        coarse_prediction = self.decoder(
            z_stm,
        )

        #######################################################################
        # Coarse prediction validation
        #######################################################################

        if not torch.isfinite(
            coarse_prediction.trajectories
        ).all():
            raise FloatingPointError(
                "Decoder produced non-finite trajectories."
            )

        if not torch.isfinite(
            coarse_prediction.probabilities
        ).all():
            raise FloatingPointError(
                "Decoder produced non-finite probabilities."
            )

        #######################################################################
        # Anchor-based trajectory refinement
        #
        # Refinement is deliberately isolated behind the
        # refinement_enabled flag.
        #
        # This allows the training pipeline to disable refinement
        # while diagnosing Encoder/GSTA/ARP-MSPA numerical stability
        # and GPU memory usage.
        #######################################################################

        if self.refinement_enabled:

            if self.refinement is None:
                raise RuntimeError(
                    "refinement_enabled=True but the Refinement "
                    "module was not constructed."
                )

            refined_prediction = self.refinement(
                z_stm=z_stm,
                prediction=coarse_prediction,
            )

            ###################################################################
            # Refined prediction validation
            ###################################################################

            if not torch.isfinite(
                refined_prediction.trajectories
            ).all():
                raise FloatingPointError(
                    "Refinement produced non-finite trajectories."
                )

            if not torch.isfinite(
                refined_prediction.scores
            ).all():
                raise FloatingPointError(
                    "Refinement produced non-finite scores."
                )

            if not torch.isfinite(
                refined_prediction.offsets
            ).all():
                raise FloatingPointError(
                    "Refinement produced non-finite offsets."
                )

        else:

            refined_prediction = None

        #######################################################################
        # Return
        #######################################################################

        return (
            coarse_prediction,
            refined_prediction,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"observation_steps={self.observation_steps}, "
            f"prediction_steps={self.prediction_steps}, "
            f"map_points={self.map_points}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_encoder_layers={self.num_encoder_layers}, "
            f"num_modes={self.num_modes}, "
            f"interaction_radius={self.interaction_radius}, "
            f"refinement_iterations="
            f"{self.refinement_iterations}, "
            f"refinement_enabled="
            f"{self.refinement_enabled}"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "DSTNet",
]
