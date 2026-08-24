"""
models.encoders.encoder

Top-level scene encoder for DSTNet.

Paper
-----
DSTNet, Section III-C

Pipeline
--------

    Agent Embeddings
          Ea
           │
           │
    Map Embeddings
          Em
           │
           ├───────────────┐
           │               │
           ▼               │
    SceneGraph             │
           │               │
           ▼               │
 Relative Spatio-Temporal  │
      Embedding            │
           │               │
           ▼               │
          Er               │
           │               │
           └───────┬───────┘
                   ▼
                  GSTA
                   │
                   ▼
              Z_scene
                   │
                   ▼
             Tri-ATM × L
                   │
                   ▼
                 Z_STM

Tensor notation
---------------

B : batch size
N : number of agents
M : number of map elements
H : observation history length
K : number of prediction modes
D : hidden feature dimension
U : number of SceneGraph edges
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn, Tensor

from datasets.scene_graph_builder import SceneGraph

from models.attention.tri_atm import TriATM
from models.encoders.gsta import GSTA
from models.encoders.relative_spatiotemporal_embeddings import (
    RelativeSpatioTemporalEmbeddingModule,
)


###############################################################################
# Encoder
###############################################################################


class Encoder(nn.Module):
    """
    DSTNet top-level scene encoder.

    The encoder combines the verified Phase-1, Phase-2 and Phase-3
    components:

        1. Relative Spatio-Temporal Embedding
        2. GSTA
        3. Tri-ATM × num_layers

    The encoder does NOT perform:

        - CSV parsing
        - graph construction
        - agent trajectory encoding
        - map encoding
        - prediction decoding
        - trajectory refinement

    Those responsibilities belong to their respective modules.

    Parameters
    ----------
    hidden_dim
        Hidden feature dimension D.

    num_heads
        Number of attention heads.

    num_layers
        Number of stacked Tri-ATM modules.

        The current DSTNet configuration uses 2.

    num_modes
        Number of prediction modes K.

    observation_steps
        Number of observed historical states H.

    interaction_radius
        Maximum spatial interaction radius used by MSPA.

    dropout
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_modes: int = 6,
        observation_steps: int = 20,
        interaction_radius: float = 30.0,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        #######################################################################
        # Configuration
        #######################################################################

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if num_modes <= 0:
            raise ValueError(
                "num_modes must be positive."
            )

        if observation_steps <= 0:
            raise ValueError(
                "observation_steps must be positive."
            )

        if interaction_radius <= 0.0:
            raise ValueError(
                "interaction_radius must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.hidden_dim = int(
            hidden_dim
        )

        self.num_heads = int(
            num_heads
        )

        self.num_layers = int(
            num_layers
        )

        self.num_modes = int(
            num_modes
        )

        self.observation_steps = int(
            observation_steps
        )

        self.interaction_radius = float(
            interaction_radius
        )

        self.dropout = float(
            dropout
        )

        #######################################################################
        # Relative Spatio-Temporal Embedding
        #
        # SceneGraph
        #     │
        #     ▼
        # RelativeSpatioTemporalEmbeddingModule
        #     │
        #     ▼
        # Er
        #
        # Er contains:
        #
        #     edge_index : (2,U)
        #     embeddings : (U,D)
        #     edge_type  : (U,)
        #
        # The current implementation uses the SceneGraph directly.
        #######################################################################

        self.relative_embedding = (
            RelativeSpatioTemporalEmbeddingModule(
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
        )

        #######################################################################
        # Global Spatio-Temporal Attention
        #
        # Ea : (B,N,H,D)
        # Em : (B,M,D)
        # Er : edge-indexed relative representation
        #
        # Output:
        #
        # Z_scene : (B,N,H,K,D)
        #######################################################################

        self.gsta = GSTA(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_modes=self.num_modes,
            observation_steps=self.observation_steps,
            dropout=self.dropout,
        )

        #######################################################################
        # Tri-Attention Spatio-Temporal Modules
        #
        # Each block:
        #
        #     Z_scene
        #        │
        #       MSPA
        #        │
        #       MHCA
        #        │
        #       MMIA
        #        │
        #        ▼
        #      Z_STM
        #
        # The output of one block becomes the input to the next.
        #######################################################################

        self.layers = nn.ModuleList(
            [
                TriATM(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    interaction_radius=self.interaction_radius,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    ###########################################################################
    # Validation
    ###########################################################################

    def _validate_inputs(
        self,
        agent_features: Tensor,
        map_features: Tensor,
        positions: Tensor,
        graph: SceneGraph | Sequence[SceneGraph],
        agent_mask: Tensor | None,
        map_mask: Tensor | None,
    ) -> None:
        """
        Validate encoder inputs before executing the network.
        """

        #######################################################################
        # Agent features
        #######################################################################

        if agent_features.ndim != 4:

            raise ValueError(
                "agent_features must have shape "
                "(B,N,H,D)."
            )

        batch_size, num_agents, history_steps, hidden_dim = (
            agent_features.shape
        )

        if hidden_dim != self.hidden_dim:

            raise ValueError(
                "agent_features hidden dimension mismatch. "
                f"Expected {self.hidden_dim}, "
                f"got {hidden_dim}."
            )

        if history_steps != self.observation_steps:

            raise ValueError(
                "agent_features history dimension mismatch. "
                f"Expected {self.observation_steps}, "
                f"got {history_steps}."
            )

        #######################################################################
        # Map features
        #######################################################################

        if map_features.ndim != 3:

            raise ValueError(
                "map_features must have shape "
                "(B,M,D)."
            )

        if map_features.shape[0] != batch_size:

            raise ValueError(
                "agent_features and map_features must have "
                "the same batch size."
            )

        if map_features.shape[-1] != self.hidden_dim:

            raise ValueError(
                "map_features hidden dimension mismatch. "
                f"Expected {self.hidden_dim}, "
                f"got {map_features.shape[-1]}."
            )

        #######################################################################
        # Positions
        #######################################################################

        if positions.ndim != 3:

            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        if positions.shape != (
            batch_size,
            num_agents,
            2,
        ):

            raise ValueError(
                "positions must have shape "
                "(B,N,2) matching agent_features."
            )

        #######################################################################
        # Agent mask
        #######################################################################

        if agent_mask is not None:

            if agent_mask.ndim != 2:

                raise ValueError(
                    "agent_mask must have shape (B,N)."
                )

            if agent_mask.shape != (
                batch_size,
                num_agents,
            ):

                raise ValueError(
                    "agent_mask must have shape "
                    "(B,N) matching agent_features."
                )

        #######################################################################
        # Lane mask
        #######################################################################

        if map_mask is not None:

            if map_mask.ndim != 2:

                raise ValueError(
                    "map_mask must have shape (B,M)."
                )

            if map_mask.shape != (
                batch_size,
                map_features.shape[1],
            ):

                raise ValueError(
                    "map_mask must have shape "
                    "(B,M) matching map_features."
                )

        #######################################################################
        # SceneGraph
        #######################################################################

        if isinstance(
            graph,
            SceneGraph,
        ):

            return

        if isinstance(
            graph,
            Sequence,
        ):

            if len(graph) != batch_size:

                raise ValueError(
                    "When graph is a sequence, it must contain "
                    f"exactly {batch_size} SceneGraph objects; "
                    f"got {len(graph)}."
                )

            for index, scene_graph in enumerate(graph):

                if not isinstance(
                    scene_graph,
                    SceneGraph,
                ):

                    raise TypeError(
                        "graph sequence element "
                        f"{index} is not a SceneGraph."
                    )

            return

        raise TypeError(
            "graph must be a SceneGraph or a sequence of "
            "SceneGraph objects."
        )

    ###########################################################################
    # Relative Embedding
    ###########################################################################

    def _build_relative_embeddings(
        self,
        graph: SceneGraph | Sequence[SceneGraph],
        batch_size: int,
    ):
        """
        Build Er for one scene or a batch of scenes.

        The relative embedding module operates on one SceneGraph
        at a time. For B > 1, construct one RelativeSpatioTemporalEmbedding
        object per scene and pass the resulting sequence to GSTA.
        """

        if isinstance(
            graph,
            SceneGraph,
        ):

            if batch_size != 1:

                raise ValueError(
                    "A single SceneGraph can only be used with "
                    "batch size 1. Provide one SceneGraph per "
                    "batch element for B > 1."
                )

            return self.relative_embedding(
                graph,
            )

        return [
            self.relative_embedding(
                scene_graph,
            )
            for scene_graph in graph
        ]

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        agent_features: Tensor,
        map_features: Tensor,
        positions: Tensor,
        graph: SceneGraph | Sequence[SceneGraph],
        *,
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Encode a complete scene.

        Parameters
        ----------
        agent_features
            Agent embeddings Ea.

            Shape:

                (B,N,H,D)

        map_features
            Map embeddings Em.

            Shape:

                (B,M,D)

        positions
            Current agent positions used by MSPA.

            Shape:

                (B,N,2)

        graph
            Precomputed SceneGraph.

            For B=1:

                SceneGraph

            For B>1:

                Sequence[SceneGraph]

        agent_mask
            Valid-agent mask.

            Shape:

                (B,N)

        map_mask
            Valid-map mask.

            Shape:

                (B,M)

            It is passed to GSTA. The current Tri-ATM operates
            only on the agent/mode scene representation, so the
            lane mask is not passed into Tri-ATM.

        Returns
        -------
        Tensor

            Z_STM

            Shape:

                (B,N,H,K,D)
        """

        #######################################################################
        # Validate inputs
        #######################################################################

        self._validate_inputs(
            agent_features=agent_features,
            map_features=map_features,
            positions=positions,
            graph=graph,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

        batch_size = agent_features.shape[0]

        num_agents = agent_features.shape[1]

        history_steps = agent_features.shape[2]

        #######################################################################
        # Build relative spatio-temporal embeddings
        #######################################################################

        relative = self._build_relative_embeddings(
            graph=graph,
            batch_size=batch_size,
        )

        #######################################################################
        # GSTA
        #
        # Ea + Em + Er
        #       │
        #       ▼
        #      GSTA
        #       │
        #       ▼
        # Z_scene
        #
        # Use the direct return value from GSTA rather than relying
        # on the cached scene_prediction_embeddings property.
        #######################################################################

        scene_embeddings = self.gsta(
            Ea=agent_features,
            Em=map_features,
            Er=relative,
            scene_graph=graph,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

        #######################################################################
        # Validate GSTA output
        #######################################################################

        expected_shape = (
            batch_size,
            num_agents,
            history_steps,
            self.num_modes,
            self.hidden_dim,
        )

        if scene_embeddings.shape != expected_shape:

            raise RuntimeError(
                "GSTA produced an unexpected scene representation. "
                f"Expected {expected_shape}, "
                f"got {tuple(scene_embeddings.shape)}."
            )

        #######################################################################
        # Tri-ATM stack
        #
        # Z_scene
        #     ↓
        #    MSPA
        #     ↓
        #    MHCA
        #     ↓
        #    MMIA
        #     ↓
        #   Z_STM
        #
        # Current TriATM requires actual agent positions for MSPA.
        #######################################################################

        for layer_index, layer in enumerate(
            self.layers
        ):

            scene_embeddings = layer(
                scene_embeddings,
                positions,
                agent_mask=agent_mask,
            )

            ###################################################################
            # Validate every layer.
            ###################################################################

            if scene_embeddings.shape != expected_shape:

                raise RuntimeError(
                    "Tri-ATM layer "
                    f"{layer_index} changed the expected "
                    "scene representation shape. "
                    f"Expected {expected_shape}, "
                    f"got {tuple(scene_embeddings.shape)}."
                )

        #######################################################################
        # Final output
        #######################################################################

        return scene_embeddings

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"modes={self.num_modes}, "
            f"history={self.observation_steps}, "
            f"layers={self.num_layers}, "
            f"interaction_radius={self.interaction_radius}"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "Encoder",
]
