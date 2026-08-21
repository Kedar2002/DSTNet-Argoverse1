"""
models.encoders.gsta

Global Spatio-Temporal Attention (GSTA) for DSTNet.

Paper
-----
DSTNet, Section III-C.

Paper computation flow
----------------------

        Ea       Em       Er
         │        │        │
         └────┬───┴───┬────┘
              │       │
              ▼       ▼
             ZT       ZS
              │       │
              ▼       ▼
        Temporal    Spatial
        Self-Attn   Self-Attn
              │       │
              └───┬───┘
                  │
          Symmetric Cross-Attn
             Eq. (5)-(6)
                  │
             ┌────┴────┐
             ▼         ▼
            qT         qS
             │         │
             └────┬────┘
                  ▼
             Z_scene
               Eq. (9)

Tensor convention
-----------------

B : batch size
N : number of agents
H : observation history length
M : number of map nodes
K : number of prediction modes
D : hidden dimension

Input
-----

Ea
    (B,N,H,D)

Em
    (B,M,D)

Er
    RelativeSpatioTemporalEmbedding

Output
------

Z_scene
    (B,N,H,K,D)

Important
---------
The paper specifies the GSTA computation flow and attention
equations, but does not explicitly specify the exact operation
used to convert edge-indexed Er into node-associated temporal
and spatial features.

This implementation therefore uses mean aggregation of incident
edge embeddings onto graph nodes before the attention stages.
That is an implementation detail, not a claim that the paper
explicitly specifies mean aggregation.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from datasets.scene_graph_builder import SceneGraph
from models.model_types import RelativeSpatioTemporalEmbedding


###############################################################################
# GSTA
###############################################################################


class GSTA(nn.Module):
    """
    Global Spatio-Temporal Attention.

    Implements the GSTA computation flow described in Section III-C
    of the DSTNet paper.

    Parameters
    ----------
    hidden_dim
        Hidden feature dimension D.

    num_heads
        Number of attention heads.

    num_modes
        Number of prediction modes K.

    observation_steps
        Number of observed historical states H.

    dropout
        Attention dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_modes: int = 6,
        observation_steps: int = 20,
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

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        if num_modes <= 0:
            raise ValueError(
                "num_modes must be positive."
            )

        if observation_steps <= 0:
            raise ValueError(
                "observation_steps must be positive."
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

        self.num_modes = int(
            num_modes
        )

        self.observation_steps = int(
            observation_steps
        )

        #######################################################################
        # Temporal Self-Attention
        #
        # Eq. (3)
        #
        # Z_i^T = MHA(Z_i^T, Z_i^T, Z_i^T)
        #
        # Attention is performed independently for each agent across
        # its historical states.
        #######################################################################

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_norm = nn.LayerNorm(
            hidden_dim
        )

        #######################################################################
        # Spatial Self-Attention
        #
        # Eq. (4)
        #
        # Z_j^S = MHA(Z_j^S, Z_j^S, Z_j^S)
        #
        # Attention is performed across map elements.
        #######################################################################

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.spatial_norm = nn.LayerNorm(
            hidden_dim
        )

        #######################################################################
        # Temporal -> Spatial Cross-Attention
        #
        # Eq. (5)
        #
        # Z_i^T = MHA(Z_i^T, Z_j^S, Z_j^S)
        #######################################################################

        self.temporal_to_spatial = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_cross_norm = nn.LayerNorm(
            hidden_dim
        )

        #######################################################################
        # Spatial -> Temporal Cross-Attention
        #
        # Eq. (6)
        #
        # Z_j^S = MHA(Z_j^S, Z_i^T, Z_i^T)
        #######################################################################

        self.spatial_to_temporal = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.spatial_cross_norm = nn.LayerNorm(
            hidden_dim
        )

        #######################################################################
        # Learnable Temporal Query Bank
        #
        # q^T_(n,t,k)
        #
        # The paper defines one learnable query for every:
        #
        #     agent × historical timestep × prediction mode
        #
        # The agent dimension is represented by broadcasting the same
        # learnable query bank over all agents.
        #######################################################################

        self.temporal_queries = nn.Parameter(
            torch.empty(
                observation_steps,
                num_modes,
                hidden_dim,
            )
        )

        #######################################################################
        # Learnable Spatial Query Bank
        #
        # q^S_(n,t,k)
        #######################################################################

        self.spatial_queries = nn.Parameter(
            torch.empty(
                observation_steps,
                num_modes,
                hidden_dim,
            )
        )

        #######################################################################
        # Query -> Temporal Feature Attention
        #######################################################################

        self.temporal_query_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        #######################################################################
        # Query -> Spatial Feature Attention
        #######################################################################

        self.spatial_query_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        #######################################################################
        # Cached scene representation
        #######################################################################

        self._scene_prediction_embeddings: Tensor | None = None

        #######################################################################
        # Parameter initialization
        #######################################################################

        self.reset_parameters()

    ###########################################################################
    # Initialization
    ###########################################################################

    def reset_parameters(
        self,
    ) -> None:
        """
        Initialize learnable query parameters.
        """

        nn.init.xavier_uniform_(
            self.temporal_queries
        )

        nn.init.xavier_uniform_(
            self.spatial_queries
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        *,
        Ea: Tensor,
        Em: Tensor,
        Er: (
            RelativeSpatioTemporalEmbedding
            | Sequence[RelativeSpatioTemporalEmbedding]
        ),
        scene_graph: (
            SceneGraph
            | Sequence[SceneGraph]
        ),
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Apply GSTA.

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
            Relative spatio-temporal embeddings.

            One object per batch scene.

        scene_graph
            SceneGraph corresponding to each Er.

        agent_mask
            Shape:

                (B,N)

        map_mask
            Shape:

                (B,M)

        Returns
        -------
        Tensor

            Z_scene

            Shape:

                (B,N,H,K,D)
        """

        self._validate_inputs(
            Ea=Ea,
            Em=Em,
        )

        batch_size = Ea.shape[0]

        graphs = self._as_sequence(
            scene_graph,
            batch_size,
            "scene_graph",
        )

        relative_embeddings = self._as_sequence(
            Er,
            batch_size,
            "Er",
        )

        self._validate_masks(
            Ea=Ea,
            Em=Em,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

        #######################################################################
        # Build Z_i^T and Z_j^S
        #######################################################################

        temporal_features, spatial_features = (
            self._build_gsta_inputs(
                Ea=Ea,
                Em=Em,
                Er=relative_embeddings,
                scene_graphs=graphs,
            )
        )

        #######################################################################
        # Eq. (3)
        #
        # Temporal self-attention.
        #######################################################################

        temporal_features = (
            self._temporal_self_attention(
                temporal_features,
                agent_mask=agent_mask,
            )
        )

        #######################################################################
        # Eq. (4)
        #
        # Spatial self-attention.
        #######################################################################

        spatial_features = (
            self._spatial_self_attention(
                spatial_features,
                map_mask=map_mask,
            )
        )

        #######################################################################
        # Eq. (5)
        #
        # Temporal features attend to spatial features.
        #######################################################################

        temporal_features = (
            self._temporal_to_spatial_attention(
                temporal_features,
                spatial_features,
                map_mask=map_mask,
            )
        )

        #######################################################################
        # Eq. (6)
        #
        # Spatial features attend to temporal features.
        #######################################################################

        spatial_features = (
            self._spatial_to_temporal_attention(
                spatial_features,
                temporal_features,
                agent_mask=agent_mask,
            )
        )

        #######################################################################
        # Eq. (7)
        #
        # Temporal learnable queries.
        #######################################################################

        temporal_scene = (
            self._temporal_query_attention(
                temporal_features,
                agent_mask=agent_mask,
            )
        )

        #######################################################################
        # Eq. (8)
        #
        # Spatial learnable queries.
        #######################################################################

        spatial_scene = (
            self._spatial_query_attention(
                spatial_features,
                num_agents=Ea.shape[1],
                map_mask=map_mask,
            )
        )

        #######################################################################
        # Eq. (9)
        #
        # Z_scene = qT + qS
        #######################################################################

        Z_scene = (
            temporal_scene
            + spatial_scene
        )

        self._scene_prediction_embeddings = (
            Z_scene
        )

        return Z_scene

    ###########################################################################
    # Build GSTA Inputs
    ###########################################################################

    def _build_gsta_inputs(
        self,
        *,
        Ea: Tensor,
        Em: Tensor,
        Er: Sequence[
            RelativeSpatioTemporalEmbedding
        ],
        scene_graphs: Sequence[
            SceneGraph
        ],
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        Construct relation-aware temporal and spatial features.

        Temporal stream
        ---------------

            Ea + Er associated with agent-state nodes

        Shape:

            (B,N,H,D)

        Spatial stream
        --------------

            Em + Er associated with map nodes

        Shape:

            (B,M,D)

        Implementation note
        -------------------
        The paper states that Er is integrated into the temporal
        and spatial scene features, but does not specify the exact
        edge-to-node aggregation operator.

        We use mean aggregation over incident edges.
        """

        batch_size, num_agents, num_steps, hidden_dim = (
            Ea.shape
        )

        num_maps = Em.shape[1]

        expected_states = (
            num_agents
            * num_steps
        )

        temporal_features = Ea.clone()

        spatial_features = Em.clone()

        for batch_index in range(
            batch_size
        ):

            graph = scene_graphs[
                batch_index
            ]

            relative = Er[
                batch_index
            ]

            graph.validate()

            ###################################################################
            # Graph / tensor consistency
            ###################################################################

            if graph.num_agent_states != expected_states:

                raise ValueError(
                    "SceneGraph agent-state count does not match "
                    "Ea temporal dimensions: "
                    f"graph={graph.num_agent_states}, "
                    f"expected={expected_states}."
                )

            if graph.num_map_nodes != num_maps:

                raise ValueError(
                    "SceneGraph map-node count does not match "
                    "Em: "
                    f"graph={graph.num_map_nodes}, "
                    f"model={num_maps}."
                )

            if relative.edge_index.ndim != 2:

                raise ValueError(
                    "Er.edge_index must have shape (2,U)."
                )

            if relative.edge_index.shape[0] != 2:

                raise ValueError(
                    "Er.edge_index must have shape (2,U)."
                )

            if relative.embeddings.ndim != 2:

                raise ValueError(
                    "Er.embeddings must have shape (U,D)."
                )

            if relative.embeddings.shape[0] != (
                relative.edge_index.shape[1]
            ):

                raise ValueError(
                    "Er.edge_index and Er.embeddings must "
                    "contain the same number of edges."
                )

            if relative.embeddings.shape[1] != hidden_dim:

                raise ValueError(
                    "Er hidden dimension does not match "
                    "Ea/Em hidden dimension."
                )

            if relative.edge_type is not None:

                if relative.edge_type.shape != (
                    relative.edge_index.shape[1],
                ):

                    raise ValueError(
                        "Er.edge_type must have shape (U,)."
                    )

            ###################################################################
            # No edges
            ###################################################################

            if relative.embeddings.shape[0] == 0:
                continue

            ###################################################################
            # Move edge data to the feature device.
            ###################################################################

            edge_index = (
                relative.edge_index.to(
                    device=Ea.device,
                    dtype=torch.long,
                )
            )

            edge_embeddings = (
                relative.embeddings.to(
                    device=Ea.device,
                    dtype=Ea.dtype,
                )
            )

            ###################################################################
            # Unified graph node space
            #
            # Agent states:
            #
            #     [0, Ns)
            #
            # Map nodes:
            #
            #     [Ns, Ns+M)
            ###################################################################

            state_count = (
                graph.num_agent_states
            )

            map_offset = state_count

            source = edge_index[0]
            target = edge_index[1]

            ###################################################################
            # Validate edge bounds
            ###################################################################

            total_nodes = (
                state_count
                + graph.num_map_nodes
            )

            if edge_index.numel():

                if int(edge_index.min()) < 0:

                    raise ValueError(
                        "Er.edge_index contains "
                        "negative indices."
                    )

                if int(edge_index.max()) >= total_nodes:

                    raise ValueError(
                        "Er.edge_index contains an index "
                        "outside the unified SceneGraph "
                        "node range."
                    )

            ###################################################################
            # Aggregate relation embeddings onto agent states
            ###################################################################

            state_relation_sum = torch.zeros(
                (
                    state_count,
                    hidden_dim,
                ),
                dtype=Ea.dtype,
                device=Ea.device,
            )

            state_relation_count = torch.zeros(
                (
                    state_count,
                    1,
                ),
                dtype=Ea.dtype,
                device=Ea.device,
            )

            ###################################################################
            # Source state endpoints
            ###################################################################

            source_state = (
                source < state_count
            )

            if torch.any(
                source_state
            ):

                source_indices = source[
                    source_state
                ]

                source_embeddings = (
                    edge_embeddings[
                        source_state
                    ]
                )

                state_relation_sum.index_add_(
                    0,
                    source_indices,
                    source_embeddings,
                )

                state_relation_count.index_add_(
                    0,
                    source_indices,
                    torch.ones(
                        (
                            source_indices.numel(),
                            1,
                        ),
                        dtype=Ea.dtype,
                        device=Ea.device,
                    ),
                )

            ###################################################################
            # Target state endpoints
            ###################################################################

            target_state = (
                target < state_count
            )

            if torch.any(
                target_state
            ):

                target_indices = target[
                    target_state
                ]

                target_embeddings = (
                    edge_embeddings[
                        target_state
                    ]
                )

                state_relation_sum.index_add_(
                    0,
                    target_indices,
                    target_embeddings,
                )

                state_relation_count.index_add_(
                    0,
                    target_indices,
                    torch.ones(
                        (
                            target_indices.numel(),
                            1,
                        ),
                        dtype=Ea.dtype,
                        device=Ea.device,
                    ),
                )

            ###################################################################
            # Mean aggregation
            ###################################################################

            state_relation_mean = (
                state_relation_sum
                /
                state_relation_count.clamp_min(
                    1.0
                )
            )

            ###################################################################
            # Add relation information to Ea
            ###################################################################

            state_features = (
                Ea[
                    batch_index
                ]
                .reshape(
                    state_count,
                    hidden_dim,
                )
            )

            state_features = (
                state_features
                + state_relation_mean
            )

            temporal_features[
                batch_index
            ] = state_features.reshape(
                num_agents,
                num_steps,
                hidden_dim,
            )

            ###################################################################
            # Aggregate relation embeddings onto map nodes
            ###################################################################

            map_relation_sum = torch.zeros(
                (
                    num_maps,
                    hidden_dim,
                ),
                dtype=Ea.dtype,
                device=Ea.device,
            )

            map_relation_count = torch.zeros(
                (
                    num_maps,
                    1,
                ),
                dtype=Ea.dtype,
                device=Ea.device,
            )

            ###################################################################
            # Source map endpoints
            ###################################################################

            source_map = (
                source >= map_offset
            )

            if torch.any(
                source_map
            ):

                source_map_indices = (
                    source[
                        source_map
                    ]
                    - map_offset
                )

                source_map_embeddings = (
                    edge_embeddings[
                        source_map
                    ]
                )

                map_relation_sum.index_add_(
                    0,
                    source_map_indices,
                    source_map_embeddings,
                )

                map_relation_count.index_add_(
                    0,
                    source_map_indices,
                    torch.ones(
                        (
                            source_map_indices.numel(),
                            1,
                        ),
                        dtype=Ea.dtype,
                        device=Ea.device,
                    ),
                )

            ###################################################################
            # Target map endpoints
            ###################################################################

            target_map = (
                target >= map_offset
            )

            if torch.any(
                target_map
            ):

                target_map_indices = (
                    target[
                        target_map
                    ]
                    - map_offset
                )

                target_map_embeddings = (
                    edge_embeddings[
                        target_map
                    ]
                )

                map_relation_sum.index_add_(
                    0,
                    target_map_indices,
                    target_map_embeddings,
                )

                map_relation_count.index_add_(
                    0,
                    target_map_indices,
                    torch.ones(
                        (
                            target_map_indices.numel(),
                            1,
                        ),
                        dtype=Ea.dtype,
                        device=Ea.device,
                    ),
                )

            ###################################################################
            # Mean aggregation
            ###################################################################

            map_relation_mean = (
                map_relation_sum
                /
                map_relation_count.clamp_min(
                    1.0
                )
            )

            ###################################################################
            # Add relation information to Em
            ###################################################################

            spatial_features[
                batch_index
            ] = (
                Em[
                    batch_index
                ]
                + map_relation_mean
            )

        return (
            temporal_features,
            spatial_features,
        )

    ###########################################################################
    # Temporal Self-Attention
    ###########################################################################

    def _temporal_self_attention(
        self,
        features: Tensor,
        *,
        agent_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (3).

        Perform self-attention across the historical states of
        each agent independently.

        Input
        -----

            (B,N,H,D)

        Output
        ------

            (B,N,H,D)
        """

        batch_size, num_agents, num_steps, hidden_dim = (
            features.shape
        )

        x = self.temporal_norm(
            features
        )

        #######################################################################
        # Flatten batch and agent dimensions.
        #
        # Each agent becomes an independent attention sequence:
        #
        #     (B*N,H,D)
        #######################################################################

        x = x.reshape(
            batch_size * num_agents,
            num_steps,
            hidden_dim,
        )

        key_padding_mask = None

        if agent_mask is not None:

            valid = (
                agent_mask
                .to(dtype=torch.bool)
            )

            key_padding_mask = (
                ~valid
            ).unsqueeze(-1).expand(
                batch_size,
                num_agents,
                num_steps,
            )

            key_padding_mask = (
                key_padding_mask
                .reshape(
                    batch_size * num_agents,
                    num_steps,
                )
            )

        attended, _ = (
            self.temporal_attention(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        attended = self.dropout(
            attended
        )

        attended = attended.reshape(
            batch_size,
            num_agents,
            num_steps,
            hidden_dim,
        )

        return (
            features
            + attended
        )

    ###########################################################################
    # Spatial Self-Attention
    ###########################################################################

    def _spatial_self_attention(
        self,
        features: Tensor,
        *,
        map_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (4).

        Perform self-attention across map nodes.

        Input
        -----

            (B,M,D)

        Output
        ------

            (B,M,D)
        """

        x = self.spatial_norm(
            features
        )

        key_padding_mask = None

        if map_mask is not None:

            key_padding_mask = ~map_mask.to(
                dtype=torch.bool
            )

        attended, _ = (
            self.spatial_attention(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        attended = self.dropout(
            attended
        )

        return (
            features
            + attended
        )

    ###########################################################################
    # Temporal -> Spatial
    ###########################################################################

    def _temporal_to_spatial_attention(
        self,
        temporal_features: Tensor,
        spatial_features: Tensor,
        *,
        map_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (5).

        Temporal features attend to spatial/map features.

        Input temporal:

            (B,N,H,D)

        Spatial:

            (B,M,D)

        Output:

            (B,N,H,D)
        """

        batch_size, num_agents, num_steps, hidden_dim = (
            temporal_features.shape
        )

        query = self.temporal_cross_norm(
            temporal_features
        )

        query = query.reshape(
            batch_size,
            num_agents * num_steps,
            hidden_dim,
        )

        key = self.spatial_cross_norm(
            spatial_features
        )

        key_padding_mask = None

        if map_mask is not None:

            key_padding_mask = ~map_mask.to(
                dtype=torch.bool
            )

        attended, _ = (
            self.temporal_to_spatial(
                query=query,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        attended = self.dropout(
            attended
        )

        attended = attended.reshape(
            batch_size,
            num_agents,
            num_steps,
            hidden_dim,
        )

        return (
            temporal_features
            + attended
        )

    ###########################################################################
    # Spatial -> Temporal
    ###########################################################################

    def _spatial_to_temporal_attention(
        self,
        spatial_features: Tensor,
        temporal_features: Tensor,
        *,
        agent_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (6).

        Spatial/map features attend to temporal agent-state
        features.

        Input spatial:

            (B,M,D)

        Temporal:

            (B,N,H,D)

        Output:

            (B,M,D)
        """

        batch_size, num_agents, num_steps, hidden_dim = (
            temporal_features.shape
        )

        query = self.spatial_cross_norm(
            spatial_features
        )

        key = self.temporal_cross_norm(
            temporal_features
        )

        key = key.reshape(
            batch_size,
            num_agents * num_steps,
            hidden_dim,
        )

        key_padding_mask = None

        if agent_mask is not None:

            valid = agent_mask.to(
                dtype=torch.bool
            )

            key_padding_mask = (
                ~valid
            ).unsqueeze(-1).expand(
                batch_size,
                num_agents,
                num_steps,
            )

            key_padding_mask = (
                key_padding_mask.reshape(
                    batch_size,
                    num_agents * num_steps,
                )
            )

        attended, _ = (
            self.spatial_to_temporal(
                query=query,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        attended = self.dropout(
            attended
        )

        return (
            spatial_features
            + attended
        )

    ###########################################################################
    # Temporal Learnable Queries
    ###########################################################################

    def _temporal_query_attention(
        self,
        temporal_features: Tensor,
        *,
        agent_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (7).

        Generate q^T_(n,t,k).

        The learned query bank has shape:

            (H,K,D)

        and is broadcast over N agents.

        Queries:

            (B,N,H,K,D)

        are flattened to:

            (B,N*H*K,D)

        and attend to:

            (B,N*H,D)

        temporal features.

        Output:

            (B,N,H,K,D)
        """

        batch_size, num_agents, num_steps, hidden_dim = (
            temporal_features.shape
        )

        if num_steps != self.observation_steps:

            raise ValueError(
                "Temporal feature length does not match "
                f"observation_steps={self.observation_steps}."
            )

        #######################################################################
        # Expand learned queries over batch and agents.
        #######################################################################

        queries = (
            self.temporal_queries
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                num_agents,
                num_steps,
                self.num_modes,
                hidden_dim,
            )
        )

        queries = queries.reshape(
            batch_size,
            num_agents
            * num_steps
            * self.num_modes,
            hidden_dim,
        )

        #######################################################################
        # Temporal keys / values
        #######################################################################

        memory = temporal_features.reshape(
            batch_size,
            num_agents * num_steps,
            hidden_dim,
        )

        #######################################################################
        # Agent mask applies to every historical state.
        #######################################################################

        key_padding_mask = None

        if agent_mask is not None:

            valid = agent_mask.to(
                dtype=torch.bool
            )

            key_padding_mask = (
                ~valid
            ).unsqueeze(-1).expand(
                batch_size,
                num_agents,
                num_steps,
            )

            key_padding_mask = (
                key_padding_mask.reshape(
                    batch_size,
                    num_agents * num_steps,
                )
            )

        output, _ = (
            self.temporal_query_attention(
                query=queries,
                key=memory,
                value=memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        output = self.dropout(
            output
        )

        return output.reshape(
            batch_size,
            num_agents,
            num_steps,
            self.num_modes,
            hidden_dim,
        )

    ###########################################################################
    # Spatial Learnable Queries
    ###########################################################################

    def _spatial_query_attention(
        self,
        spatial_features: Tensor,
        *,
        num_agents: int,
        map_mask: Tensor | None,
    ) -> Tensor:
        """
        Eq. (8).

        Generate q^S_(n,t,k).

        The spatial query bank has shape:

            (H,K,D)

        and is broadcast over agents.

        Every query attends to the spatial/map feature sequence.

        Output:

            (B,N,H,K,D)
        """

        batch_size, num_maps, hidden_dim = (
            spatial_features.shape
        )

        #######################################################################
        # Expand learned queries over batch and agents.
        #######################################################################

        queries = (
            self.spatial_queries
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                num_agents,
                self.observation_steps,
                self.num_modes,
                hidden_dim,
            )
        )

        queries = queries.reshape(
            batch_size,
            num_agents
            * self.observation_steps
            * self.num_modes,
            hidden_dim,
        )

        #######################################################################
        # Map memory
        #######################################################################

        memory = spatial_features

        key_padding_mask = None

        if map_mask is not None:

            key_padding_mask = ~map_mask.to(
                dtype=torch.bool
            )

        output, _ = (
            self.spatial_query_attention(
                query=queries,
                key=memory,
                value=memory,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        output = self.dropout(
            output
        )

        return output.reshape(
            batch_size,
            num_agents,
            self.observation_steps,
            self.num_modes,
            hidden_dim,
        )

    ###########################################################################
    # Input Validation
    ###########################################################################

    def _validate_inputs(
        self,
        *,
        Ea: Tensor,
        Em: Tensor,
    ) -> None:
        """
        Validate primary GSTA tensors.
        """

        if Ea.ndim != 4:

            raise ValueError(
                "Ea must have shape (B,N,H,D)."
            )

        if Em.ndim != 3:

            raise ValueError(
                "Em must have shape (B,M,D)."
            )

        if Ea.shape[-1] != self.hidden_dim:

            raise ValueError(
                "Ea hidden dimension does not match "
                f"hidden_dim={self.hidden_dim}."
            )

        if Em.shape[-1] != self.hidden_dim:

            raise ValueError(
                "Em hidden dimension does not match "
                f"hidden_dim={self.hidden_dim}."
            )

        if Ea.shape[2] != self.observation_steps:

            raise ValueError(
                "Ea observation dimension does not match "
                f"observation_steps={self.observation_steps}."
            )

        if Ea.shape[0] != Em.shape[0]:

            raise ValueError(
                "Ea and Em must have the same batch size."
            )

    ###########################################################################
    # Mask Validation
    ###########################################################################

    def _validate_masks(
        self,
        *,
        Ea: Tensor,
        Em: Tensor,
        agent_mask: Tensor | None,
        map_mask: Tensor | None,
    ) -> None:
        """
        Validate optional padding masks.
        """

        batch_size = Ea.shape[0]

        num_agents = Ea.shape[1]

        num_maps = Em.shape[1]

        if agent_mask is not None:

            if agent_mask.shape != (
                batch_size,
                num_agents,
            ):

                raise ValueError(
                    "agent_mask must have shape "
                    "(B,N)."
                )

        if map_mask is not None:

            if map_mask.shape != (
                batch_size,
                num_maps,
            ):

                raise ValueError(
                    "map_mask must have shape "
                    "(B,M)."
                )

    ###########################################################################
    # Sequence Normalization
    ###########################################################################

    @staticmethod
    def _as_sequence(
        value: (
            SceneGraph
            | RelativeSpatioTemporalEmbedding
            | Sequence[SceneGraph]
            | Sequence[RelativeSpatioTemporalEmbedding]
        ),
        batch_size: int,
        name: str,
    ) -> list:
        """
        Normalize a single object or sequence into a batch list.
        """

        if isinstance(
            value,
            (
                SceneGraph,
                RelativeSpatioTemporalEmbedding,
            ),
        ):

            if batch_size != 1:

                raise ValueError(
                    f"{name} contains one object but "
                    f"batch size is {batch_size}."
                )

            return [
                value
            ]

        result = list(
            value
        )

        if len(result) != batch_size:

            raise ValueError(
                f"{name} contains {len(result)} objects "
                f"but batch size is {batch_size}."
            )

        return result

    ###########################################################################
    # Scene Embedding Accessor
    ###########################################################################

    @property
    def scene_prediction_embeddings(
        self,
    ) -> Tensor | None:
        """
        Return the scene prediction embedding produced by the
        most recent forward pass.
        """

        return self._scene_prediction_embeddings

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_modes={self.num_modes}, "
            f"observation_steps={self.observation_steps}"
        )


###############################################################################
# Public API
###############################################################################

__all__ = [
    "GSTA",
]
