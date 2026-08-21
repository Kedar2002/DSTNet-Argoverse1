"""
models.attention.arp_mspa

Adaptive Radius Prediction Multi-scale Spatial Pattern Attention
(ARP-MSPA).

Proposed extension of the current DSTNet MSPA implementation.

Baseline contract preserved
---------------------------
Input:
    scene_embeddings : (B,N,H,K,D)
    positions        : (B,N,2)
    agent_mask       : (B,N), optional

Output:
    (B,N,H,K,D)

The original MSPA uses fixed per-head radii:

    r_h = hR / H_a

ARP-MSPA replaces the fixed spatial-range mechanism with an
agent-specific learned radius:

    r_i = r_min + (r_max-r_min)
          * sigmoid(W2 GELU(W1 x_i + b1) + b2)

The predicted radius is converted into a soft spatial bias:

    B_ij = -d_ij^2 / (2 r_i^2)

and added to the attention logits:

    A_ij = softmax(
        Q_i K_j^T / sqrt(d_k)
        - d_ij^2 / (2 r_i^2)
    )

All other major MSPA components are retained:
    - Q/K/V projections
    - multi-head attention
    - learnable alpha_h head weights
    - output projection
    - agent masking
    - (B,N,H,K,D) tensor contract

For computational sparsity, r_max is also used as a hard candidate
radius. Thus agents outside r_max are not evaluated, while agents
inside r_max receive a differentiable distance-dependent bias based
on the target agent's learned radius.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ARPMSPA(nn.Module):
    """
    Adaptive Radius Prediction Multi-scale Spatial Pattern Attention.

    Parameters
    ----------
    hidden_dim
        Feature dimension D.

    num_heads
        Number of attention heads.

    interaction_radius
        Maximum candidate interaction radius R. This is retained for
        compatibility with the current MSPA constructor. If r_max is
        supplied, r_max is used as the effective maximum radius.

    r_min
        Minimum allowed learned radius.

    r_max
        Maximum allowed learned radius. If None, interaction_radius
        is used.

    radius_hidden_dim
        Hidden dimension of the adaptive radius MLP. If None, hidden_dim
        is used.

    dropout
        Attention dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        interaction_radius: float,
        r_min: float,
        r_max: float | None = None,
        radius_hidden_dim: int | None = None,
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

        if interaction_radius <= 0:
            raise ValueError(
                "interaction_radius must be positive."
            )

        if r_min <= 0:
            raise ValueError(
                "r_min must be positive."
            )

        if r_max is None:
            r_max = interaction_radius

        if r_max <= r_min:
            raise ValueError(
                "r_max must be greater than r_min."
            )

        if interaction_radius < r_max:
            raise ValueError(
                "interaction_radius must be >= r_max."
            )

        if radius_hidden_dim is None:
            radius_hidden_dim = hidden_dim

        if radius_hidden_dim <= 0:
            raise ValueError(
                "radius_hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        # Retain the baseline name for compatibility and readability.
        self.interaction_radius = float(
            interaction_radius
        )

        self.r_min = float(r_min)
        self.r_max = float(r_max)

        self.radius_hidden_dim = int(
            radius_hidden_dim
        )

        self.head_dim = (
            hidden_dim // num_heads
        )

        #######################################################################
        # Q, K, V
        #
        # Same projections as the current MSPA implementation.
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
        # Adaptive Radius Predictor
        #
        # Agent feature:
        #
        #     x_i = mean_{h,k}(Z_scene[i,h,k,:])
        #
        # Shape:
        #
        #     (B,N,H,K,D) -> (B,N,D)
        #
        # Radius:
        #
        #     r_i = r_min + (r_max-r_min)
        #           sigmoid(
        #               W2 GELU(W1 x_i + b1) + b2
        #           )
        #######################################################################

        self.radius_predictor = nn.Sequential(
            nn.Linear(
                hidden_dim,
                radius_hidden_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                radius_hidden_dim,
                1,
                bias=True,
            ),
        )

        #######################################################################
        # Initialize the final radius layer so that the initial predicted
        # radius is the midpoint of [r_min, r_max].
        #
        # sigmoid(0) = 0.5
        #
        # Therefore:
        #
        # r_i = r_min + 0.5(r_max-r_min)
        #
        # at initialization.
        #######################################################################

        final_radius_layer = self.radius_predictor[-1]

        if isinstance(final_radius_layer, nn.Linear):
            nn.init.zeros_(
                final_radius_layer.weight
            )
            nn.init.zeros_(
                final_radius_layer.bias
            )

        #######################################################################
        # Learnable alpha_h head weights.
        #
        # Retained from the current MSPA.
        #######################################################################

        self.alpha = nn.Parameter(
            torch.ones(num_heads)
        )

        #######################################################################
        # Output projection.
        #######################################################################

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    ###########################################################################
    # Agent features for radius prediction
    ###########################################################################

    def _agent_features(
        self,
        scene_embeddings: Tensor,
    ) -> Tensor:
        """
        Build one D-dimensional feature vector per agent.

        Input
            (B,N,H,K,D)

        Output
            (B,N,D)

        The historical and mode dimensions are mean-pooled only for
        radius prediction. The original attention computation still
        preserves H and K.
        """

        if scene_embeddings.ndim != 5:
            raise ValueError(
                "scene_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        return scene_embeddings.mean(
            dim=(2, 3)
        )

    ###########################################################################
    # Adaptive radius prediction
    ###########################################################################

    def predict_radius(
        self,
        agent_features: Tensor,
    ) -> Tensor:
        """
        Predict one bounded interaction radius per target agent.

        Parameters
        ----------
        agent_features
            (B,N,D)

        Returns
        -------
        Tensor
            (B,N)

        Formula
        -------
        r_i = r_min + (r_max-r_min)
              sigmoid(
                  W2 GELU(W1 x_i + b1) + b2
              )
        """

        if agent_features.ndim != 3:
            raise ValueError(
                "agent_features must have shape (B,N,D)."
            )

        if agent_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected agent feature dimension "
                f"{self.hidden_dim}, got "
                f"{agent_features.shape[-1]}."
            )

        raw_radius_score = self.radius_predictor(
            agent_features
        ).squeeze(-1)

        radius_fraction = torch.sigmoid(
            raw_radius_score
        )

        radius = (
            self.r_min
            + (
                self.r_max
                - self.r_min
            )
            * radius_fraction
        )

        return radius

    ###########################################################################
    # Pairwise distances
    ###########################################################################

    def _pairwise_distances(
        self,
        positions: Tensor,
    ) -> Tensor:
        """
        Compute pairwise Euclidean distances.

        positions
            (B,N,2)

        returns
            (B,N,N)
        """

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        if positions.shape[-1] != 2:
            raise ValueError(
                "positions must have last dimension 2."
            )

        delta = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        return torch.linalg.norm(
            delta,
            dim=-1,
        )

    ###########################################################################
    # Maximum-radius candidate mask
    ###########################################################################

    def _build_candidate_mask(
        self,
        positions: Tensor,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Build the maximum-radius candidate mask.

        Unlike the original MSPA, ARP-MSPA does not create one hard
        neighbourhood mask per attention head.

        Instead, all agents within r_max are candidates and receive
        the differentiable adaptive spatial bias.

        Returns
        -------
        Tensor
            (B,N,N)
        """

        distances = self._pairwise_distances(
            positions
        )

        candidate_mask = (
            distances <= self.r_max
        )

        #######################################################################
        # Self attention is always permitted for valid agents.
        #######################################################################

        batch_size, num_agents, _ = positions.shape

        identity = torch.eye(
            num_agents,
            dtype=torch.bool,
            device=positions.device,
        )

        candidate_mask = (
            candidate_mask
            | identity.unsqueeze(0)
        )

        #######################################################################
        # Remove padded agents.
        #######################################################################

        if agent_mask is not None:

            if agent_mask.shape != (
                batch_size,
                num_agents,
            ):
                raise ValueError(
                    "agent_mask must have shape (B,N)."
                )

            valid_pairs = (
                agent_mask[:, :, None]
                & agent_mask[:, None, :]
            )

            candidate_mask = (
                candidate_mask
                & valid_pairs
            )

            ###################################################################
            # Restore self connections for valid agents.
            ###################################################################

            candidate_mask = (
                candidate_mask
                | (
                    identity.unsqueeze(0)
                    & agent_mask.bool().unsqueeze(-1)
                )
            )

        return candidate_mask

    ###########################################################################
    # Adaptive spatial bias
    ###########################################################################

    def _adaptive_spatial_bias(
        self,
        positions: Tensor,
        radii: Tensor,
    ) -> Tensor:
        """
        Compute target-agent-specific spatial bias.

        Parameters
        ----------
        positions
            (B,N,2)

        radii
            (B,N)

        Returns
        -------
        Tensor
            (B,N,N)

        Formula
        -------
        B_ij = -d_ij^2 / (2 r_i^2)

        Note that r_i belongs to the TARGET agent i.
        """

        distances = self._pairwise_distances(
            positions
        )

        radius_squared = (
            radii
            .clamp_min(
                torch.finfo(
                    radii.dtype
                ).eps
            )
            .square()
        )

        bias = -(
            distances.square()
            / (
                2.0
                * radius_squared.unsqueeze(-1)
            )
        )

        return bias

    ###########################################################################
    # One attention head
    ###########################################################################

    def _single_head_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        candidate_mask: Tensor,
        spatial_bias: Tensor,
    ) -> Tensor:
        """
        Adaptive spatial attention for one head.

        query
            (B,N,H,K,Dh)

        key
            (B,N,H,K,Dh)

        value
            (B,N,H,K,Dh)

        candidate_mask
            (B,N,N)

        spatial_bias
            (B,N,N)

        output
            (B,N,H,K,Dh)
        """

        #######################################################################
        # Agent-to-agent attention scores.
        #
        # (B,N,H,K,Dh)
        # (B,M,H,K,Dh)
        #
        # ->
        #
        # (B,N,M,H,K)
        #######################################################################

        scores = torch.einsum(
            "bntkd,bmskd->bnmtk",
            query,
            key,
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        #######################################################################
        # Add adaptive spatial bias.
        #
        # Bias is shared across historical steps, modes and heads because
        # r_i is predicted at the agent level.
        #######################################################################

        scores = (
            scores
            + spatial_bias.unsqueeze(-1).unsqueeze(-1)
        )

        #######################################################################
        # Hard maximum-radius candidate mask.
        #
        # This preserves sparse computation beyond r_max while the learned
        # radius controls the strength of spatial suppression inside r_max.
        #######################################################################

        valid = (
            candidate_mask
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        scores = scores.masked_fill(
            ~valid,
            torch.finfo(
                scores.dtype
            ).min,
        )

        #######################################################################
        # Softmax over neighbour-agent dimension.
        #######################################################################

        attention = torch.softmax(
            scores,
            dim=2,
        )

        attention = attention.masked_fill(
            ~valid,
            0.0,
        )

        attention = self.dropout(
            attention
        )

        #######################################################################
        # Weighted aggregation.
        #######################################################################

        return torch.einsum(
            "bnmtk,bmskd->bntkd",
            attention,
            value,
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        scene_embeddings: Tensor,
        positions: Tensor,
        *,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Apply ARP-MSPA.

        Parameters
        ----------
        scene_embeddings
            (B,N,H,K,D)

        positions
            (B,N,2)

        agent_mask
            (B,N), optional

        Returns
        -------
        Tensor
            (B,N,H,K,D)
        """

        if scene_embeddings.ndim != 5:
            raise ValueError(
                "scene_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        (
            batch_size,
            num_agents,
            history_steps,
            num_modes,
            hidden_dim,
        ) = scene_embeddings.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got {hidden_dim}."
            )

        if positions.shape != (
            batch_size,
            num_agents,
            2,
        ):
            raise ValueError(
                "positions must have shape (B,N,2) "
                "matching scene_embeddings."
            )

        if agent_mask is not None and agent_mask.shape != (
            batch_size,
            num_agents,
        ):
            raise ValueError(
                "agent_mask must have shape (B,N)."
            )

        #######################################################################
        # Agent-specific radius prediction.
        #
        # Z_scene:
        #
        # (B,N,H,K,D)
        #
        # ->
        #
        # x_i:
        #
        # (B,N,D)
        #######################################################################

        agent_features = self._agent_features(
            scene_embeddings
        )

        radii = self.predict_radius(
            agent_features
        )

        #######################################################################
        # Q / K / V
        #######################################################################

        Q = self.query_projection(
            scene_embeddings
        )

        K = self.key_projection(
            scene_embeddings
        )

        V = self.value_projection(
            scene_embeddings
        )

        #######################################################################
        # Split heads.
        #######################################################################

        Q = Q.reshape(
            batch_size,
            num_agents,
            history_steps,
            num_modes,
            self.num_heads,
            self.head_dim,
        )

        K = K.reshape(
            batch_size,
            num_agents,
            history_steps,
            num_modes,
            self.num_heads,
            self.head_dim,
        )

        V = V.reshape(
            batch_size,
            num_agents,
            history_steps,
            num_modes,
            self.num_heads,
            self.head_dim,
        )

        #######################################################################
        # Move head dimension.
        #######################################################################

        Q = Q.permute(
            0,
            4,
            1,
            2,
            3,
            5,
        )

        K = K.permute(
            0,
            4,
            1,
            2,
            3,
            5,
        )

        V = V.permute(
            0,
            4,
            1,
            2,
            3,
            5,
        )

        #######################################################################
        # Maximum-radius candidate set.
        #######################################################################

        candidate_mask = self._build_candidate_mask(
            positions,
            agent_mask,
        )

        #######################################################################
        # Adaptive spatial bias.
        #######################################################################

        spatial_bias = self._adaptive_spatial_bias(
            positions,
            radii,
        )

        #######################################################################
        # One output per head.
        #
        # Unlike the original MSPA, all heads use the same adaptive
        # target-agent radius and candidate set. The multi-head mechanism
        # is retained for feature subspace specialization.
        #######################################################################

        head_outputs: list[Tensor] = []

        for head_index in range(
            self.num_heads
        ):

            output = self._single_head_attention(
                query=Q[:, head_index],
                key=K[:, head_index],
                value=V[:, head_index],
                candidate_mask=candidate_mask,
                spatial_bias=spatial_bias,
            )

            output = (
                self.alpha[head_index]
                * output
            )

            head_outputs.append(
                output
            )

        #######################################################################
        # Concatenate heads.
        #######################################################################

        output = torch.cat(
            head_outputs,
            dim=-1,
        )

        #######################################################################
        # Output transformation.
        #######################################################################

        output = self.output_projection(
            output
        )

        #######################################################################
        # Mask invalid target agents.
        #######################################################################

        if agent_mask is not None:

            output = output.masked_fill(
                ~agent_mask.bool()
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1),
                0.0,
            )

        return output

    ###########################################################################
    # Diagnostics
    ###########################################################################

    def extra_repr(self) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"interaction_radius={self.interaction_radius}, "
            f"r_min={self.r_min}, "
            f"r_max={self.r_max}, "
            f"radius_hidden_dim={self.radius_hidden_dim}"
        )


__all__ = ["ARPMSPA"]
