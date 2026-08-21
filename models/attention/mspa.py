"""
models.attention.mspa

Multi-scale Spatial Pattern Attention (MSPA).

DSTNet
------
Section III-D
Equations (10)-(15)

MSPA operates on the scene prediction embeddings produced
by GSTA.

Input
-----
Z_scene

    (B,N,H,K,D)

Output
------
Z^S

    (B,N,H,K,D)

The spatial attention operates over the agent dimension N.
The historical dimension H and mode dimension K are preserved.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MSPA(nn.Module):
    """
    Multi-scale Spatial Pattern Attention.

    For each attention head i:

        r_i = iR / H_a

    where:

        H_a = number of attention heads
        R   = maximum interaction radius

    Each head attends only to agents inside its
    corresponding spatial radius.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        interaction_radius: float,
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

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.interaction_radius = float(
            interaction_radius
        )

        self.head_dim = (
            hidden_dim // num_heads
        )

        #######################################################################
        # Q, K, V
        #
        # Eqs. (10)-(12)
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
        # Learnable alpha_i
        #######################################################################

        self.alpha = nn.Parameter(
            torch.ones(num_heads)
        )

        #######################################################################
        # Output projection
        #
        # Eq. (15)
        #######################################################################

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    ###########################################################################
    # Interaction radii
    ###########################################################################

    def _interaction_radii(self) -> Tensor:
        """
        Compute:

            r_i = iR / H_a

        for i = 1,...,H_a.
        """

        indices = torch.arange(
            1,
            self.num_heads + 1,
            device=self.alpha.device,
            dtype=self.alpha.dtype,
        )

        return (
            indices
            * self.interaction_radius
            / self.num_heads
        )

    ###########################################################################
    # Radius masks
    ###########################################################################

    def _build_radius_masks(
        self,
        positions: Tensor,
        agent_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Build spatial neighbourhood masks.

        Parameters
        ----------
        positions
            (B,N,2)

        agent_mask
            (B,N)

        Returns
        -------
        Tensor
            (B,num_heads,N,N)
        """

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        if positions.shape[-1] != 2:
            raise ValueError(
                "positions must have last dimension 2."
            )

        #######################################################################
        # Pairwise distances
        #######################################################################

        delta = (
            positions[:, :, None, :]
            - positions[:, None, :, :]
        )

        distances = torch.linalg.norm(
            delta,
            dim=-1,
        )

        radii = self._interaction_radii().to(
            device=positions.device,
            dtype=positions.dtype,
        )

        #######################################################################
        # (B,N,N)
        #
        # ->
        #
        # (B,H_a,N,N)
        #######################################################################

        masks = (
            distances.unsqueeze(1)
            <= radii.view(
                1,
                self.num_heads,
                1,
                1,
            )
        )

        #######################################################################
        # Self attention is always permitted for valid agents.
        #######################################################################

        num_agents = positions.shape[1]

        identity = torch.eye(
            num_agents,
            dtype=torch.bool,
            device=positions.device,
        )

        masks = (
            masks
            | identity.view(
                1,
                1,
                num_agents,
                num_agents,
            )
        )

        #######################################################################
        # Remove padded agents.
        #######################################################################

        if agent_mask is not None:

            if agent_mask.shape != (
                positions.shape[0],
                positions.shape[1],
            ):
                raise ValueError(
                    "agent_mask must have shape (B,N)."
                )

            valid_pairs = (
                agent_mask[:, :, None]
                & agent_mask[:, None, :]
            )

            masks = masks & valid_pairs.unsqueeze(1)

            ###################################################################
            # Restore self connections for valid agents.
            ###################################################################

            masks = (
                masks
                | (
                    identity.view(
                        1,
                        1,
                        num_agents,
                        num_agents,
                    )
                    & agent_mask.bool().unsqueeze(1).unsqueeze(-1)
                )
            )

        return masks

    ###########################################################################
    # One attention head
    ###########################################################################

    def _single_head_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Spatial attention for one head.

        query
            (B,N,H,K,Dh)

        key
            (B,N,H,K,Dh)

        value
            (B,N,H,K,Dh)

        mask
            (B,N,N)

        output
            (B,N,H,K,Dh)
        """

        #######################################################################
        # Agent-to-agent attention scores.
        #
        # Scores:
        #
        # (B,N,N,H,K)
        #######################################################################

        scores = torch.einsum(
            "bntkd,bmskd->bnmtk",
            query,
            key,
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        valid = mask.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        scores = scores.masked_fill(
            ~valid,
            torch.finfo(
                scores.dtype
            ).min,
        )

        #######################################################################
        # Softmax over neighbour agent dimension.
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
        Apply MSPA.

        Parameters
        ----------
        scene_embeddings
            (B,N,H,K,D)

        positions
            (B,N,2)

        agent_mask
            (B,N)

        Returns
        -------
        Z^S

            (B,N,H,K,D)
        """

        if scene_embeddings.ndim != 5:
            raise ValueError(
                "scene_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        batch_size, num_agents, history_steps, num_modes, hidden_dim = (
            scene_embeddings.shape
        )

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
            0, 4, 1, 2, 3, 5
        )

        K = K.permute(
            0, 4, 1, 2, 3, 5
        )

        V = V.permute(
            0, 4, 1, 2, 3, 5
        )

        #######################################################################
        # Multi-scale neighbourhoods.
        #######################################################################

        radius_masks = self._build_radius_masks(
            positions,
            agent_mask,
        )

        #######################################################################
        # One output per head.
        #######################################################################

        head_outputs: list[Tensor] = []

        for head_index in range(
            self.num_heads
        ):

            output = self._single_head_attention(
                query=Q[:, head_index],
                key=K[:, head_index],
                value=V[:, head_index],
                mask=radius_masks[:, head_index],
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

    def extra_repr(self) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"interaction_radius={self.interaction_radius}"
        )


__all__ = ["MSPA"]
