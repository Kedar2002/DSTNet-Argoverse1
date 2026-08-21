"""
models.attention.mmia

Multi-Mode Interaction Attention (MMIA).

DSTNet
------
Section III-D
Equation (24)

Input
-----
Z^ST

    (B,N,H,K,D)

Output
------
Z^STM

    (B,N,H,K,D)

Attention is performed ONLY across prediction modes K.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MMIA(nn.Module):
    """
    Multi-Mode Interaction Attention.

    For every agent n and historical timestep t:

        Z^STM_(n,t,k)
            =
        MHA(
            Z^ST_(n,t,k),
            Z^ST_(n,t,1:K),
            Z^ST_(n,t,1:K)
        )

    Therefore K is the attention sequence dimension.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
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

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = (
            hidden_dim // num_heads
        )

        #######################################################################
        # QKV
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
        # Output projection
        #######################################################################

        self.output_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        spatio_temporal_embeddings: Tensor,
    ) -> Tensor:
        """
        Apply Eq. (24).

        Input:

            (B,N,H,K,D)

        Output:

            (B,N,H,K,D)
        """

        if spatio_temporal_embeddings.ndim != 5:
            raise ValueError(
                "spatio_temporal_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        B, N, H, K_modes, D = (
            spatio_temporal_embeddings.shape
        )

        if D != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {D}."
            )

        #######################################################################
        # Collapse B, N and H.
        #
        # Every (agent,time) pair gets an independent K-mode attention
        # sequence.
        #######################################################################

        x = spatio_temporal_embeddings.reshape(
            B * N * H,
            K_modes,
            D,
        )

        #######################################################################
        # QKV
        #######################################################################

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        #######################################################################
        # Split heads.
        #######################################################################

        Q = Q.reshape(
            B * N * H,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )

        K = K.reshape(
            B * N * H,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )

        V = V.reshape(
            B * N * H,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).transpose(
            1,
            2,
        )

        #######################################################################
        # Attention over K.
        #######################################################################

        scores = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1,
            ),
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        #######################################################################
        # Aggregate values.
        #######################################################################

        output = torch.matmul(
            attention,
            V,
        )

        #######################################################################
        # Merge heads.
        #######################################################################

        output = output.transpose(
            1,
            2,
        ).reshape(
            B * N * H,
            K_modes,
            D,
        )

        output = self.output_projection(
            output
        )

        #######################################################################
        # Restore scene dimensions.
        #######################################################################

        return output.reshape(
            B,
            N,
            H,
            K_modes,
            D,
        )

    def extra_repr(self) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}"
        )


__all__ = ["MMIA"]
