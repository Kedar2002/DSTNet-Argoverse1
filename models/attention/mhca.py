"""
models.attention.mhca

Multi-Head Historical Causal Attention (MHCA).

DSTNet
------
Section III-D
Equations (16)-(23)

Input
-----
Z^S

    (B,N,H,K,D)

Output
------
Z^ST

    (B,N,H,K,D)

Attention is performed over historical time within each
temporal window. The prediction-mode dimension K is preserved.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MHCA(nn.Module):
    """
    Multi-Head Historical Causal Attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_sizes: tuple[int, ...] = (
            2,
            4,
            8,
        ),
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

        if not window_sizes:
            raise ValueError(
                "window_sizes cannot be empty."
            )

        if any(
            size <= 0
            for size in window_sizes
        ):
            raise ValueError(
                "All window sizes must be positive."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = (
            hidden_dim // num_heads
        )
        self.window_sizes = tuple(
            int(size)
            for size in window_sizes
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
        # Multi-scale fusion
        #######################################################################

        self.output_projection = nn.Linear(
            hidden_dim * len(window_sizes),
            hidden_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    ###########################################################################
    # Causal mask
    ###########################################################################

    @staticmethod
    def _causal_mask(
        size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:

        mask = torch.zeros(
            size,
            size,
            device=device,
            dtype=dtype,
        )

        future = torch.triu(
            torch.ones(
                size,
                size,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return mask.masked_fill(
            future,
            torch.finfo(dtype).min,
        )

    ###########################################################################
    # One temporal window
    ###########################################################################

    def _apply_window(
        self,
        features: Tensor,
        window_size: int,
    ) -> Tensor:
        """
        Apply causal attention at one temporal scale.

        Input/output:

            (B,N,H,K,D)
        """

        B, N, H, K_modes, D = features.shape

        num_groups = math.ceil(
            H / window_size
        )

        padded_H = (
            num_groups * window_size
        )

        pad = padded_H - H

        if pad > 0:

            features = torch.cat(
                (
                    features,
                    features.new_zeros(
                        B,
                        N,
                        pad,
                        K_modes,
                        D,
                    ),
                ),
                dim=2,
            )

        #######################################################################
        # (B,N,G,S,K,D)
        #######################################################################

        x = features.reshape(
            B,
            N,
            num_groups,
            window_size,
            K_modes,
            D,
        )

        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        #######################################################################
        # (B,N,G,S,K,D)
        # ->
        # (B,N,G,H_a,S,K,Dh)
        #######################################################################

        Q = Q.reshape(
            B,
            N,
            num_groups,
            window_size,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).permute(
            0, 1, 2, 5, 3, 4, 6
        )

        K = K.reshape(
            B,
            N,
            num_groups,
            window_size,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).permute(
            0, 1, 2, 5, 3, 4, 6
        )

        V = V.reshape(
            B,
            N,
            num_groups,
            window_size,
            K_modes,
            self.num_heads,
            self.head_dim,
        ).permute(
            0, 1, 2, 5, 3, 4, 6
        )

        #######################################################################
        # Attention over historical timestep S.
        #######################################################################

        scores = torch.einsum(
            "bnghsmd,bnghtmd->bnghsmt",
            Q,
            K,
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        causal = self._causal_mask(
            window_size,
            device=scores.device,
            dtype=scores.dtype,
        )

        scores = scores + causal.view(
            1,
            1,
            1,
            1,
            window_size,
            1,
            window_size,
        )

        #######################################################################
        # Padding mask.
        #######################################################################

        valid = torch.arange(
            padded_H,
            device=features.device,
        ) < H

        valid = valid.reshape(
            num_groups,
            window_size,
        )

        key_valid = valid.view(
            1,
            1,
            num_groups,
            1,
            1,
            1,
            window_size,
        )

        scores = scores.masked_fill(
            ~key_valid,
            torch.finfo(scores.dtype).min,
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = attention.masked_fill(
            ~key_valid,
            0.0,
        )

        attention = self.dropout(
            attention
        )

        #######################################################################
        # Weighted values.
        #######################################################################

        attended = torch.einsum(
            "bnghsmt,bnghtmd->bnghsmd",
            attention,
            V,
        )

        #######################################################################
        # (B,N,G,H_a,S,K,Dh)
        # ->
        # (B,N,G,S,K,D)
        #######################################################################

        attended = attended.permute(
            0,
            1,
            2,
            4,
            5,
            3,
            6,
        ).reshape(
            B,
            N,
            padded_H,
            K_modes,
            D,
        )

        return attended[:, :, :H]

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        spatial_embeddings: Tensor,
        *,
        agent_mask: Tensor | None = None,
    ) -> Tensor:

        if spatial_embeddings.ndim != 5:
            raise ValueError(
                "spatial_embeddings must have shape "
                "(B,N,H,K,D)."
            )

        B, N, H, K_modes, D = (
            spatial_embeddings.shape
        )

        if D != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {D}."
            )

        if agent_mask is not None:

            if agent_mask.shape != (
                B,
                N,
            ):
                raise ValueError(
                    "agent_mask must have shape (B,N)."
                )

        #######################################################################
        # Multi-scale temporal attention.
        #######################################################################

        outputs = [
            self._apply_window(
                spatial_embeddings,
                size,
            )
            for size in self.window_sizes
        ]

        #######################################################################
        # Concatenate temporal scales.
        #######################################################################

        output = torch.cat(
            outputs,
            dim=-1,
        )

        #######################################################################
        # Eq. (23)
        #######################################################################

        output = self.output_projection(
            output
        )

        output = self.dropout(
            output
        )

        #######################################################################
        # Invalid agents.
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
            f"window_sizes={self.window_sizes}"
        )


__all__ = ["MHCA"]
