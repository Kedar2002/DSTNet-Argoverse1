"""
models.layers.mode_embedding

Learnable mode embedding for DSTNet.

The original DSTNet paper performs Multi-Mode Interaction
Attention (MMIA) on mode-specific embeddings Z_ST(n,t,k),
but does not explicitly describe how the initial K mode
embeddings are created.

This module introduces a lightweight learnable mode
embedding that expands encoder features into K prediction
modes before MMIA.

Input
-----
(B, N, C)

Output
------
(B, N, K, C)
"""

from __future__ import annotations

import torch
from torch import nn


class ModeEmbedding(nn.Module):
    """
    Learnable mode embedding.

    Parameters
    ----------
    hidden_dim
        Feature dimension.

    num_modes
        Number of prediction modes.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_modes: int = 6,
    ) -> None:

        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_modes = num_modes

        ###################################################################
        # Learnable mode tokens
        ###################################################################

        self.mode_tokens = nn.Parameter(
            torch.empty(
                num_modes,
                hidden_dim,
            )
        )

        ###############################################################################
        # Mode Projection
        ###############################################################################

        self.projection = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
        )

        nn.init.xavier_uniform_(
            self.mode_tokens,
        )

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_modes(self) -> int:
        return self._num_modes

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features

            Shape (B,N,C)

        Returns
        -------
        Tensor

            Shape (B,N,K,C)
        """

        if features.ndim != 3:
            raise ValueError(
                "Expected input shape (B,N,C)."
            )

        batch_size, num_agents, hidden_dim = features.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension "
                f"{self.hidden_dim}, got {hidden_dim}."
            )

        ###################################################################
        # Expand encoder features
        ###################################################################

        features = features.unsqueeze(
            2,
        )

        features = features.expand(
            batch_size,
            num_agents,
            self.num_modes,
            hidden_dim,
        )

        ###################################################################
        # Add learnable mode embedding
        ###################################################################

        mode_tokens = self.mode_tokens.view(
            1,
            1,
            self.num_modes,
            hidden_dim,
        )

        features = features + mode_tokens

        features = self.projection(
            features,
        )
        
        return features

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "ModeEmbedding("
            f"hidden_dim={self.hidden_dim}, "
            f"num_modes={self.num_modes})"
        )
