"""
models.attention.mhca

Multi-Head Historical Context Attention

DSTNet
Section III-D
Figure 5
Equations (16)-(23)

Pipeline

Input Features
        │
Historical Window Extraction
        │
Historical Causal Mask
        │
Multi-Window Attention
        │
Feature Fusion
        │
Residual
        │
LayerNorm
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.layers.normalization import LayerNorm


###############################################################################
# Historical Causal Mask
###############################################################################


class HistoricalMask(nn.Module):
    """
    Lower triangular causal mask.

    Each timestep may only attend to
    itself and previous history.
    """

    def forward(
        self,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:

        return torch.tril(
            torch.ones(
                window_size,
                window_size,
                dtype=torch.bool,
                device=device,
            )
        )


###############################################################################
# Sliding Window Extraction
###############################################################################


class SlidingWindowExtractor(nn.Module):
    """
    Construct historical windows.

    Example

    Window Size = 4

        t0

        t0 t1

        t0 t1 t2

        t0 t1 t2 t3

        t1 t2 t3 t4

        ...

    Returns

        (B,N,S,C)
    """

    def __init__(
        self,
        window_size: int,
    ) -> None:

        super().__init__()

        self.window_size = int(window_size)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, channels = features.shape

        windows = []

        for t in range(sequence_length):

            start = max(
                0,
                t - self.window_size + 1,
            )

            window = features[
                :,
                start:t + 1,
                :,
            ]

            if window.shape[1] < self.window_size:

                pad = features.new_zeros(
                    batch_size,
                    self.window_size - window.shape[1],
                    channels,
                )

                window = torch.cat(
                    (
                        pad,
                        window,
                    ),
                    dim=1,
                )

            windows.append(
                window.unsqueeze(1)
            )

        return torch.cat(
            windows,
            dim=1,
        )


###############################################################################
# MHCA
###############################################################################


class MHCA(nn.Module):
    """
    Multi-Head Historical Context Attention.

    Figure 5

    Equations (16)-(23)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_sizes: tuple[int, ...] = (
            2,
            4,
            8,
        ),
        causal: bool = True,
        use_multi_scale: bool = True,
    ) -> None:

        super().__init__()

        if hidden_dim % num_heads != 0:

            raise ValueError(
                "hidden_dim must be divisible "
                "by num_heads."
            )

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.window_sizes = tuple(window_sizes)

        self.causal = causal

        self.use_multi_scale = use_multi_scale

        #######################################################################
        # Learnable Q K V
        #######################################################################

        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        #######################################################################
        # Output Projection
        #######################################################################

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        #######################################################################
        # Historical Window Extractors
        #######################################################################

        self.window_extractors = nn.ModuleDict(
            {
                str(size): SlidingWindowExtractor(
                    size,
                )
                for size in self.window_sizes
            }
        )

        #######################################################################
        # Shared Historical Mask
        #######################################################################

        self.history_mask = HistoricalMask()

        #######################################################################

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = LayerNorm(
            hidden_dim,
        )

