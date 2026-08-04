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
from torch import Tensor, nn

from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm


###############################################################################
# Historical Causal Mask
###############################################################################


class HistoricalMask(nn.Module):
    """
    Lower-triangular causal mask.

    Future timesteps are masked so that each timestep only attends
    to itself and its past history.
    """

    def forward(
        self,
        window_size: int,
        device: torch.device,
    ) -> Tensor:

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
    Extract fixed-length historical windows.

    Input
    -----
    (B,N,T,C)

    Output
    ------
    (B,N,T,W,C)

    where

        W = window size
    """

    def __init__(
        self,
        window_size: int,
    ) -> None:

        super().__init__()

        self.window_size = int(window_size)

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:

        B, N, T, C = features.shape

        windows = []

        for t in range(T):

            start = max(
                0,
                t - self.window_size + 1,
            )

            window = features[
                :,
                :,
                start:t + 1,
                :,
            ]

            if window.shape[2] < self.window_size:

                pad = features.new_zeros(
                    B,
                    N,
                    self.window_size - window.shape[2],
                    C,
                )

                window = torch.cat(
                    (
                        pad,
                        window,
                    ),
                    dim=2,
                )

            windows.append(
                window.unsqueeze(2)
            )

        return torch.cat(
            windows,
            dim=2,
        )


###############################################################################
# Multi-Head Historical Context Attention
###############################################################################


class MHCA(nn.Module):
    """
    Multi-Head Historical Context Attention.

    Implements the temporal branch of the Tri-Attention Module.
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
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_dim % num_heads != 0:

            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.hidden_dim = hidden_dim

        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.window_sizes = tuple(window_sizes)

        #######################################################################
        # QKV Projection
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
        # Historical Windows
        #######################################################################

        self.window_extractors = nn.ModuleDict(
            {
                str(size): SlidingWindowExtractor(size)
                for size in self.window_sizes
            }
        )

        self.history_mask = HistoricalMask()

        #######################################################################
        # Multi-scale Fusion
        #######################################################################

        self.window_weights = nn.Parameter(
            torch.ones(
                len(self.window_sizes),
            )
        )

        #######################################################################
        # Output Projection
        #######################################################################

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        #######################################################################
        # Feed Forward
        #######################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            expansion=expansion,
            dropout=dropout,
        )

        #######################################################################
        # Normalization
        #######################################################################

        self.norm = LayerNorm(
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    ###########################################################################
    # Head Utilities
    ###########################################################################

    def _split_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        (B,N,T,C)
            ->
        (B,N,H,T,D)
        """

        B, N, T, _ = x.shape

        x = x.view(
            B,
            N,
            T,
            self.num_heads,
            self.head_dim,
        )

        x = x.permute(
            0,
            1,
            3,
            2,
            4,
        )

        return x

    def _merge_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        (B,N,H,T,D)
            ->
        (B,N,T,C)
        """

        B, N, H, T, D = x.shape

        x = x.permute(
            0,
            1,
            3,
            2,
            4,
        )

        return x.reshape(
            B,
            N,
            T,
            H * D,
        )

    ###########################################################################
    # Historical Attention
    ###########################################################################

    def _historical_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Historical causal attention.

        Input
        -----
        (B,N,W,C)

        Output
        ------
        (B,N,W,C)
        """

        B, N, W, _ = query.shape

        q = self._split_heads(
            self.q_proj(query)
        )

        k = self._split_heads(
            self.k_proj(key)
        )

        v = self._split_heads(
            self.v_proj(value)
        )

        #######################################################################
        # Attention Scores
        #######################################################################

        scores = torch.matmul(
            q,
            k.transpose(
                -2,
                -1,
            ),
        )

        scores = scores / math.sqrt(
            self.head_dim,
        )

        #######################################################################
        # Historical Mask
        #######################################################################

        mask = self.history_mask(
            W,
            query.device,
        )

        scores = scores.masked_fill(
            ~mask.view(
                1,
                1,
                1,
                W,
                W,
            ),
            torch.finfo(scores.dtype).min,
        )

        #######################################################################
        # Softmax
        #######################################################################

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention,
        )

        #######################################################################
        # Aggregate
        #######################################################################

        output = torch.matmul(
            attention,
            v,
        )

        output = self._merge_heads(
            output,
        )

        return output

    ###########################################################################
    # One Temporal Window
    ###########################################################################

    def _window_attention(
        self,
        windows: Tensor,
    ) -> Tensor:
        """
        Apply MHCA inside every temporal window.

        Input
        -----
        windows

        Shape

            (B,N,T,W,C)

        Returns
        -------
            (B,N,T,C)
        """

        B, N, T, W, C = windows.shape

        outputs = []

        for t in range(T):

            window = windows[
                :,
                :,
                t,
            ]

            attended = self._historical_attention(
                window,
                window,
                window,
            )

            outputs.append(
                attended[
                    :,
                    :,
                    -1,
                ].unsqueeze(2)
            )

        return torch.cat(
            outputs,
            dim=2,
        )

    ###########################################################################
    # Multi-scale Fusion
    ###########################################################################

    def _multi_scale_attention(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Apply multiple historical window sizes and fuse their outputs.

        Parameters
        ----------
        features
            Shape (B,N,T,C)

        Returns
        -------
        Tensor
            Shape (B,N,T,C)
        """

        outputs = []

        #######################################################################
        # Multi-scale windows
        #######################################################################

        for window_size in self.window_sizes:

            windows = self.window_extractors[
                str(window_size)
            ](
                features,
            )

            outputs.append(
                self._window_attention(
                    windows,
                )
            )

        #######################################################################
        # Learnable fusion
        #######################################################################

        fused = torch.stack(outputs, dim=0)

        weights = torch.softmax(
            self.window_weights,
            dim=0,
        ).view(-1, 1, 1, 1, 1)

        fused = (weights * fused).sum(dim=0)

        return fused

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Multi-Head Historical Context Attention.

        Parameters
        ----------
        features
            Shape (B,N,T,C)

        Returns
        -------
        Tensor
            Shape (B,N,T,C)
        """

        residual = features

        #######################################################################
        # Pre-Norm
        #######################################################################

        x = self.norm(
            features,
        )

        #######################################################################
        # Multi-scale Historical Attention
        #######################################################################

        x = self._multi_scale_attention(
            x,
        )

        #######################################################################
        # Output Projection
        #######################################################################

        x = self.out_proj(
            x,
        )

        x = self.dropout(
            x,
        )

        #######################################################################
        # Residual
        #######################################################################

        x = residual + x

        #######################################################################
        # Feed Forward
        #######################################################################

        x = self.feed_forward(
            x,
        )

        return x

    ###########################################################################
    # Representation
    ###########################################################################

    def extra_repr(
        self,
    ) -> str:

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"window_sizes={self.window_sizes}"
        )

