"""
models.attention.tri_atm

Tri-Attention Spatio-Temporal Module (Tri-ATM)

DSTNet implementation.

Pipeline

MSPA
    ↓
MHCA
    ↓
MMIA
    ↓
Feed Forward
"""

from __future__ import annotations

import math

import torch
from torch import nn

from models.layers.attention import MultiHeadAttention
from models.layers.feed_forward import FeedForward
from models.layers.normalization import LayerNorm
from models.model_types import GraphData
from models.model_types import RelativeFeatures

###############################################################################
# Constants
###############################################################################

DEFAULT_RADII = (
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    60.0,
    float("inf"),
)

###############################################################################
# Geometry Utilities
###############################################################################


def pairwise_distance(
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    positions

        (B,N,2)

    Returns
    -------
    (B,N,N)
    """

    delta = (
        positions[:, :, None, :]
        -
        positions[:, None, :, :]
    )

    return torch.linalg.norm(
        delta,
        dim=-1,
    )

###############################################################################
# Sparse Neighborhood Builder
###############################################################################


class SpatialNeighborhoodBuilder(nn.Module):
    """
    Multi-scale neighborhood construction.

    Builds one sparse neighborhood graph
    for every attention head.
    """

    def __init__(
        self,
        radii: tuple[float, ...] = DEFAULT_RADII,
    ) -> None:

        super().__init__()

        self.radii = radii

    def forward(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        positions

            (B,N,2)

        Returns
        -------
        Tensor

            (B,H,N,N)
        """

        distance = pairwise_distance(
            positions,
        )

        masks = []

        for radius in self.radii:

            if math.isinf(radius):

                mask = torch.ones_like(
                    distance,
                    dtype=torch.bool,
                )

            else:

                mask = distance <= radius

            masks.append(mask)

        return torch.stack(
            masks,
            dim=1,
        )

    def __repr__(
        self,
    ) -> str:

        return (
            "SpatialNeighborhoodBuilder("
            f"heads={len(self.radii)})"
        )

###############################################################################
# Relative Attention Bias
###############################################################################


class RelativeAttentionBias(nn.Module):
    """
    Convert relative embeddings into
    attention logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ) -> None:

        super().__init__()

        self.projection = nn.Linear(
            hidden_dim,
            num_heads,
        )

    def forward(
        self,
        relative: RelativeFeatures,
    ) -> torch.Tensor:
        """
        Returns

            (B,H,N,N)
        """

        bias = self.projection(
            relative.embedding,
        )

        return bias.permute(
            0,
            3,
            1,
            2,
        )

###############################################################################
# Temporal Utilities
###############################################################################


class TemporalCausalMask(nn.Module):
    """
    Lower triangular causal attention mask.

    Future timesteps cannot be attended.
    """

    def forward(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        )

###############################################################################
# Multi-Scale Spatial Sparse Attention
###############################################################################


class MSPA(nn.Module):
    """
    Multi-Scale Spatial Sparse Attention.

    Implements the spatial interaction stage of Tri-ATM.

    Inputs
    ------
        features : (B,N,C)

        positions : (B,N,2)

        relative : RelativeFeatures

    Output
    ------
        (B,N,C)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        radii: tuple[float, ...] = DEFAULT_RADII,
    ) -> None:

        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        #######################################################################
        # Linear projections
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

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        #######################################################################
        # Spatial utilities
        #######################################################################

        self.neighborhood_builder = (
            SpatialNeighborhoodBuilder(
                radii=radii,
            )
        )

        self.relative_bias = (
            RelativeAttentionBias(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
        )

        #######################################################################

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = LayerNorm(
            hidden_dim,
        )

    ###########################################################################

    def _reshape(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        B, N, _ = x.shape

        x = x.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(
            1,
            2,
        )

    ###########################################################################

    def forward(
        self,
        *,
        features: torch.Tensor,
        positions: torch.Tensor,
        relative: RelativeFeatures,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        residual = features

        if positions.ndim != 3:
            raise ValueError(
                "positions must have shape (B,N,2)."
            )

        #######################################################################
        # QKV
        #######################################################################

        q = self._reshape(
            self.q_proj(features)
        )

        k = self._reshape(
            self.k_proj(features)
        )

        v = self._reshape(
            self.v_proj(features)
        )

        #######################################################################
        # Scaled Dot Product
        #######################################################################

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores /= math.sqrt(
            self.head_dim,
        )

        #######################################################################
        # Relative Geometry
        #######################################################################

        scores = (
            scores
            +
            self.relative_bias(relative)
        )

        #######################################################################
        # Sparse Neighborhood
        #######################################################################

        sparse_mask = (
            self.neighborhood_builder(
                positions,
            )
        )

        if mask is not None:

            mask = mask[:, None, None, :]

            sparse_mask = (
                sparse_mask
                &
                mask
            )

        scores = scores.masked_fill(
            ~sparse_mask,
            torch.finfo(scores.dtype).min,
        )

        #######################################################################
        # Attention
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

        output = output.transpose(
            1,
            2,
        )

        B, N, _ = features.shape

        output = output.reshape(
            B,
            N,
            self.hidden_dim,
        )

        output = self.out_proj(
            output,
        )

        #######################################################################
        # Residual
        #######################################################################

        output = residual + self.dropout(
            output,
        )

        output = self.norm(
            output,
        )

        return output

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "MSPA("
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads})"
        )

###############################################################################
# Multi-Scale Historical Context Attention
###############################################################################


class MHCA(nn.Module):
    """
    Multi-Scale Historical Context Attention.

    DSTNet does not explicitly define the internal attention
    formulation of MHCA.

    Therefore this implementation follows the standard
    Transformer Multi-Head Attention while preserving the
    module ordering and interfaces described in the paper.

    Input
    -----
        features

            (B,N,C)

    Output
    ------
        (B,N,C)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = LayerNorm(
            hidden_dim,
        )

    ###########################################################################

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query

            (B,N,C)

        key

            (B,M,C)

        value

            (B,M,C)

        mask

            Optional attention mask.

        attention_bias

            Optional relative attention bias.

        Returns
        -------
        Tensor

            (B,N,C)
        """

        residual = query

        output = self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            attention_bias=attention_bias,
        )

        output = residual + self.dropout(
            output,
        )

        output = self.norm(
            output,
        )

        return output

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "MHCA("
            f"hidden_dim={self.hidden_dim})"
        )

###############################################################################
# Multi-Mode Interaction Attention
###############################################################################


class MMIA(nn.Module):
    """
    Multi-Mode Interaction Attention.

    DSTNet describes MMIA as the final interaction stage of
    Tri-ATM. The paper does not provide the complete internal
    implementation.

    This implementation performs feature interaction between
    the spatial (MSPA) and contextual (MHCA) features using a
    learnable fusion MLP followed by residual refinement.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        #######################################################################
        # Feature Fusion
        #######################################################################

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        #######################################################################

        self.dropout = nn.Dropout(
            dropout,
        )

        self.norm = LayerNorm(
            hidden_dim,
        )

    ###########################################################################

    def forward(
        self,
        *,
        spatial_features: torch.Tensor,
        contextual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_features

            (B,N,C)

        contextual_features

            (B,N,C)

        Returns
        -------
        Tensor

            (B,N,C)
        """

        residual = spatial_features

        ###################################################################
        # Feature Interaction
        ###################################################################

        fused = torch.cat(
            (
                spatial_features,
                contextual_features,
            ),
            dim=-1,
        )

        fused = self.fusion(
            fused,
        )

        ###################################################################
        # Residual
        ###################################################################

        fused = residual + self.dropout(
            fused,
        )

        fused = self.norm(
            fused,
        )

        return fused

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "MMIA("
            f"hidden_dim={self.hidden_dim})"
        )

###############################################################################
# Tri-Attention Spatio-Temporal Module
###############################################################################


class TriATM(nn.Module):
    """
    Tri-Attention Spatio-Temporal Module.

    Pipeline

        MSPA
            ↓
        MHCA
            ↓
        MMIA
            ↓
        Feed Forward

    This module represents one encoder block of DSTNet.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim

        #######################################################################
        # Attention Stages
        #######################################################################

        self.mspa = MSPA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.mhca = MHCA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.mmia = MMIA(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        #######################################################################
        # Feed Forward
        #######################################################################

        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    ###########################################################################

    def forward(
        self,
        *,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        relative: RelativeFeatures,
        graph: GraphData,
        positions: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        agent_features

            (B,N,C)

        lane_features

            (B,L,C)

        relative

            RelativeFeatures

        graph

            GraphData

        positions

            (B,N,2)

        Returns
        -------
        agent_features

            (B,N,C)

        lane_features

            (B,L,C)
        """

        _ = graph

        ###################################################################
        # Multi-Scale Spatial Sparse Attention
        ###################################################################

        spatial_features = self.mspa(
            features=agent_features,
            positions=positions,
            relative=relative,
            mask=agent_mask,
        )

        ###################################################################
        # Historical / Context Attention
        ###################################################################

        contextual_features = self.mhca(
            query=spatial_features,
            key=spatial_features,
            value=spatial_features,
            mask=agent_mask,
        )

        ###################################################################
        # Interaction
        ###################################################################

        fused_features = self.mmia(
            spatial_features=spatial_features,
            contextual_features=contextual_features,
        )

        ###################################################################
        # Feed Forward
        ###################################################################

        fused_features = self.feed_forward(
            fused_features,
        )

        ###################################################################
        # Return
        ###################################################################

        return (
            fused_features,
            lane_features,
        )

    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "TriATM("
            f"hidden_dim={self.hidden_dim})"
        )


