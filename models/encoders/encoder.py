"""
models.encoders.encoder

Top-level encoder for DSTNet.

Pipeline

Agent Features
Lane Features
        │
        ▼
RelativeEmbedding
        │
        ▼
GSTA
        │
        ▼
TriATM × L
        │
        ▼
Encoded Features
"""

from __future__ import annotations

from torch import nn
import torch

from models.attention.tri_atm import TriATM
from models.encoders.relative_embedding import RelativeEmbedding
from models.encoders.gsta import GSTA


class Encoder(nn.Module):
    """
    DSTNet encoder.

    The encoder consists of:

        RelativeEmbedding
                ↓
            GSTA
                ↓
        TriATM × num_layers
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.relative_embedding = RelativeEmbedding(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.gsta = GSTA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.layers = nn.ModuleList(
            [
                TriATM(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        positions: torch.Tensor,
        headings: torch.Tensor,
        graph,
        *,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        relative = self.relative_embedding(
            positions,
            headings,
        )

        agent_features, lane_features = self.gsta(
            agent_features=agent_features,
            lane_features=lane_features,
            relative=relative,
            graph=graph,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )

        for layer in self.layers:

            agent_features, lane_features = layer(
                agent_features=agent_features,
                lane_features=lane_features,
                relative=relative,
                graph=graph,
                positions=positions,
                agent_mask=agent_mask,
                lane_mask=lane_mask,
            )

        agent_features = self.norm(
            agent_features
        )

        lane_features = self.norm(
            lane_features
        )

        return (
            agent_features,
            lane_features,
        )

    def __repr__(self) -> str:

        return (
            "Encoder("
            f"hidden_dim={self.hidden_dim}, "
            f"layers={self.num_layers})"
        )


