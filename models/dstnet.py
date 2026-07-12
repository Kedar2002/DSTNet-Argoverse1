"""
models.dstnet

Complete DSTNet model.

Pipeline

Observed Agent Trajectories
Observed Lane Centerlines
            │
            ▼
AgentEncoder
LaneEncoder
            │
            ▼
Encoder (GSTA + TriATM)
            │
            ▼
ModeEmbedding
            │
            ▼
Decoder
            │
            ▼
Adaptive Anchor Refinement
            │
            ▼
Final Prediction
"""

from __future__ import annotations

import torch
from torch import nn

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.lane_encoder import LaneEncoder
from models.encoders.encoder import Encoder

from models.layers.mode_embedding import ModeEmbedding

from models.decoder.decoder import Decoder

from models.refinement.refinement import Refinement

from models.model_types import (
    ModeFeatures,
    RefinedPrediction,
)

class DSTNet(nn.Module):
    """
    Complete DSTNet.

    Implements the complete pipeline
    described in the paper.
    """
    def __init__(
        self,
        observation_steps: int = 20,
        prediction_steps: int = 30,
        lane_points: int = 20,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_modes: int = 6,
        refinement_iterations: int = 2,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        ###############################################################
        # Encoders
        ###############################################################

        self.agent_encoder = AgentEncoder(
            observation_steps=observation_steps,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.lane_encoder = LaneEncoder(
            num_points=lane_points,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        ###############################################################
        # Global Encoder
        ###############################################################

        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        ###############################################################
        # Mode Expansion
        ###############################################################

        self.mode_embedding = ModeEmbedding(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
        )

        ###############################################################
        # Decoder
        ###############################################################

        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            prediction_steps=prediction_steps,
            dropout=dropout,
        )

        ###############################################################
        # Refinement
        ###############################################################

        self.refinement = Refinement(
            hidden_dim=hidden_dim,
            prediction_steps=prediction_steps,
            refinement_iterations=refinement_iterations,
        )

    def forward(
        self,
        *,
        agent_trajectories: torch.Tensor,
        lane_centerlines: torch.Tensor,
        positions: torch.Tensor,
        headings: torch.Tensor,
        graph,
        agent_mask: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> RefinedPrediction:

        ###############################################################
        # Local Encoders
        ###############################################################

        agent_features = self.agent_encoder(
            agent_trajectories,
        )

        lane_features = self.lane_encoder(
            lane_centerlines,
        )

        ###############################################################
        # Global Encoder
        ###############################################################

        agent_features, lane_features = self.encoder(
            agent_features=agent_features,
            lane_features=lane_features,
            positions=positions,
            headings=headings,
            graph=graph,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )

        ###############################################################
        # Mode Expansion
        ###############################################################

        mode_features = self.mode_embedding(
            agent_features,
        )

        mode_features = ModeFeatures(
            features=mode_features,
        )

        ###############################################################
        # Coarse Trajectory Decoder
        ###############################################################

        prediction = self.decoder(
            mode_features,
        )

        ###############################################################
        # Adaptive Anchor-based Refinement
        ###############################################################

        refined_prediction = self.refinement(
            encoder_features=agent_features,
            prediction=prediction,
        )

        ###############################################################
        # Output
        ###############################################################

        return refined_prediction

    ###################################################################
    # Representation
    ###################################################################

    def __repr__(
        self,
    ) -> str:

        return (
            "DSTNet("
            f"agent_encoder={self.agent_encoder}, "
            f"lane_encoder={self.lane_encoder}, "
            f"encoder={self.encoder}, "
            f"decoder={self.decoder}, "
            f"refinement={self.refinement})"
        )


