"""
datasets/scene_data.py

Tensor representation of one processed Argoverse scene.

RawScene
    ↓
Geometry
    ↓
Feature Extraction
    ↓
Graph Builder
    ↓
SceneData
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(slots=True)
class SceneData:
    """
    Processed scene used by the PyTorch Dataset.

    Every field is already converted into tensors and normalized.
    """

    ###########################################################################
    # Metadata
    ###########################################################################

    sequence_id: str

    city: str

    ###########################################################################
    # Agent Features
    ###########################################################################

    agent_features: torch.Tensor
    """
    Shape:
        (N_agents, observation_steps, feature_dim)
    """

    ###########################################################################
    # Lane Features
    ###########################################################################

    lane_features: torch.Tensor
    """
    Shape:
        (N_lanes, lane_points, lane_feature_dim)
    """

    ###########################################################################
    # Future Target
    ###########################################################################

    target_future: torch.Tensor
    """
    Shape:
        (prediction_steps, 2)
    """

    ###########################################################################
    # Graph Connectivity
    ###########################################################################

    agent_agent_edge_index: torch.Tensor
    """
    Shape:
        (2, Eaa)
    """

    lane_agent_edge_index: torch.Tensor
    """
    Shape:
        (2, Ela)
    """

    ###########################################################################
    # Masks
    ###########################################################################

    agent_mask: torch.Tensor

    lane_mask: torch.Tensor

    ###########################################################################
    # Optional Metadata
    ###########################################################################

    rotation: Optional[torch.Tensor] = None

    translation: Optional[torch.Tensor] = None

    ###########################################################################
    # Validation
    ###########################################################################

    def __post_init__(self) -> None:

        if self.agent_features.ndim != 3:
            raise ValueError(
                "agent_features must be 3D."
            )

        if self.lane_features.ndim != 3:
            raise ValueError(
                "lane_features must be 3D."
            )

        if self.target_future.ndim != 2:
            raise ValueError(
                "target_future must be 2D."
            )

        if self.target_future.shape[1] != 2:
            raise ValueError(
                "target_future must have shape (T,2)."
            )

        if self.agent_agent_edge_index.ndim != 2:
            raise ValueError(
                "agent_agent_edge_index must be 2D."
            )

        if self.lane_agent_edge_index.ndim != 2:
            raise ValueError(
                "lane_agent_edge_index must be 2D."
            )

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def num_agents(self) -> int:
        return self.agent_features.shape[0]

    @property
    def num_lanes(self) -> int:
        return self.lane_features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.agent_features.shape[-1]

    @property
    def lane_feature_dim(self) -> int:
        return self.lane_features.shape[-1]

    @property
    def observation_steps(self) -> int:
        return self.agent_features.shape[1]

    @property
    def prediction_steps(self) -> int:
        return self.target_future.shape[0]

    ###########################################################################
    # Device Utilities
    ###########################################################################

    def to(
        self,
        device: torch.device | str,
    ) -> "SceneData":

        return SceneData(
            sequence_id=self.sequence_id,
            city=self.city,
            agent_features=self.agent_features.to(device),
            lane_features=self.lane_features.to(device),
            target_future=self.target_future.to(device),
            agent_agent_edge_index=self.agent_agent_edge_index.to(device),
            lane_agent_edge_index=self.lane_agent_edge_index.to(device),
            agent_mask=self.agent_mask.to(device),
            lane_mask=self.lane_mask.to(device),
            rotation=None
            if self.rotation is None
            else self.rotation.to(device),
            translation=None
            if self.translation is None
            else self.translation.to(device),
        )

    ###########################################################################
    # Utilities
    ###########################################################################

    def pin_memory(self) -> "SceneData":

        return self.to("cpu")

    def summary(self) -> dict:

        return {
            "sequence_id": self.sequence_id,
            "city": self.city,
            "num_agents": self.num_agents,
            "num_lanes": self.num_lanes,
            "feature_dim": self.feature_dim,
            "prediction_steps": self.prediction_steps,
        }

    def __len__(self) -> int:
        return self.num_agents

    def __repr__(self) -> str:

        return (
            "SceneData("
            f"sequence={self.sequence_id}, "
            f"agents={self.num_agents}, "
            f"lanes={self.num_lanes})"
        )
