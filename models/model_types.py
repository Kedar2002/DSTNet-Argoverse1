"""
models.model_types

Strongly-typed data structures used throughout the DSTNet model.

These classes define the interfaces exchanged between model
components. They are intentionally lightweight and contain only
tensors or immutable metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


###############################################################################
# Relative Features
###############################################################################


@dataclass(slots=True)
class RelativeFeatures:
    """
    Pairwise relative geometric features.

    Shapes
    ------
    dx               (B,N,N)

    dy               (B,N,N)

    distance         (B,N,N)

    heading_delta    (B,N,N)

    embedding        (B,N,N,C)
    """

    dx: torch.Tensor

    dy: torch.Tensor

    distance: torch.Tensor

    heading_delta: torch.Tensor

    embedding: torch.Tensor


###############################################################################
# Graph Data
###############################################################################


@dataclass(slots=True)
class GraphData:
    """
    Graph representation used by the encoder.
    """

    adjacency: torch.Tensor

    edge_index: torch.Tensor

    edge_features: RelativeFeatures

    agent_mask: torch.Tensor | None = None

    lane_mask: torch.Tensor | None = None


###############################################################################
# Prediction
###############################################################################


@dataclass(slots=True)
class Prediction:
    """
    Decoder output.
    """

    trajectories: torch.Tensor

    scores: torch.Tensor


###############################################################################
# Refined Prediction
###############################################################################


@dataclass(slots=True)
class RefinedPrediction:
    """
    Final refined trajectories.
    """

    trajectories: torch.Tensor

    scores: torch.Tensor

    offsets: torch.Tensor | None = None

###############################################################################
# Encoded Graph
###############################################################################

@dataclass(slots=True)
class EncodedGraph:
    """
    Output of GraphEncoder.
    """

    agent_features: torch.Tensor

    lane_features: torch.Tensor

    graph: GraphData
