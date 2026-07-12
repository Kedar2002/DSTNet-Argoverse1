"""
models.encoders.graph_encoder

Graph encoder for DSTNet.

This module prepares graph-aware tensors for the encoder stack.
No graph attention is performed here.

Responsibilities
----------------
* Collect encoded node features
* Prepare adjacency matrices
* Prepare edge features
* Prepare attention masks
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.model_types import (
    GraphData,
    RelativeFeatures,
    EncodedGraph,
)


class GraphEncoder(nn.Module):
    """
    Graph preparation module.

    This module converts encoded scene features into the graph
    representation consumed by GSTA and the Tri-ATM blocks.
    """

    def __init__(self) -> None:

        super().__init__()

    ###########################################################################
    # Forward
    ###########################################################################

    def forward(
        self,
        agent_features: torch.Tensor,
        lane_features: torch.Tensor,
        graph: GraphData,
    ) -> EncodedGraph:

        return EncodedGraph(
            agent_features=agent_features,
            lane_features=lane_features,
            graph=graph,
        )

    ###########################################################################
    # Representation
    ###########################################################################

    def __repr__(
        self,
    ) -> str:

        return "GraphEncoder()"
