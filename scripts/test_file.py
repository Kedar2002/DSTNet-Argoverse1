"""
Test GraphEncoder.
"""

import torch

from models.encoders.graph_encoder import GraphEncoder
from models.model_types import (
    EncodedGraph,
    GraphData,
    RelativeFeatures,
)


def main():

    batch_size = 2
    num_agents = 16
    num_lanes = 32
    hidden_dim = 256

    ###########################################################################
    # Relative Features
    ###########################################################################

    relative = RelativeFeatures(
        dx=torch.randn(
            batch_size,
            num_agents,
            num_agents,
        ),
        dy=torch.randn(
            batch_size,
            num_agents,
            num_agents,
        ),
        distance=torch.randn(
            batch_size,
            num_agents,
            num_agents,
        ),
        heading_delta=torch.randn(
            batch_size,
            num_agents,
            num_agents,
        ),
        embedding=torch.randn(
            batch_size,
            num_agents,
            num_agents,
            hidden_dim,
        ),
    )

    ###########################################################################
    # Graph
    ###########################################################################

    graph = GraphData(
        adjacency=torch.randint(
            0,
            2,
            (
                batch_size,
                num_agents,
                num_agents,
            ),
        ).bool(),
        edge_index=torch.randint(
            0,
            num_agents,
            (
                2,
                128,
            ),
        ),
        edge_features=relative,
        agent_mask=None,
        lane_mask=None,
    )

    ###########################################################################
    # Features
    ###########################################################################

    agent_features = torch.randn(
        batch_size,
        num_agents,
        hidden_dim,
    )

    lane_features = torch.randn(
        batch_size,
        num_lanes,
        hidden_dim,
    )

    ###########################################################################
    # Graph Encoder
    ###########################################################################

    encoder = GraphEncoder()

    encoded = encoder(
        agent_features,
        lane_features,
        graph,
    )

    ###########################################################################
    # Validation
    ###########################################################################

    assert isinstance(
        encoded,
        EncodedGraph,
    )

    assert encoded.agent_features.shape == (
        batch_size,
        num_agents,
        hidden_dim,
    )

    assert encoded.lane_features.shape == (
        batch_size,
        num_lanes,
        hidden_dim,
    )

    assert encoded.graph.adjacency.shape == (
        batch_size,
        num_agents,
        num_agents,
    )

    assert encoded.graph.edge_index.shape[0] == 2

    assert encoded.graph.edge_features.embedding.shape == (
        batch_size,
        num_agents,
        num_agents,
        hidden_dim,
    )

    print()

    print("=" * 60)
    print("GraphEncoder Test")
    print("=" * 60)

    print()

    print("Agent Features")
    print(encoded.agent_features.shape)

    print()

    print("Lane Features")
    print(encoded.lane_features.shape)

    print()

    print("Adjacency")
    print(encoded.graph.adjacency.shape)

    print()

    print("Edge Index")
    print(encoded.graph.edge_index.shape)

    print()

    print("Relative Embedding")
    print(
        encoded.graph.edge_features.embedding.shape
    )

    print()

    print("✓ GraphEncoder test passed.")

    print()


if __name__ == "__main__":
    main()
