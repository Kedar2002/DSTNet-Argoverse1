"""
Unit test for GSTA.

Run:

python tests/test_gsta.py
"""

import torch

from models.encoders.gsta import GSTA
from models.model_types import RelativeFeatures, GraphData


def create_relative_features(
    batch_size: int,
    num_agents: int,
    num_lanes: int,
):
    """
    Creates dummy RelativeFeatures.
    """

    return RelativeFeatures(
        relative_positions=torch.randn(
            batch_size,
            num_agents,
            num_lanes,
            2,
        ),
        relative_headings=torch.randn(
            batch_size,
            num_agents,
            num_lanes,
        ),
        distances=torch.rand(
            batch_size,
            num_agents,
            num_lanes,
        ),
    )


def create_graph():
    """
    Dummy graph.
    """

    return GraphData(
        edge_index=torch.empty(
            2,
            0,
            dtype=torch.long,
        ),
        edge_attr=None,
        num_nodes=0,
    )


def main():

    torch.manual_seed(42)

    batch_size = 2
    num_agents = 32
    num_lanes = 64
    hidden_dim = 128

    model = GSTA(
        hidden_dim=hidden_dim,
        num_heads=8,
        num_queries=6,
        dropout=0.1,
    )

    model.eval()

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

    relative = create_relative_features(
        batch_size,
        num_agents,
        num_lanes,
    )

    graph = create_graph()

    with torch.no_grad():

        out_agents, out_lanes = model(
            agent_features,
            lane_features,
            relative,
            graph,
        )

    print()

    print("=" * 60)
    print("GSTA TEST")
    print("=" * 60)

    print("Input agent shape :", agent_features.shape)
    print("Output agent shape:", out_agents.shape)

    print()

    print("Input lane shape  :", lane_features.shape)
    print("Output lane shape :", out_lanes.shape)

    print()

    scene = model.scene_embedding

    print("Scene embedding   :", scene.shape)

    print()

    assert out_agents.shape == (
        batch_size,
        num_agents,
        hidden_dim,
    )

    assert out_lanes.shape == (
        batch_size,
        num_lanes,
        hidden_dim,
    )

    assert scene.shape == (
        batch_size,
        6,
        hidden_dim,
    )

    print("✓ All GSTA tests passed.")


if __name__ == "__main__":
    main()
