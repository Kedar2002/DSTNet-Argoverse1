"""
scripts.test_gsta

Unit test for the GSTA layer.
"""

import torch

from models.encoders.gsta import GSTA


def main():

    print("=" * 80)
    print("GSTA Test")
    print("=" * 80)

    B = 2
    Na = 12
    Nl = 20
    C = 256

    agents = torch.randn(
        B,
        Na,
        C,
    )

    lanes = torch.randn(
        B,
        Nl,
        C,
    )

    agent_mask = torch.ones(
        B,
        Na,
        dtype=torch.bool,
    )

    lane_mask = torch.ones(
        B,
        Nl,
        dtype=torch.bool,
    )

    model = GSTA(
        hidden_dim=C,
        num_heads=8,
        dropout=0.1,
    )

    print(model)
    print()

    out_agents, out_lanes = model(
        agent_features=agents,
        lane_features=lanes,
        agent_mask=agent_mask,
        lane_mask=lane_mask,
    )

    print("Input Agent Shape :", agents.shape)
    print("Output Agent Shape:", out_agents.shape)
    print()

    print("Input Lane Shape  :", lanes.shape)
    print("Output Lane Shape :", out_lanes.shape)

    assert out_agents.shape == (
        B,
        Na,
        C,
    )

    assert out_lanes.shape == (
        B,
        Nl,
        C,
    )

    print()
    print("✓ GSTA test passed")


if __name__ == "__main__":
    main()
