import torch

from models.encoders.agent_encoder import AgentEncoder
from models.encoders.lane_encoder import LaneEncoder
from models.encoders.relative_embedding import RelativeEmbedding
from models.attention.tri_atm import TriATM


def main():

    B = 2
    N = 16
    L = 24
    T = 20
    H = 256

    ############################################################

    agent_traj = torch.randn(B, N, T, 2)
    lane_points = torch.randn(B, L, T, 2)

    positions = agent_traj[:, :, -1]

    headings = torch.randn(B, N)

    ############################################################

    agent_encoder = AgentEncoder(
        observation_steps=T,
        hidden_dim=H,
    )

    lane_encoder = LaneEncoder(
        num_points=T,
        hidden_dim=H,
    )

    relative_embedding = RelativeEmbedding(
        hidden_dim=H,
    )

    ############################################################

    agent_features = agent_encoder(agent_traj)

    lane_features = lane_encoder(lane_points)

    relative = relative_embedding(
        positions,
        headings,
    )

    ############################################################

    triatm = TriATM(
        hidden_dim=H,
        num_heads=8,
    )

    agent_features, lane_features = triatm(
        agent_features=agent_features,
        lane_features=lane_features,
        relative=relative,
        graph=None,
        positions=positions,
    )

    ############################################################

    print()

    print("=" * 60)

    print("TriATM Test Passed")

    print("=" * 60)

    print()

    print("Agent Features :", agent_features.shape)

    print("Lane Features  :", lane_features.shape)

    print()

    assert agent_features.shape == (B, N, H)

    assert lane_features.shape == (B, L, H)

    print("✓ Shapes Verified")


if __name__ == "__main__":
    main()
