import torch

from models.encoders.lane_encoder import LaneEncoder


def main():

    model = LaneEncoder()

    lanes = torch.randn(
        2,      # batch
        128,    # lanes
        20,     # sampled points
        2,
    )

    output = model(
        lanes,
    )

    print(model)
    print()

    print("Input :", lanes.shape)
    print("Output:", output.shape)

    assert output.shape == (
        2,
        128,
        256,
    )

    print("\nLaneEncoder test passed.")


if __name__ == "__main__":
    main()
