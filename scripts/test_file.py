import torch

from models.encoders.relative_embedding import RelativeEmbedding


def main():

    model = RelativeEmbedding(
        hidden_dim=256,
    )

    positions = torch.randn(
        2,
        32,
        2,
    )

    headings = torch.randn(
        2,
        32,
    )

    output = model(
        positions,
        headings,
    )

    print(model)

    print()

    print("Positions")

    print(positions.shape)

    print()

    print("Headings")

    print(headings.shape)

    print()

    print("Embedding")

    print(output.shape)


if __name__ == "__main__":
    main()
