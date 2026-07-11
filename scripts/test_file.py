import torch

from models.layers.feed_forward import FeedForward


def main():

    model = FeedForward(
        hidden_dim=256,
        expansion=4,
        dropout=0.1,
    )

    x = torch.randn(
        8,
        20,
        256,
    )

    y = model(x)

    print(model)

    print()

    print("Input")

    print(x.shape)

    print()

    print("Output")

    print(y.shape)


if __name__ == "__main__":
    main()
