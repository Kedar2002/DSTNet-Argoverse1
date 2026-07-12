import torch

from models.attention.mspa import MSPA


def main():

    model = MSPA(
        hidden_dim=256,
        num_heads=8,
    )

    x = torch.randn(
        2,
        32,
        256,
    )

    y = model(x)

    print(model)

    print()

    print("Input :", x.shape)

    print("Output:", y.shape)


if __name__ == "__main__":
    main()
