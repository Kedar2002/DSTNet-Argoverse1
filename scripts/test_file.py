import torch

from models.layers.mlp import MLP


def main():

    model = MLP(
        input_dim=16,
        hidden_dims=[32, 64],
        output_dim=8,
        dropout=0.1,
    )

    x = torch.randn(
        4,
        16,
    )

    y = model(x)

    print(model)

    print()

    print(x.shape)

    print(y.shape)


if __name__ == "__main__":
    main()
