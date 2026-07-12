"""
scripts.test_mmia

Unit test for Multi-Modal Interaction Attention.
"""

import torch

from models.attention.mmia import MMIA


def main():

    print("=" * 80)
    print("MMIA Test")
    print("=" * 80)

    B = 2
    N = 16
    C = 256

    spatial = torch.randn(
        B,
        N,
        C,
    )

    cross = torch.randn(
        B,
        N,
        C,
    )

    model = MMIA(
        hidden_dim=C,
    )

    print(model)
    print()

    output = model(
        spatial,
        cross,
    )

    print("Spatial :", spatial.shape)
    print("Cross   :", cross.shape)
    print("Output  :", output.shape)

    assert output.shape == (
        B,
        N,
        C,
    )

    print()

    print("✓ MMIA test passed")


if __name__ == "__main__":
    main()
