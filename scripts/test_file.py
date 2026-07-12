"""
scripts.test_mspa

Unit tests for the Multi-head Spatial Attention module.
"""

import torch

from models.attention.mspa import MSPA


def main():

    print("=" * 80)
    print("MSPA Test")
    print("=" * 80)

    B = 2
    N = 16
    C = 256

    x = torch.randn(
        B,
        N,
        C,
    )

    mask = torch.ones(
        B,
        N,
        dtype=torch.bool,
    )

    model = MSPA(
        hidden_dim=C,
        num_heads=8,
        dropout=0.1,
    )

    print(model)
    print()

    #######################################################################
    # Forward
    #######################################################################

    y = model(
        x,
        mask=mask,
    )

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)

    assert y.shape == (
        B,
        N,
        C,
    )

    #######################################################################
    # Attention Weights
    #######################################################################

    y, weights = model(
        x,
        mask=mask,
        return_attention=True,
    )

    print()

    print("Attention Shape :", weights.shape)

    assert weights.shape == (
        B,
        8,
        N,
        N,
    )

    print()

    print("✓ MSPA test passed")


if __name__ == "__main__":
    main()
