"""
scripts.test_mhca

Unit test for Multi-Head Cross Attention.
"""

import torch

from models.attention.mhca import MHCA


def main():

    print("=" * 80)
    print("MHCA Test")
    print("=" * 80)

    B = 2
    Nq = 12
    Nk = 20
    C = 256

    query = torch.randn(
        B,
        Nq,
        C,
    )

    context = torch.randn(
        B,
        Nk,
        C,
    )

    mask = torch.ones(
        B,
        Nk,
        dtype=torch.bool,
    )

    model = MHCA(
        hidden_dim=C,
        num_heads=8,
    )

    print(model)
    print()

    ###############################################################
    # Forward
    ###############################################################

    output = model(
        query,
        context,
        context_mask=mask,
    )

    print("Query Shape  :", query.shape)
    print("Context Shape:", context.shape)
    print("Output Shape :", output.shape)

    assert output.shape == (
        B,
        Nq,
        C,
    )

    ###############################################################
    # Attention Weights
    ###############################################################

    output, weights = model(
        query,
        context,
        context_mask=mask,
        return_attention=True,
    )

    print()

    print("Attention Shape:", weights.shape)

    assert weights.shape == (
        B,
        8,
        Nq,
        Nk,
    )

    print()

    print("✓ MHCA test passed")


if __name__ == "__main__":
    main()
