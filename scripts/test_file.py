"""
scripts.test_attention

Unit tests for the generic MultiHeadAttention layer.
"""

from __future__ import annotations

import torch

from models.layers.attention import MultiHeadAttention


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def test_self_attention() -> None:

    print_header("Self Attention")

    B = 2
    N = 16
    C = 256

    x = torch.randn(B, N, C)

    model = MultiHeadAttention(
        hidden_dim=C,
        num_heads=8,
    )

    y = model(
        x,
        x,
        x,
    )

    print("Input :", x.shape)
    print("Output:", y.shape)

    assert y.shape == (B, N, C)

    print("✓ Passed")


def test_cross_attention() -> None:

    print_header("Cross Attention")

    B = 2
    N = 12
    M = 20
    C = 256

    query = torch.randn(B, N, C)

    key = torch.randn(B, M, C)

    value = torch.randn(B, M, C)

    model = MultiHeadAttention(
        hidden_dim=C,
        num_heads=8,
    )

    output = model(
        query,
        key,
        value,
    )

    print("Query :", query.shape)
    print("Key   :", key.shape)
    print("Value :", value.shape)
    print("Output:", output.shape)

    assert output.shape == (B, N, C)

    print("✓ Passed")


def test_mask() -> None:

    print_header("Attention Mask")

    B = 2
    N = 12
    C = 256

    x = torch.randn(B, N, C)

    mask = torch.ones(
        B,
        N,
        dtype=torch.bool,
    )

    mask[:, -3:] = False

    model = MultiHeadAttention(
        hidden_dim=C,
        num_heads=8,
    )

    y = model(
        x,
        x,
        x,
        mask=mask,
    )

    print("Mask Shape :", mask.shape)
    print("Output     :", y.shape)

    assert y.shape == (B, N, C)

    print("✓ Passed")


def test_attention_bias() -> None:

    print_header("Attention Bias")

    B = 2
    N = 10
    C = 256

    x = torch.randn(B, N, C)

    bias = torch.randn(
        B,
        1,
        N,
        N,
    )

    model = MultiHeadAttention(
        hidden_dim=C,
        num_heads=8,
    )

    y = model(
        x,
        x,
        x,
        attention_bias=bias,
    )

    print("Bias Shape :", bias.shape)
    print("Output     :", y.shape)

    assert y.shape == (B, N, C)

    print("✓ Passed")


def test_return_attention() -> None:

    print_header("Return Attention Weights")

    B = 2
    N = 8
    C = 256

    x = torch.randn(B, N, C)

    model = MultiHeadAttention(
        hidden_dim=C,
        num_heads=8,
    )

    output, weights = model(
        x,
        x,
        x,
        return_attention=True,
    )

    print("Output Shape    :", output.shape)
    print("Attention Shape :", weights.shape)

    assert output.shape == (B, N, C)

    assert weights.shape == (
        B,
        8,
        N,
        N,
    )

    print("✓ Passed")


def main() -> None:

    print()
    print("=" * 80)
    print("DSTNet MultiHeadAttention Test")
    print("=" * 80)

    test_self_attention()

    test_cross_attention()

    test_mask()

    test_attention_bias()

    test_return_attention()

    print()
    print("=" * 80)
    print("All Attention Tests Passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
