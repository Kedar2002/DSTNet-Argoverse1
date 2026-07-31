import torch

from models.encoders.relative_embedding import RelativeEmbedding


def main():

    B = 2
    N = 6

    model = RelativeEmbedding()

    positions = torch.randn(
        B,
        N,
        2,
    )

    headings = torch.randn(
        B,
        N,
    )

    relative = model(
        positions=positions,
        headings=headings,
    )

    print("=" * 80)
    print("RelativeEmbedding Test")
    print("=" * 80)

    print("dx            :", relative.dx.shape)
    print("dy            :", relative.dy.shape)
    print("distance      :", relative.distance.shape)
    print("heading_delta :", relative.heading_delta.shape)
    print("embedding     :", relative.embedding.shape)

    assert relative.dx.shape == (B, N, N)
    assert relative.dy.shape == (B, N, N)
    assert relative.distance.shape == (B, N, N)
    assert relative.heading_delta.shape == (B, N, N)
    assert relative.embedding.shape == (B, N, N, 256)

    print()
    print("✓ RelativeEmbedding test passed")


if __name__ == "__main__":
    main()
