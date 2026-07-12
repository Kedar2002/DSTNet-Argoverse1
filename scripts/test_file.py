import torch

from models.model_types import (
    GraphData,
    Prediction,
    RefinedPrediction,
    RelativeFeatures,
)


def main():

    relative = RelativeFeatures(
        dx=torch.randn(2, 8, 8),
        dy=torch.randn(2, 8, 8),
        distance=torch.randn(2, 8, 8),
        heading_delta=torch.randn(2, 8, 8),
        embedding=torch.randn(2, 8, 8, 256),
    )

    graph = GraphData(
        adjacency=torch.randint(0, 2, (2, 8, 8)),
        edge_index=torch.randint(0, 8, (2, 32)),
        edge_features=relative,
    )

    prediction = Prediction(
        trajectories=torch.randn(2, 6, 30, 2),
        scores=torch.randn(2, 6),
    )

    refined = RefinedPrediction(
        trajectories=torch.randn(2, 6, 30, 2),
        scores=torch.randn(2, 6),
    )

    print(relative)

    print()

    print(graph)

    print()

    print(prediction)

    print()

    print(refined)


if __name__ == "__main__":
    main()
