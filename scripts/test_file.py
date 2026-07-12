"""
scripts.test_collate

Integration test for the complete batching pipeline.

Pipeline

CSV
    ↓
SceneParser
    ↓
ScenePreprocessor
    ↓
SceneData
    ↓
DataLoader
    ↓
collate_fn
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.collate import collate_fn
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


###############################################################################
# Dummy Map
###############################################################################


class DummyLaneSegment:

    def __init__(self):

        import numpy as np

        self.centerline = np.array(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.is_intersection = False
        self.turn_direction = "NONE"
        self.has_traffic_control = False


class DummyMap:

    def __init__(self):

        self.city_lane_centerlines_dict = {
            "PIT": {
                0: DummyLaneSegment(),
                1: DummyLaneSegment(),
            },
            "MIA": {
                0: DummyLaneSegment(),
                1: DummyLaneSegment(),
            },
        }

    def get_lane_ids_in_xy_bbox(
        self,
        x,
        y,
        city,
        query_search_range_manhattan=50.0,
    ):

        return [0, 1]


###############################################################################
# Main
###############################################################################


def main():

    print("=" * 80)
    print("DSTNet Collate Test")
    print("=" * 80)
    print()

    ###########################################################################
    # Build dataset
    ###########################################################################

    parser = SceneParser(
        DummyMap(),
    )

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=50.0,
        lane_radius=50.0,
    )

    dataset = ArgoverseDataset(
        root="data/argoverse1/train",
        parser=parser,
        preprocessor=preprocessor,
    )

    print(dataset)
    print()

    ###########################################################################
    # DataLoader
    ###########################################################################

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    ###########################################################################
    # Metadata
    ###########################################################################

    print("=" * 80)
    print("Metadata")
    print("=" * 80)

    print("Sequence IDs :", batch["metadata"]["sequence_id"])
    print("Cities       :", batch["metadata"]["city"])
    print("Origin Shape :", batch["metadata"]["origin"].shape)
    print("Heading Shape:", batch["metadata"]["heading"].shape)

    ###########################################################################
    # Agents
    ###########################################################################

    print()
    print("=" * 80)
    print("Agent Tensors")
    print("=" * 80)

    for key, value in batch["agents"].items():

        print(f"{key:<20} {tuple(value.shape)}")

    ###########################################################################
    # Lanes
    ###########################################################################

    print()
    print("=" * 80)
    print("Lane Tensors")
    print("=" * 80)

    for key, value in batch["lanes"].items():

        print(f"{key:<20} {tuple(value.shape)}")

    ###########################################################################
    # Graphs
    ###########################################################################

    print()
    print("=" * 80)
    print("Graphs")
    print("=" * 80)

    print("Scenes :", len(batch["graphs"]))

    for i, graph in enumerate(batch["graphs"]):

        print()

        print(f"Scene {i}")

        print(
            "  Agent-Agent :",
            graph.agent_agent_edges.shape,
        )

        print(
            "  Lane-Lane  :",
            graph.lane_lane_edges.shape,
        )

        print(
            "  Lane-Agent :",
            graph.lane_agent_edges.shape,
        )

    ###########################################################################
    # Assertions
    ###########################################################################

    print()
    print("=" * 80)
    print("Running Assertions")
    print("=" * 80)

    assert batch["agents"]["observed"].ndim == 4
    assert batch["agents"]["future"].ndim == 4
    assert batch["agents"]["velocity"].ndim == 4
    assert batch["agents"]["acceleration"].ndim == 4

    assert batch["agents"]["heading"].ndim == 3
    assert batch["agents"]["speed"].ndim == 3

    assert batch["agents"]["observed_mask"].ndim == 3
    assert batch["agents"]["future_mask"].ndim == 3
    assert batch["agents"]["agent_mask"].ndim == 2

    assert batch["lanes"]["centerline"].ndim == 4
    assert batch["lanes"]["direction"].ndim == 4
    assert batch["lanes"]["mask"].ndim == 2

    print("✓ Tensor dimensions verified.")

    ###########################################################################

    print()
    print("=" * 80)
    print("Collate Test Passed")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
