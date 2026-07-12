"""
scripts.test_scene_data

Inspect one processed SceneData object.

This script is used to verify that the preprocessing pipeline
produces the expected data before batching.
"""

from datasets.argoverse_dataset import ArgoverseDataset
from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


###############################################################################
# Temporary Dummy Map
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
            "MIA": {
                0: DummyLaneSegment(),
                1: DummyLaneSegment(),
            },
            "PIT": {
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
    print("DSTNet SceneData Inspection")
    print("=" * 80)
    print()

    map_api = DummyMap()

    parser = SceneParser(
        map_api,
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

    print("Dataset Summary")
    print(dataset.summary())
    print()

    scene = dataset[0]

    print("=" * 80)
    print("Scene Summary")
    print("=" * 80)

    print(scene)
    print(scene.summary())

    print()

    print("=" * 80)
    print("Agents")
    print("=" * 80)

    print(f"Total Agents : {scene.num_agents}")
    print()

    for idx, agent in enumerate(scene.agents):

        print(
            f"[{idx:02d}] "
            f"Track={agent['track_id']:<8} "
            f"Category={agent['category']:<8} "
            f"Observed={agent['observed'].shape} "
            f"Future={agent['future'].shape} "
            f"Velocity={agent['velocity'].shape} "
            f"Heading={agent['heading'].shape}"
        )

    print()

    print("=" * 80)
    print("Lanes")
    print("=" * 80)

    print(f"Total Lanes : {scene.num_lanes}")
    print()

    for idx, lane in enumerate(scene.lanes):

        print(
            f"[{idx:02d}] "
            f"Lane={lane['lane_id']} "
            f"Centerline={lane['centerline'].shape} "
            f"Direction={lane['direction'].shape}"
        )

    print()

    print("=" * 80)
    print("Graph")
    print("=" * 80)

    print(
        "Agent-Agent Edges :",
        scene.graph.agent_agent_edges.shape,
    )

    print(
        "Lane-Lane Edges  :",
        scene.graph.lane_lane_edges.shape,
    )

    print(
        "Lane-Agent Edges :",
        scene.graph.lane_agent_edges.shape,
    )

    print()

    print("=" * 80)
    print("Inspection Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
