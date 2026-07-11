"""
Test the complete preprocessing pipeline.

Pipeline:
CSV
    ↓
SceneParser
    ↓
RawScene
    ↓
ScenePreprocessor
    ↓
SceneData
"""

from pathlib import Path

from datasets.preprocess import ScenePreprocessor
from datasets.scene_data import SceneData
from datasets.scene_parser import SceneParser


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:

    ###########################################################################
    # Scene
    ###########################################################################

    csv_file = Path(
        "data/argoverse1/train/1.csv"      # Replace with an existing scene
    )

    ###########################################################################
    # Parse
    ###########################################################################

    parser = SceneParser(
        map_api=None,
    )

    raw_scene = parser(csv_file)

    ###########################################################################
    # Preprocess
    ###########################################################################

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=30.0,
        lane_radius=20.0,
    )

    scene: SceneData = preprocessor.preprocess(raw_scene)

    ###########################################################################
    # Scene Summary
    ###########################################################################

    print_separator("Scene Summary")

    print(scene)

    print()

    print(scene.summary())

    ###########################################################################
    # Metadata
    ###########################################################################

    print_separator("Metadata")

    print("Sequence :", scene.sequence_id)
    print("City     :", scene.city)

    print()

    print("Origin")

    print(scene.origin)

    print()

    print("Heading")

    print(scene.heading)

    ###########################################################################
    # Statistics
    ###########################################################################

    print_separator("Statistics")

    print("Agents              :", scene.num_agents)
    print("Lanes               :", scene.num_lanes)

    print()

    print("Agent-Agent Edges   :", scene.num_agent_edges)
    print("Lane-Lane Edges     :", scene.num_lane_edges)
    print("Lane-Agent Edges    :", scene.num_lane_agent_edges)

    ###########################################################################
    # First Agent
    ###########################################################################

    if scene.num_agents > 0:

        agent = scene.agents[0]

        print_separator("First Agent")

        print("Track ID")

        print(agent["track_id"])

        print()

        print("Object Type")

        print(agent["object_type"])

        print()

        print("Category")

        print(agent["category"])

        print()

        print("Observed")

        print(agent["observed"].shape)

        print()

        print("Future")

        print(agent["future"].shape)

        print()

        print("Velocity")

        print(agent["velocity"].shape)

        print()

        print("Speed")

        print(agent["speed"].shape)

        print()

        print("Acceleration")

        print(agent["acceleration"].shape)

        print()

        print("Heading")

        print(agent["heading"].shape)

    ###########################################################################
    # First Lane
    ###########################################################################

    if scene.num_lanes > 0:

        lane = scene.lanes[0]

        print_separator("First Lane")

        print("Lane ID")

        print(lane["lane_id"])

        print()

        print("Centerline")

        print(lane["centerline"].shape)

        print()

        print("Direction")

        print(lane["direction"].shape)

        print()

        print("Intersection")

        print(lane["is_intersection"])

        print()

        print("Turn")

        print(lane["turn_direction"])

        print()

        print("Traffic Control")

        print(lane["has_traffic_control"])

    ###########################################################################
    # Graph
    ###########################################################################

    print_separator("Graph")

    print(
        "Agent-Agent :",
        scene.graph.agent_agent_edges.shape,
    )

    print(
        "Lane-Lane   :",
        scene.graph.lane_lane_edges.shape,
    )

    print(
        "Lane-Agent  :",
        scene.graph.lane_agent_edges.shape,
    )

    if scene.graph.agent_agent_edges.size > 0:

        print()

        print("First Agent-Agent Edges")

        print(scene.graph.agent_agent_edges[:, :20])

    print_separator("Done")

    print("Preprocessing pipeline executed successfully.")


if __name__ == "__main__":
    main()
