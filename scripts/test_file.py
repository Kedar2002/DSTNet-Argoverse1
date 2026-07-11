"""
Test the preprocessing pipeline.
"""

from pathlib import Path

from datasets.preprocess import ScenePreprocessor
from datasets.scene_parser import SceneParser


def main() -> None:

    ###########################################################################
    # Select one scene
    ###########################################################################

    csv_file = Path(
        "data/argoverse1/train/1.csv"   # <-- Replace with one of your files
    )

    ###########################################################################
    # Parse scene
    ###########################################################################

    parser = SceneParser(
        map_api=None,
    )

    scene = parser(csv_file)

    ###########################################################################
    # Preprocessor
    ###########################################################################

    preprocessor = ScenePreprocessor(
        observation_steps=20,
        prediction_steps=30,
        lane_sample_points=20,
        agent_radius=30.0,
        lane_radius=20.0,
    )

    processed = preprocessor.preprocess(scene)

    ###########################################################################
    # Print summary
    ###########################################################################

    print("=" * 70)
    print("Processed Scene")
    print("=" * 70)

    print()

    print("Sequence :", processed["metadata"].sequence_id)

    print("City     :", processed["metadata"].city)

    print()

    print("Origin")

    print(processed["origin"])

    print()

    print("Heading")

    print(processed["heading"])

    print()

    print("Number of Agents")

    print(len(processed["agents"]))

    print()

    print("Number of Lanes")

    print(len(processed["lanes"]))

    print()

    ###########################################################################
    # First agent
    ###########################################################################

    first = processed["agents"][0]

    print("=" * 70)
    print("First Agent")
    print("=" * 70)

    print("Track ID")

    print(first["track_id"])

    print()

    print("Observed")

    print(first["observed"].shape)

    print()

    print("Future")

    print(first["future"].shape)

    print()

    print("Velocity")

    print(first["velocity"].shape)

    print()

    print("Speed")

    print(first["speed"].shape)

    print()

    print("Acceleration")

    print(first["acceleration"].shape)

    print()

    print("Heading")

    print(first["heading"].shape)

    print()

    ###########################################################################
    # Graph
    ###########################################################################

    graph = processed["graph"]

    print("=" * 70)
    print("Graph")
    print("=" * 70)

    print()

    print(
        "Agent-Agent :",
        graph.agent_agent_edges.shape,
    )

    print(
        "Lane-Lane   :",
        graph.lane_lane_edges.shape,
    )

    print(
        "Lane-Agent  :",
        graph.lane_agent_edges.shape,
    )

    print()

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
