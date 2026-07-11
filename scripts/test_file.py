"""
Test GraphBuilder on an Argoverse 1 scene.

Currently, lane parsing is disabled because we are replacing
the deprecated argoverse-api with our own map loader.

Expected:
    - Agent-Agent graph is populated
    - Lane-Lane graph is empty
    - Lane-Agent graph is empty
"""

from pathlib import Path

from datasets.graph_builder import GraphBuilder
from datasets.scene_parser import SceneParser


def main() -> None:

    ###########################################################################
    # Select a scene
    ###########################################################################

    csv_file = Path(
        "data/argoverse1/train/1.csv"      # <-- Replace with one of your files
    )

    ###########################################################################
    # Parse scene
    ###########################################################################

    parser = SceneParser(map_api=None)

    scene = parser(csv_file)

    ###########################################################################
    # Build graph
    ###########################################################################

    builder = GraphBuilder(
        agent_radius=30.0,
        lane_radius=20.0,
    )

    graph = builder.build(scene)

    ###########################################################################
    # Print summary
    ###########################################################################

    print("=" * 70)
    print("Scene")
    print("=" * 70)

    print(scene)

    print()

    print(scene.summary())

    print()

    print(f"Tracks : {scene.num_tracks}")
    print(f"Lanes  : {scene.num_lanes}")

    print()

    print("=" * 70)
    print("Graph")
    print("=" * 70)

    print(f"Agent-Agent : {graph.agent_agent_edges.shape}")
    print(f"Lane-Lane   : {graph.lane_lane_edges.shape}")
    print(f"Lane-Agent  : {graph.lane_agent_edges.shape}")

    print()

    if graph.agent_agent_edges.size > 0:

        print("First 20 Agent-Agent Edges")

        print(graph.agent_agent_edges[:, :20])

    else:

        print("No Agent-Agent edges.")

    print()

    if graph.lane_lane_edges.size > 0:

        print("First 20 Lane-Lane Edges")

        print(graph.lane_lane_edges[:, :20])

    else:

        print("Lane graph empty (expected).")

    print()

    if graph.lane_agent_edges.size > 0:

        print("First 20 Lane-Agent Edges")

        print(graph.lane_agent_edges[:, :20])

    else:

        print("Lane-Agent graph empty (expected).")

    print()

    print("=" * 70)
    print("Statistics")
    print("=" * 70)

    print(
        "Agent-Agent edges:",
        graph.agent_agent_edges.shape[1],
    )

    print(
        "Lane-Lane edges:",
        graph.lane_lane_edges.shape[1],
    )

    print(
        "Lane-Agent edges:",
        graph.lane_agent_edges.shape[1],
    )


if __name__ == "__main__":
    main()
